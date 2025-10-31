# Copyright Universitat Pompeu Fabra 2020-2023  https://www.compscience.org
# Distributed under the MIT License.
# (See accompanying file README.md file or copy at http://opensource.org/licenses/MIT)

from os.path import join
from tqdm import tqdm
import torch
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader
from lightning import LightningDataModule
from lightning_utilities.core.rank_zero import rank_zero_warn
from torchmdnet import datasets
from torchmdnet.utils import make_splits, MissingEnergyException
from torchmdnet.models.utils import scatter
import warnings

class DataModule(LightningDataModule):
    """A LightningDataModule that supports dynamic Custom datasets with arbitrary predicted properties."""

    def __init__(self, hparams, dataset=None):
        super(DataModule, self).__init__()
        self.save_hyperparameters(hparams)
        self._mean, self._std = None, None
        self._saved_dataloaders = {}
        self.dataset = dataset

    def setup(self, stage):
        # ------------------------------------------------------------
        # 1. Construct dataset
        # ------------------------------------------------------------
        if self.dataset is None:
            if self.hparams["dataset"] == "Custom":
                # --- use new dynamic interface ---
                from torchmdnet.datasets import Custom

                pred_file_dict = getattr(self.hparams, "pred_file_dict", None)
                if pred_file_dict is None:
                    raise ValueError(
                        "For Custom dataset, pred_file_dict must be provided, e.g. {'y':'energy_*.npy','neg_dy':'force_*.npy'}"
                    )

                self.dataset = Custom(
                    coordglob=self.hparams["coord_files"],
                    embedglob=self.hparams["embed_files"],
                    pred_file_dict=pred_file_dict,
                    preload_memory_limit=self.hparams["dataset_preload_limit"],
                )

                print(f"[DataModule] Loaded Custom dataset with fields: {list(self.dataset.files.keys())}")
            else:
                # --- standard TorchMD datasets ---
                dataset_arg = {}
                if self.hparams.get("dataset_arg") is not None:
                    dataset_arg = self.hparams["dataset_arg"]
                if self.hparams["dataset"] == "HDF5":
                    dataset_arg["dataset_preload_limit"] = self.hparams["dataset_preload_limit"]
                self.dataset = getattr(datasets, self.hparams["dataset"])(
                    self.hparams["dataset_root"], **dataset_arg
                )

        # ------------------------------------------------------------
        # 2. Split indices
        # ------------------------------------------------------------
        self.idx_train, self.idx_val, self.idx_test = make_splits(
            len(self.dataset),
            self.hparams["train_size"],
            self.hparams["val_size"],
            self.hparams["test_size"],
            self.hparams["seed"],
            join(self.hparams["log_dir"], "splits.npz"),
            self.hparams["splits"],
        )
        print(f"[DataModule] Splits: train={len(self.idx_train)}, val={len(self.idx_val)}, test={len(self.idx_test)}")

        self.train_dataset = Subset(self.dataset, self.idx_train)
        self.val_dataset = Subset(self.dataset, self.idx_val)
        self.test_dataset = Subset(self.dataset, self.idx_test)

        # ------------------------------------------------------------
        # 3. Optional standardization (deprecated)
        # ------------------------------------------------------------
        if self.hparams.get("standardize", False):
            warnings.warn(
                "The standardize option is deprecated and will be removed in the future.",
                DeprecationWarning,
            )
            self._standardize()
    def train_dataloader(self):
        return self._get_dataloader(self.train_dataset, "train")

    def val_dataloader(self):
        loaders = [self._get_dataloader(self.val_dataset, "val")]
        # To allow to report the performance on the testing dataset during training
        # we send the trainer two dataloaders every few steps and modify the
        # validation step to understand the second dataloader as test data.
        if self._is_test_during_training_epoch():
            loaders.append(self._get_dataloader(self.test_dataset, "test"))
        return loaders

    def test_dataloader(self):
        return self._get_dataloader(self.test_dataset, "test")

    @property
    def atomref(self):
        """Returns the atomref of the dataset if it has one, otherwise None."""
        if hasattr(self.dataset, "get_atomref"):
            return self.dataset.get_atomref()
        return None

    @property
    def mean(self):
        """Returns the mean of the dataset if it has one, otherwise None."""
        return self._mean

    @property
    def std(self):
        """Returns the standard deviation of the dataset if it has one, otherwise None."""
        return self._std

    def _is_test_during_training_epoch(self):
        return (
            len(self.test_dataset) > 0
            and self.hparams["test_interval"] > 0
            and self.trainer.current_epoch > 0
            and self.trainer.current_epoch % self.hparams["test_interval"] == 0
        )

    def _get_dataloader(self, dataset, stage, store_dataloader=True):
        if stage in self._saved_dataloaders and store_dataloader:
            return self._saved_dataloaders[stage]

        if stage == "train":
            batch_size = self.hparams["batch_size"]
        elif stage in ["val", "test"]:
            batch_size = self.hparams["inference_batch_size"]

        shuffle = stage == "train"
        dl = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            num_workers=self.hparams["num_workers"],
            persistent_workers=False,
            pin_memory=True,
            shuffle=shuffle,
        )

        if store_dataloader:
            self._saved_dataloaders[stage] = dl
        return dl

    def _standardize(self):
        """Compute mean/std for each non-derivative property (e.g. y, charge)."""
        keys_to_standardize = [
            k for k in self.dataset.files.keys()
            if not any(s in k.lower() for s in ["neg_dy","dy", "grad","pos","z","force"])
        ]
        print(f"[DataModule] Computing mean/std for: {keys_to_standardize}")
    
        stats = {}
        data_loader = self._get_dataloader(self.train_dataset, "val", store_dataloader=False)
    
        for key in keys_to_standardize:
            ys = []
            for batch in tqdm(data_loader, desc=f"computing mean/std for {key}"):
                if not hasattr(batch, key) or getattr(batch, key) is None:
                    continue
                yval = getattr(batch, key)
                if yval.ndim > 1:
                    # flatten over atoms if per-atom quantity
                    yval = yval.reshape(-1, yval.shape[-1] if yval.ndim > 1 else 1)
                ys.append(yval.detach().cpu())
            if len(ys) == 0:
                continue
            ys = torch.cat(ys, dim=0)
            print(key)
            stats[key] = {
                "mean": ys.mean(dim=0),
                "std": ys.std(dim=0)
            }
            print(f"  {key}: mean={stats[key]['mean']}, std={stats[key]['std']}")
        
        self._mean = {k: v["mean"] for k, v in stats.items()}
        self._std = {k: v["std"] for k, v in stats.items()}


