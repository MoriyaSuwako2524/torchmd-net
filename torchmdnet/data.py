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

