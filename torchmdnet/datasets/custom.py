import glob
import numpy as np
import torch
from torch_geometric.data import Dataset, Data

__all__ = ["Custom"]


class Custom(Dataset):
    """
    Universal custom dataset for TorchMD-Net supporting arbitrary predicted properties.

    Example:
        >>> pred_files = {"y": "energy_*.npy", "neg_dy": "force_*.npy", "vec": "dipole_*.npy"}
        >>> dataset = Custom(coordglob="coord_*.npy", embedglob="embed_*.npy",
        ...                  pred_file_dict=pred_files)
    """

    def __init__(
        self,
        coordglob,
        embedglob,
        pred_file_dict=None,
        preload_memory_limit=1024,
        transform=None,
        pre_transform=None,
        pre_filter=None,
    ):
        super().__init__(None, transform, pre_transform, pre_filter)

        # --- Base required fields ---
        self.fields = [
            ("pos", "pos", torch.float32),
            ("z", "types", torch.long),
        ]
        self.files = {
            "pos": sorted(glob.glob(coordglob)),
            "z": sorted(glob.glob(embedglob)),
        }

        assert len(self.files["pos"]) == len(self.files["z"]), (
            f"Number of coordinate files {len(self.files['pos'])} "
            f"does not match number of embed files {len(self.files['z'])}."
        )

        # --- Dynamically register predicted properties ---
        self.pred_file_dict = pred_file_dict or {}
        for key, pattern in self.pred_file_dict.items():
            files = sorted(glob.glob(pattern))
            if not files:
                raise ValueError(f"No matching files found for pattern '{pattern}' (key='{key}')")
            assert len(files) == len(self.files["pos"]), (
                f"Number of coordinate files {len(self.files['pos'])} "
                f"does not match number of {key} files {len(files)}."
            )
            self.files[key] = files
            self.fields.append((key, key, torch.float32))

        print(f"Registered fields: {[f[0] for f in self.fields]}")
        print("Number of files:", len(self.files["pos"]))

        # --- Indexing and caching setup ---
        self.cached = False
        total_data_size = self._initialize_index()
        data_size_limit = preload_memory_limit * 1024 * 1024

        if total_data_size < data_size_limit:
            self.cached = True
            print(f"Preloading dataset (size {total_data_size / 1024**2:.2f} MB)")
            self._preload_data()
        else:
            self._store_numpy_memmaps()

    # ==============================================================
    # === Core loading utilities ===
    # ==============================================================

    def _preload_data(self):
        self.stored_data = {}
        self.stored_data["pos"] = [torch.from_numpy(np.load(f)) for f in self.files["pos"]]
        self.stored_data["z"] = [
            torch.from_numpy(np.load(f).astype(int))
            .unsqueeze(0)
            .expand(self.stored_data["pos"][i].shape[0], -1)
            for i, f in enumerate(self.files["z"])
        ]

        for key in self.pred_file_dict.keys():
            self.stored_data[key] = [torch.from_numpy(np.load(f)) for f in self.files[key]]

    def _store_numpy_memmaps(self):
        self.stored_data = {}
        self.stored_data["pos"] = [np.load(f, mmap_mode="r") for f in self.files["pos"]]
        self.stored_data["z"] = []
        for i, f in enumerate(self.files["z"]):
            arr = np.load(f).astype(int)
            broadcasted = np.broadcast_to(arr[np.newaxis, :], (len(self.stored_data["pos"][i]), arr.shape[0]))
            self.stored_data["z"].append(broadcasted)
        for key in self.pred_file_dict.keys():
            self.stored_data[key] = [np.load(f, mmap_mode="r") for f in self.files[key]]

    def _initialize_index(self):
        """Build index and perform consistency checks."""
        self.index = []
        nfiles = len(self.files["pos"])
        total_data_size = 0
        for i in range(nfiles):
            pos_data = np.load(self.files["pos"][i], mmap_mode="r")
            z_data = np.load(self.files["z"][i]).astype(int)
            size = pos_data.shape[0]
            total_data_size += pos_data.nbytes + z_data.nbytes
            self.index.extend(list(zip([i] * size, range(size))))
            assert pos_data.shape[1] == z_data.shape[0], (
                f"Atom count mismatch in file {i}: pos {pos_data.shape[1]} vs z {z_data.shape[0]}"
            )

            # Check all predicted quantities
            for key in self.pred_file_dict.keys():
                arr = np.load(self.files[key][i], mmap_mode="r")
                total_data_size += arr.nbytes
                # Allow both (N, 1), (N,3), or (N_atoms,3)
                if arr.shape[0] != size:
                    raise ValueError(
                        f"Inconsistent frame count for '{key}' in file {i}: {arr.shape[0]} vs {size}"
                    )

        print(f"Total frames: {len(self.index)}, total data size: {total_data_size/1024**2:.2f} MB")
        return total_data_size



    def get(self, idx):
        fileid, frameid = self.index[idx]
        data = Data()
        for key, _, _ in self.fields:
            arr = self.stored_data[key][fileid][frameid]
            arr = arr if self.cached else torch.from_numpy(np.array(arr))
            data[key] = arr
        return data

    def len(self):
        return len(self.index)
