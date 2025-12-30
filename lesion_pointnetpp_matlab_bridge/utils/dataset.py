import os
import numpy as np
import torch
from torch.utils.data import Dataset

TARGET_N = 8192

class LesionSegDataset(Dataset):
    def __init__(self, list_file: str, root_dir: str = "data", has_label: bool = True, augment: bool = False):
        self.root_dir = root_dir
        self.has_label = has_label
        self.augment = augment
        with open(list_file, "r") as f:
            self.files = [ln.strip() for ln in f if ln.strip()]

    def __len__(self):
        return len(self.files)

    def _augment_xyz(self, xyz: np.ndarray) -> np.ndarray:
        # rotate around Z + jitter
        theta = np.random.uniform(0, 2*np.pi)
        c, s = np.cos(theta), np.sin(theta)
        R = np.array([[c, -s, 0],
                      [s,  c, 0],
                      [0,  0, 1]], dtype=np.float32)
        xyz = xyz @ R.T
        sigma, clip = 0.01, 0.05
        jitter = np.clip(sigma*np.random.randn(*xyz.shape), -clip, clip).astype(np.float32)
        return xyz + jitter

    def __getitem__(self, idx: int):
        npz_path = os.path.join(self.root_dir, self.files[idx])
        data = np.load(npz_path)
        xyz = data["xyz"].astype(np.float32)  # (N,3)
        assert xyz.shape == (TARGET_N, 3), f"Expected (8192,3), got {xyz.shape} in {npz_path}"

        if self.augment:
            xyz = self._augment_xyz(xyz)

        points = torch.from_numpy(xyz.T)  # (3,N)

        if self.has_label:
            label = data["label"].astype(np.int64)  # (N,)
        else:
            label = np.zeros((TARGET_N,), dtype=np.int64)
        label = torch.from_numpy(label)

        return points, label
