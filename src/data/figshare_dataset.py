"""
Dataset loader for Figshare Brain Tumor Dataset (Cheng et al.).
Reads MATLAB .mat files (both v5/v7 via SciPy and v7.3 HDF5 via h5py).
"""

import os
import glob
import cv2
import numpy as np
import scipy.io as sio
import torch
from torch.utils.data import Dataset

try:
    import h5py
except ImportError:
    h5py = None


def read_cjdata(path: str) -> dict:
    """
    Read a single Figshare .mat file.
    Supports MATLAB v5/v7 formats (via scipy.io) and v7.3 HDF5 formats (via h5py).
    """
    try:
        c = sio.loadmat(path, squeeze_me=True, struct_as_record=False)["cjdata"]
        img = np.asarray(c.image)
        msk = np.asarray(c.tumorMask)
        lab = np.asarray(c.label).reshape(-1)[0]
        pid = np.asarray(c.PID).reshape(-1)[0]
        fmt = "scipy"
    except NotImplementedError:
        if h5py is None:
            raise RuntimeError(f"MATLAB v7.3 file encountered at {path}, but h5py is not installed.")
        with h5py.File(path, "r") as f:
            c = f["cjdata"]
            # HDF5 stores MATLAB arrays in transposed row-major order
            img = np.array(c["image"]).T
            msk = np.array(c["tumorMask"]).T
            lab = np.array(c["label"]).reshape(-1)[0]
            pid = "".join(chr(int(v)) for v in np.array(c["PID"]).reshape(-1))
        fmt = "h5py"

    return {
        "image": img,
        "mask": msk,
        "label": int(lab) - 1,  # Convert 1-indexed (1: meningioma, 2: glioma, 3: pituitary) to 0-indexed
        "pid": str(pid).strip(),
        "format": fmt,
    }


def preprocess_mri_slice(im: np.ndarray, size: int = 256) -> np.ndarray:
    """
    Per-slice min-max intensity normalization scaled to uint8 [0, 255],
    followed by area-interpolation resizing.
    """
    im = im.astype(np.float32)
    lo, hi = float(im.min()), float(im.max())
    if hi <= lo:
        return np.zeros((size, size), dtype=np.uint8)
    im8 = np.round((im - lo) / (hi - lo) * 255.0).astype(np.uint8)
    interp = cv2.INTER_AREA if im8.shape[0] > size else cv2.INTER_LINEAR
    return cv2.resize(im8, (size, size), interpolation=interp)


def preprocess_tumor_mask(msk: np.ndarray, size: int = 256) -> np.ndarray:
    """
    Binarizes mask (> 0.5) and resizes using nearest-neighbor interpolation.
    """
    m = (msk > 0.5).astype(np.uint8)
    return cv2.resize(m, (size, size), interpolation=cv2.INTER_NEAREST)


class BrainTumorMRIDataset(Dataset):
    """
    PyTorch Dataset for brain tumor MRI slices.
    Normalizes images using (x/255.0 - 0.449) / 0.226 and replicates to 3 channels.
    """
    def __init__(self, images: np.ndarray, masks: np.ndarray, labels: np.ndarray, transform=None):
        self.images = images  # uint8 (N, H, W)
        self.masks = masks    # uint8 (N, H, W)
        self.labels = labels  # int64 (N,)
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        im = self.images[idx]
        m = self.masks[idx]
        y = self.labels[idx]

        if self.transform is not None:
            augmented = self.transform(image=im, mask=m)
            im = augmented["image"]
            m = augmented["mask"]

        # Grayscale normalized tensor: (x / 255 - mean) / std
        im_norm = (im.astype(np.float32) / 255.0 - 0.449) / 0.226
        im_tensor = torch.from_numpy(im_norm).unsqueeze(0)  # (1, H, W)
        mask_tensor = torch.from_numpy(m.astype(np.float32)).unsqueeze(0)  # (1, H, W)
        has_mask = torch.tensor(m.sum() > 0, dtype=torch.bool)
        label_tensor = torch.tensor(y, dtype=torch.long)

        return im_tensor, mask_tensor, label_tensor, has_mask
