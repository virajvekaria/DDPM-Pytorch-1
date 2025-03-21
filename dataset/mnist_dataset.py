import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm


class MnistDataset(Dataset):
    """
    Custom dataset class for loading .npy files of shape (1, 64, 64).
    Assumes all .npy files are stored in a single directory with no labels.
    """
    def __init__(self, im_path):
        """
        Initialize the dataset.

        :param im_path: Path to the directory containing .npy files.
        """
        assert os.path.exists(im_path), f"Data path {im_path} does not exist"
        self.im_path = im_path
        self.files = sorted(glob.glob(os.path.join(im_path, "*.npy")))
        print(f"Found {len(self.files)} .npy files")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        npy_file = self.files[index]
        array = np.load(npy_file)  # shape (1, 64, 64)
        tensor = torch.from_numpy(array).float()

        # Optional: normalize from [0, 1] to [-1, 1] if data is already in 0-1 range
        tensor = (2 * tensor) - 1

        return tensor
