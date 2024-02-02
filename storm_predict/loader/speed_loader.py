import os
import json
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, ConcatDataset, DataLoader, Subset, random_split
from torchvision.transforms import ToTensor
from matplotlib import pyplot as plt
from PIL import Image
from torchvision import transforms, models
from tqdm import tqdm
from livelossplot import PlotLosses

class StormDataset(Dataset):
    """
    Dataset class for loading storm images and wind speed labels.

    Parameters
    ----------
    root_dir : str
        Path to the root directory containing images and labels.
    start_idx : int, optional
        Starting index for data selection (default is 0).
    end_idx : int, optional
        Ending index for data selection (default is None).
    transform : callable, optional
        Optional transform to be applied to images.

    Attributes
    ----------
    root_dir : str
        Path to the root directory containing images and labels.
    transform : callable, optional
        Optional transform to be applied to images.
    data_path : list
        List of sorted image file names in the specified range.

    Methods
    -------
    __len__()
        Returns the length of the dataset.
    __getitem__(idx)
        Retrieves and returns a sample from the dataset.
    """
    def __init__(self, root_dir, start_idx=0, end_idx=None, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.data_path = sorted([file for file in os.listdir(root_dir) if file.endswith('.jpg')])[start_idx:end_idx]

    def __len__(self):
        """
        Returns the length of the dataset.

        Returns
        -------
        int
            Length of the dataset.
        """
        return len(self.data_path) - 2

    def __getitem__(self, idx):
        """
        Retrieves and returns a sample from the dataset.

        Parameters
        ----------
        idx : int
            Index of the sample.

        Returns
        -------
        tuple
            Tuple containing the stacked image and the target wind speed.
        """
        img1 = Image.open(os.path.join(self.root_dir, self.data_path[idx])).convert('L')
        img2 = Image.open(os.path.join(self.root_dir, self.data_path[idx+1])).convert('L')
        img3 = Image.open(os.path.join(self.root_dir, self.data_path[idx+2])).convert('L')

        # Image transformation
        img1 = self.transform(img1)
        img2 = self.transform(img2)
        img3 = self.transform(img3)

        label1_path = os.path.join(self.root_dir, self.data_path[idx+2].replace('.jpg', '_label.json'))
        label3_path = os.path.join(self.root_dir, self.data_path[idx+2].replace('.jpg', '_label.json'))
        with open(label1_path, 'r') as f:
            label1 = json.load(f)['wind_speed']
        with open(label3_path, 'r') as f:
            label3 = json.load(f)['wind_speed']

        # Combine images into a 4-channel tensor
        eye_tensor = torch.eye(128).unsqueeze(0)
        stacked_img = torch.cat([float(label1)*eye_tensor, img1, img2, img3], dim=0)

        # Load targer to this stacked_img
        target = torch.tensor(float(label3))

        return stacked_img, target
    
def get_train_val_loader(dataset, train_size=0.8):
    """
    Split the dataset into training and validation sets and create DataLoader instances.

    Parameters
    ----------
    dataset : Dataset
        Dataset to be split.
    train_size : float, optional
        Fraction of the dataset to be used for training (default is 0.8).

    Returns
    -------
    DataLoader
        DataLoader for the training set.
    DataLoader
        DataLoader for the validation set.
    """
    # Split train set and validation set
    train_size = int(train_size * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # initialize dataloader
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    return train_loader, val_loader