from pathlib import Path

import numpy as np
from torch.utils.data import Dataset
import torch
import torchvision
from torchvision import transforms
from PIL import Image

from source.data.augs import simclr_augmentation
from source.data.datasets.objs.npdataset import NumpyDataset, PairDataset


def get_tetrominoes_pair(root, split="train", imsize=32, hflip=False):
    path = Path(root, f"tetrominoes_{split}.npz")
    return PairDataset(path, transform=simclr_augmentation(imsize=imsize, hflip=hflip))


def get_tetrominoes(root, split="train"):
    path = Path(root, f"tetrominoes_{split}.npz")
    return NumpyDataset(path, transform=transforms.ToTensor())
