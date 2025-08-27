from pathlib import Path

import numpy as np
from torch.utils.data import Dataset
import torch
import torchvision
from torchvision import transforms
from PIL import Image

from source.data.augs import get_color_distortion
from source.data.datasets.objs.npdataset import NumpyDataset, PairDataset
from source.data.augs import simclr_augmentation


def get_dsprites_pair(root, split="train", imsize=64, hflip=False):
    path = Path(root, f"colored_on_grayscale_{split}.npz")
    return PairDataset(path, transform=simclr_augmentation(imsize=imsize, hflip=hflip))


def get_dsprites(root, split="train", imsize=64):
    path = Path(root, f"colored_on_grayscale_{split}.npz")
    return NumpyDataset(
        path,
        transform=transforms.Compose(
            [transforms.Resize(imsize), transforms.ToTensor()]
        ),
    )
