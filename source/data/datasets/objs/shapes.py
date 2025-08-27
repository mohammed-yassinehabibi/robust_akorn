from pathlib import Path

import numpy as np
from torch.utils.data import Dataset
import torch
import torchvision
from torchvision import transforms
from PIL import Image

from source.data.augs import simclr_augmentation
from source.data.datasets.objs.npdataset import NumpyDataset, PairDataset


def get_shapes_pair(root, split="train", imsize=40):

    path = Path(root, f"{split}.npz")
    return PairDataset(path, transform=simclr_augmentation(imsize=imsize, hflip=False))


def get_shapes(root, split="train", imsize=40):
    path = Path(root, f"{split}.npz")
    return NumpyDataset(
        path,
        transform=transforms.Compose(
            [
                transforms.Resize((imsize, imsize)),
                transforms.ToTensor(),
            ]
        ),
    )
