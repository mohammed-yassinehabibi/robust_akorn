from pathlib import Path

from torchvision import transforms

from source.data.augs import simclr_augmentation
from source.data.datasets.objs.npdataset import NumpyDataset, PairDataset


def get_clevr_pair(root, split="train", imsize=128, hflip=False):
    path = Path(root, f"clevr_{split}.npz")
    return PairDataset(path, transform=simclr_augmentation(imsize=imsize, hflip=hflip))


def get_clevr(root, split="train", imsize=128):
    path = Path(root, f"clevr_{split}.npz")
    transform = transforms.Compose(
        [
            transforms.Resize(imsize),
            transforms.ToTensor(),
        ]
    )
    return NumpyDataset(path, transform=transform)
