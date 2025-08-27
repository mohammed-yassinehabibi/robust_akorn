
from pathlib import Path

import numpy as np
from torch.utils.data import Dataset
import torch
import torchvision
from torchvision import transforms
from PIL import Image

from source.data.augs import get_color_distortion


class NumpyDataset(Dataset):
    """NpzDataset: loads a npz file as dataset."""

    def __init__(self, filename, transform=torchvision.transforms.ToTensor()):
        super().__init__()

        dataset = np.load(filename)
        self.images = dataset["images"].astype(np.float32)
        if self.images.shape[1] == 1:
            self.images = np.repeat(self.images, 3, axis=1) 
        self.pixelwise_instance_labels = dataset["labels"]

        if "class_labels" in dataset:
            self.class_labels = dataset["class_labels"]
        else:
            self.class_labels = None

        if "pixelwise_class_labels" in dataset:
            self.pixelwise_class_labels = dataset["pixelwise_class_labels"]
        else:
            self.pixelwise_class_labels = None

        self.transform = transform

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx):
        img = self.images[idx]  # {"input_images": self.images[idx]}
        img = np.transpose(img, (1, 2, 0))
        img = (255 * img).astype(np.uint8)
        img = Image.fromarray(img)  # .convert('RGB')
        labels = {"pixelwise_instance_labels": self.pixelwise_instance_labels[idx]}

        if self.class_labels is not None:
            labels["class_labels"] = self.class_labels[idx]
            labels["pixelwise_class_labels"] = self.pixelwise_class_labels[idx]
        return self.transform(img), labels


class PairDataset(NumpyDataset):
    """Generate mini-batche pairs on CIFAR10 training set."""

    def __getitem__(self, idx):
        img = self.images[idx]
        img = np.transpose(img, (1, 2, 0))
        img = (255 * img).astype(np.uint8)
        img = Image.fromarray(img)  # .convert('RGB')
        imgs = [self.transform(img), self.transform(img)]
        return torch.stack(imgs)  # stack a positive pair
    