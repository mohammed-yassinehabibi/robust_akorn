# Copied from https://github.com/loeweX/RotatingFeatures/blob/main/codebase/data/PascalDataset.py

import os
from typing import Tuple, Dict

import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset


class PascalDataset(Dataset):
    def __init__(self, root_dir, partition: str, transform, transform_label) -> None:
        """
        Initialize the Pascal VOC 2012 Dataset.
        See http://host.robots.ox.ac.uk/pascal/VOC/ for more information about the dataset.

        We make use of the “trainaug” variant of this dataset, an unofficial split consisting of 10,582 images,
        which includes 1,464 images from the original segmentation train set and 9,118 images from the
        Semantic Boundaries dataset.

        Args:
            opt (DictConfig): Configuration options.
            partition (str): Dataset partition ("train", "val", or "test").
        """
        super(PascalDataset, self).__init__()

        self.partition = partition
        self.to_tensor = transforms.ToTensor()
        if self.partition == "train":
            self.partition = "trainaug"
        self.transform = transform
        self.transform_label = transform_label

        # As is common in the literature, we test our model on the validation set of the Pascal VOC dataset.
        # For validation, create a train/validation split of the official training set manually and
        # adjust the code accordingly.
        if self.partition == "test":
            self.partition = "val"

        # Load Pascal dataset.
        partition_dir = os.path.join(
            root_dir, "ImageSets", "Segmentation", f"{self.partition}.txt"
        )

        with open(partition_dir) as f:
            file_names = [x.strip() for x in f.readlines()]

        self.images = [
            os.path.join(root_dir, "JPEGImages", f"{x}.jpg") for x in file_names
        ]
        self.pixelwise_class_labels = [
            os.path.join(root_dir, "SegmentationClass", f"{x}.png") for x in file_names
        ]
        self.pixelwise_instance_labels = [
            os.path.join(root_dir, "SegmentationObject", f"{x}.png") for x in file_names
        ]

        # Normalize input images using mean and standard deviation of ImageNet.
        # self.normalize = transforms.Normalize(
        #    (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
        # )

        self.num_classes = 20

    def __len__(self) -> int:
        """
        Get the number of images in the dataset.

        Returns:
            int: Number of images.
        """
        return len(self.images)

    @staticmethod
    def _preprocess_pascal_labels(labels: torch.Tensor) -> torch.Tensor:
        """
        Preprocess Pascal VOC labels by converting to integer 255-scale and
        marking object boundaries as "ignore" label.

        Args:
            labels (torch.Tensor): The input labels.

        Returns:
            torch.Tensor: The preprocessed labels.
        """
        labels = labels * 255
        labels[labels == 255] = -1  # "Ignore" label throughout the codebase is -1.
        return labels

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Get an item from the dataset.

        Args:
            idx (int): Index of the item to retrieve.

        Returns:
            tuple: A tuple containing the input image and corresponding gt_labels.
        """
        input_image = Image.open(self.images[idx]).convert("RGB")
        pixelwise_class_labels = Image.open(self.pixelwise_class_labels[idx])

        try:
            pixelwise_instance_labels = Image.open(self.pixelwise_instance_labels[idx])
        except FileNotFoundError as e:
            # Instance labels are not available for all images in the trainaug set.
            if self.partition != "trainaug":
                raise FileNotFoundError(
                    "Instance labels should only be missing for the trainaug partition."
                ) from e
            # Create an empty target.
            pixelwise_instance_labels = Image.new(
                "L", (pixelwise_class_labels.width, pixelwise_class_labels.height), 0
            )

        pixelwise_class_labels = self._preprocess_pascal_labels(
            self.to_tensor(pixelwise_class_labels)
        )
        pixelwise_instance_labels = self._preprocess_pascal_labels(
            self.to_tensor(pixelwise_instance_labels)
        )

        input_image = self.transform(input_image)
        pixelwise_instance_labels = self.transform_label(pixelwise_instance_labels)[0]
        pixelwise_class_labels = self.transform_label(pixelwise_class_labels)[0]

        labels = {
            "pixelwise_class_labels": pixelwise_class_labels,
            "pixelwise_instance_labels": pixelwise_instance_labels,
        }
        return input_image, labels


def get_pascal(root, split="train", imsize=256, imsize_label=320):
    transform = transforms.Compose(
        [
            transforms.Resize(
                imsize, interpolation=transforms.InterpolationMode.NEAREST
            ),
            transforms.CenterCrop(imsize),
            transforms.ToTensor(),
        ]
    )

    transform_label = transforms.Compose(
        [
            transforms.Resize(
                imsize_label, interpolation=transforms.InterpolationMode.NEAREST
            ),
            transforms.CenterCrop(imsize_label),
        ]
    )
    return PascalDataset(
        root, split, transform=transform, transform_label=transform_label
    )
