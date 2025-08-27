import torch
import torchvision
from source.data.augs import get_color_distortion
from torchvision.datasets import ImageNet
from torchvision import transforms
from PIL import Image


class ImageNetPair(ImageNet):
    """Generate mini-batche pairs on CIFAR10 training set."""

    def __getitem__(self, idx):
        path, target = self.imgs[idx][0], self.targets[idx]
        img = self.loader(path)
        imgs = [self.transform(img), self.transform(img)]
        return torch.stack(imgs)


def get_imagenet(
    root,
    split="train",
    transform=transforms.Compose(
        [
            transforms.CenterCrop(256),
            transforms.ToTensor(),
        ]
    ),
):
    return ImageNet(
        root=root,
        split=split,
        transform=transform,
    )


def get_imagenet_pair(root, split="train", imsize=256, hflip=False):
    from source.data.augs import simclr_augmentation

    return ImageNetPair(
        root=root,
        split=split,
        transform=simclr_augmentation(imsize=imsize, hflip=hflip),
    )
