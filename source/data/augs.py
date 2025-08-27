import torch
from torchvision import transforms
import torchvision.transforms as T
from PIL import Image


def gauss_noise_tensor(sigma=0.1):
    def fn(img):
        out = img + sigma * torch.randn_like(img)
        out = torch.clamp(out, 0, 1)  #  pixel space is [0, 1]
        return out

    return fn


def augmentation_strong(imsize=32):
    transform_aug = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomResizedCrop(imsize, scale=(0.2, 1.0)),
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
            transforms.AugMix(),
            transforms.ToTensor(),
        ]
    )
    return transform_aug


def simclr_augmentation(imsize, hflip=False):
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(imsize),
            transforms.RandomHorizontalFlip(0.5) if hflip else lambda x: x,
            get_color_distortion(s=0.5),
            transforms.ToTensor(),
        ]
    )


def random_Linf_noise(trnsfms: transforms.Compose = None, epsilon=64 / 255):
    if trnsfms is None:
        trnsfms = transforms.Compose([transforms.ToTensor()])

    randeps = torch.rand(1).item() * epsilon

    def fn(x):
        x = x + randeps * torch.randn_like(x).sign()
        return torch.clamp(x, 0, 1)

    trnsfms.transforms.append(fn)
    return trnsfms


def get_color_distortion(s=0.5):
    # s is the strength of color distortion
    color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
    rnd_color_jitter = transforms.RandomApply([color_jitter], p=0.8)
    rnd_gray = transforms.RandomGrayscale(p=0.2)
    color_distort = transforms.Compose([rnd_color_jitter, rnd_gray])
    return color_distort


class PhiNetTransform():
    def __init__(self, image_size, mean_std=([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])):
        image_size = 224 if image_size is None else image_size # by default simsiam use image size 224
        p_blur = 0.5 if image_size > 32 else 0 # exclude cifar
        # the paper didn't specify this, feel free to change this value
        # I use the setting from simclr which is 50% chance applying the gaussian blur
        # the 32 is prepared for cifar training where they disabled gaussian blur
        self.transform = T.Compose([
            T.RandomResizedCrop(image_size, scale=(0.2, 1.0)),
            T.RandomHorizontalFlip(),
            T.RandomApply([T.ColorJitter(0.4,0.4,0.4,0.1)], p=0.8),
            T.RandomGrayscale(p=0.2),
            T.RandomApply([T.GaussianBlur(kernel_size=image_size//20*2+1, sigma=(0.1, 2.0))], p=p_blur),
            T.ToTensor(),
            T.Normalize(*mean_std)
        ])
        self.transform_ori = T.Compose([
                T.Resize(int(image_size*(8/7)), interpolation=Image.BICUBIC), # 224 -> 256 
                T.CenterCrop(image_size),
                T.ToTensor(),
                T.Normalize(*mean_std)
            ])
    def __call__(self, x):
        x1 = self.transform(x)
        x2 = self.transform(x)
        x_ori = self.transform_ori(x)
        #x_ori = self.transform(x)
        return x1, x2, x_ori
    
    