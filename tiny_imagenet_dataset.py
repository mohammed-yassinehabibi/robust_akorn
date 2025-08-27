import os
import torch
from torch.utils.data import Dataset
from torchvision.datasets.folder import ImageFolder
from PIL import Image
import pandas as pd


class TinyImageNetDataset(Dataset):
    """
    Custom Dataset for Tiny ImageNet with support for both training and validation sets.
    Tiny ImageNet has 200 classes, 64x64 images.
    """
    
    def __init__(self, root, split='train', transform=None):
        """
        Args:
            root (str): Root directory of the Tiny ImageNet dataset
            split (str): 'train' or 'val'
            transform (callable, optional): Optional transform to be applied on a sample
        """
        self.root = root
        self.split = split
        self.transform = transform
        
        if split == 'train':
            self.data_dir = os.path.join(root, 'train')
            # Use ImageFolder for training data since it's organized in class folders
            self.dataset = ImageFolder(self.data_dir, transform=None)
            self.samples = self.dataset.samples
            self.class_to_idx = self.dataset.class_to_idx
            self.classes = self.dataset.classes
        
        elif split == 'val':
            self.data_dir = os.path.join(root, 'val')
            self.images_dir = os.path.join(self.data_dir, 'images')
            self.annotations_file = os.path.join(self.data_dir, 'val_annotations.txt')
            
            # Load validation annotations
            self.val_annotations = self._load_val_annotations()
            
            # Create class mapping from training set to maintain consistency
            train_dir = os.path.join(root, 'train')
            train_dataset = ImageFolder(train_dir)
            self.class_to_idx = train_dataset.class_to_idx
            self.classes = train_dataset.classes
            
            # Prepare validation samples
            self.samples = self._prepare_val_samples()
        
        else:
            raise ValueError(f"Invalid split: {split}. Must be 'train' or 'val'")
    
    def _load_val_annotations(self):
        """Load validation annotations from val_annotations.txt"""
        annotations = {}
        with open(self.annotations_file, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    img_name = parts[0]
                    class_id = parts[1]
                    annotations[img_name] = class_id
        return annotations
    
    def _prepare_val_samples(self):
        """Prepare validation samples list"""
        samples = []
        for img_name, class_id in self.val_annotations.items():
            img_path = os.path.join(self.images_dir, img_name)
            if os.path.exists(img_path) and class_id in self.class_to_idx:
                class_idx = self.class_to_idx[class_id]
                samples.append((img_path, class_idx))
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        """
        Args:
            idx (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[idx]
        
        # Load image
        with open(path, 'rb') as f:
            img = Image.open(f)
            img = img.convert('RGB')
        
        # Apply transforms
        if self.transform is not None:
            img = self.transform(img)
        
        return img, target


def get_tiny_imagenet_dataloaders(root, batch_size, train_transform=None, val_transform=None, num_workers=1):
    """
    Convenience function to get Tiny ImageNet dataloaders
    
    Args:
        root (str): Root directory of Tiny ImageNet dataset
        batch_size (int): Batch size for dataloaders
        train_transform: Transform for training data
        val_transform: Transform for validation data
        num_workers (int): Number of worker processes for data loading
        
    Returns:
        tuple: (train_loader, val_loader)
    """
    from torch.utils.data import DataLoader
    
    # Create datasets
    train_dataset = TinyImageNetDataset(root, split='train', transform=train_transform)
    val_dataset = TinyImageNetDataset(root, split='val', transform=val_transform)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader


if __name__ == "__main__":
    # Test the dataset
    from torchvision import transforms
    
    root = '/work/YamadaU/mhabibi/tiny-imagenet-200'
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # Test training dataset
    train_dataset = TinyImageNetDataset(root, split='train', transform=transform)
    print(f"Training dataset size: {len(train_dataset)}")
    print(f"Number of classes: {len(train_dataset.classes)}")
    
    # Test validation dataset
    val_dataset = TinyImageNetDataset(root, split='val', transform=transform)
    print(f"Validation dataset size: {len(val_dataset)}")
    
    # Test a sample
    img, label = train_dataset[0]
    print(f"Sample image shape: {img.shape}")
    print(f"Sample label: {label}")
