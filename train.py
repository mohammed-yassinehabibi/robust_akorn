import argparse
import os
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import random

# Local imports
from source.models.classification.knet import AKOrN
from torchvision.models import resnet50
from source.data.augs import augmentation_strong
# Assuming ssl_architectures.py contains SimCLR and SimSiam implementations
from ssl_architectures import simclr, simsiam, byol, phinet
from augmentations import simclr_aug, simsiam_aug, phinet_aug, byol_aug, eval_aug
from tiny_imagenet_dataset import TinyImageNetDataset, get_tiny_imagenet_dataloaders

def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_config():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='Modular SSL Training Script')

    # Run control
    parser.add_argument('--run_pretraining', action='store_true', help='Run the pre-training phase.')
    parser.add_argument('--run_finetuning', action='store_true', help='Run the fine-tuning phase.')
    parser.add_argument('--load_from_pretrain', action='store_true', help='Load pre-trained weights for fine-tuning.')

    # Dataset configuration
    parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'cifar100', 'tiny-imagenet'], 
                        help='Dataset to use (cifar10, cifar100, or tiny-imagenet).')
    parser.add_argument('--tiny_imagenet_path', type=str, default='/work/YamadaU/mhabibi/tiny-imagenet-200',
                        help='Path to Tiny ImageNet dataset directory.')

    # Model and SSL configuration
    parser.add_argument('--ssl_method', type=str, default='phinet', 
                        choices=['phinet', 'simclr', 'simsiam', 'byol', 'swav'], 
                        help='SSL method for pre-training.')
    parser.add_argument('--backbone', type=str, default='akorn', choices=['akorn'], help='Backbone architecture.')
    
    # Data augmentation for fine-tuning
    parser.add_argument('--aug_strategy', type=str, default='finetuning', choices=['eval', 'simclr', 'phinet', 'finetuning'], help='Data augmentation strategy for fine-tuning.')

    # Paths
    parser.add_argument('--model_path', type=str, default='/work/YamadaU/mhabibi/models/model', help='Base path to save/load models.')
    parser.add_argument('--log_dir', type=str, default='runs', help='Directory for TensorBoard logs.')

    # Pre-training parameters
    parser.add_argument('--pretrain_epochs', type=int, default=100, help='Number of epochs for pre-training.')
    parser.add_argument('--pretrain_bs', type=int, default=512, help='Batch size for pre-training.')
    parser.add_argument('--pretrain_lr', type=float, default=1e-4, help='Learning rate for pre-training.')

    # Fine-tuning parameters
    parser.add_argument('--finetune_epochs', type=int, default=400, help='Number of epochs for fine-tuning.')
    parser.add_argument('--finetune_bs', type=int, default=512, help='Batch size for fine-tuning.')
    parser.add_argument('--finetune_lr', type=float, default=1e-4, help='Learning rate for fine-tuning.')

    # Backbone-specific parameters
    parser.add_argument('--out_dim', type=int, default=2048, help='Output dimension for the projection head.')
    parser.add_argument('--ch', type=int, default=128, help='Channels for AKOrN backbone.')

    return parser.parse_args()

def get_transforms(strategy, image_size=32):
    """Return the appropriate data transforms based on the augmentation strategy."""
    if strategy == 'phinet':
        return phinet_aug.PhiNetTransform(image_size=image_size)
    elif strategy == 'simclr':
        return simclr_aug.SimCLRTransform(image_size=image_size)
    elif strategy == 'simsiam':
        return simsiam_aug.SimSiamTransform(image_size=image_size)
    elif strategy == 'byol':
        return byol_aug.BYOL_transform(image_size=image_size)
    elif strategy == 'eval':
        return transforms.Compose([transforms.ToTensor()])
    elif strategy == 'finetuning':
        # For Tiny ImageNet, use different image size
        imsize = image_size
        return augmentation_strong(imsize=imsize)
    else:
        raise ValueError(f"Unknown augmentation strategy: {strategy}")

def get_dataloaders(args, for_pretraining=True):
    """Return the appropriate dataloaders."""
    # Determine number of classes and image size based on dataset
    if args.dataset == 'cifar10':
        num_classes = 10
        image_size = 32
    elif args.dataset == 'cifar100':
        num_classes = 100
        image_size = 32
    elif args.dataset == 'tiny-imagenet':
        num_classes = 200
        image_size = 64
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")
    
    if for_pretraining:
        # Pre-training uses the transform associated with the SSL method
        transform = get_transforms(args.ssl_method, image_size=image_size)
        
        if args.dataset == 'cifar10':
            trainset = torchvision.datasets.CIFAR10(root='/work/YamadaU/mhabibi/data', train=True, download=True, transform=transform)
        elif args.dataset == 'cifar100':
            trainset = torchvision.datasets.CIFAR100(root='/work/YamadaU/mhabibi/data', train=True, download=True, transform=transform)
        elif args.dataset == 'tiny-imagenet':
            trainset = TinyImageNetDataset(args.tiny_imagenet_path, split='train', transform=transform)
            
        train_loader = DataLoader(trainset, batch_size=args.pretrain_bs, shuffle=True, num_workers=1, pin_memory=True)
        return train_loader
    else:
        # Fine-tuning can use a different augmentation strategy
        transform_train = get_transforms(args.aug_strategy, image_size=image_size)
        transform_test = get_transforms('eval', image_size=image_size)
        
        if args.dataset == 'cifar10':
            trainset = torchvision.datasets.CIFAR10(root='/work/YamadaU/mhabibi/data', train=True, download=True, transform=transform_train)
            testset = torchvision.datasets.CIFAR10(root='/work/YamadaU/mhabibi/data', train=False, download=True, transform=transform_test)
        elif args.dataset == 'cifar100':
            trainset = torchvision.datasets.CIFAR100(root='/work/YamadaU/mhabibi/data', train=True, download=True, transform=transform_train)
            testset = torchvision.datasets.CIFAR100(root='/work/YamadaU/mhabibi/data', train=False, download=True, transform=transform_test)
        elif args.dataset == 'tiny-imagenet':
            trainset = TinyImageNetDataset(args.tiny_imagenet_path, split='train', transform=transform_train)
            testset = TinyImageNetDataset(args.tiny_imagenet_path, split='val', transform=transform_test)
            
        train_loader = DataLoader(trainset, batch_size=args.finetune_bs, shuffle=True, num_workers=1, pin_memory=True)
        test_loader = DataLoader(testset, batch_size=args.finetune_bs, shuffle=False, num_workers=1, pin_memory=True)
        return train_loader, test_loader

def get_backbone(backbone_name, ch=128, num_classes=10):
    """Return the backbone model."""
    if backbone_name == 'akorn':
        return AKOrN(n=2, ch=ch, out_classes=num_classes, L=3, T=3, J="conv", ksizes=[9,7,5], ro_ksize=3, ro_N=2,
                     norm="bn", c_norm="gn", gamma=1.0, use_omega=True, init_omg=1.0, global_omg=True,
                     learn_omg=True, ensemble=1)
    elif backbone_name == 'resnet':
        model = resnet50(pretrained=False)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    else:
        raise ValueError(f"Unknown backbone: {backbone_name}")

def get_ssl_model(args, device):
    """Return the full SSL model, assuming it has a `.backbone` attribute."""
    if args.dataset == 'cifar10':
        num_classes = 10
    elif args.dataset == 'cifar100':
        num_classes = 100
    elif args.dataset == 'tiny-imagenet':
        num_classes = 200
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")
        
    backbone = get_backbone(args.backbone, args.ch, num_classes)
    if args.ssl_method == 'phinet':
        model = phinet.XPhiNetTF(backbone=backbone, out_dim=args.out_dim)
    elif args.ssl_method == 'simclr':
        model = simclr.SimCLR(backbone)
    elif args.ssl_method == 'simsiam':
        model = simsiam.SimSiam(backbone)
    elif args.ssl_method == 'byol':
        model = byol.BYOL(backbone)
    else:
        raise ValueError(f"Unknown SSL method: {args.ssl_method}")
    return model.to(device)

def pretrain(args, model, train_loader, device):
    """Pre-training loop."""
    print(f"--- Starting Pre-training with {args.ssl_method} ---")
    optimizer = optim.Adam(model.parameters(), lr=args.pretrain_lr)
    
    # Ensure model directory exists
    model_dir = os.path.dirname(args.model_path)
    os.makedirs(model_dir, exist_ok=True)
    
    for epoch in range(args.pretrain_epochs):
        model.train()
        total_loss = 0
        loop = tqdm(train_loader, desc=f"Pre-train Epoch {epoch+1}/{args.pretrain_epochs}", leave=False)

        for data in loop:
            optimizer.zero_grad()
            
            if args.ssl_method == 'phinet':
                x1, x2, x_ori = data[0]
                x1, x2, x_ori = x1.to(device), x2.to(device), x_ori.to(device)
                outputs = model(x1, x2, x_ori)
                loss = outputs['loss'].mean()
            elif args.ssl_method in ['simclr', 'simsiam', 'byol']:
                images, _ = data
                # These augs return a list of two tensors
                p1, p2 = images[0].to(device), images[1].to(device)
                outputs = model(p1, p2)
                loss = outputs['loss']

            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        print(f"Epoch [{epoch+1}/{args.pretrain_epochs}], Avg Loss: {total_loss / len(train_loader):.4f}")

    model_name = f"{args.model_path}_{args.ssl_method}_{args.backbone}_{args.dataset}"
    if args.backbone == 'akorn':
        model_name += f"_ch{args.ch}"
    model_name += f"_pretrain{args.pretrain_epochs}"
    save_path = f"{model_name}_pretrained.pth"
    
    # Check if the model has a slow encoder (like in PhiNet or BYOL)
    if hasattr(model, 'slow_encoder'):
        # Save the slow encoder, which is typically used for inference
        torch.save(model.slow_encoder[0].state_dict(), save_path)
        print(f"Pre-trained slow encoder saved to {save_path}")
    elif hasattr(model, 'backbone'):
        # For other models, save the main backbone
        torch.save(model.backbone.state_dict(), save_path)
        print(f"Pre-trained backbone saved to {save_path}")
    else:
        # Fallback for models that might not have a .backbone or .slow_encoder attribute
        torch.save(model.state_dict(), save_path)
        print(f"Entire pre-trained model saved to {save_path}")

def finetune(args, model, train_loader, test_loader, device):
    """Fine-tuning loop for classification."""
    print("--- Starting Fine-tuning ---")
    
    # Ensure model directory exists
    model_dir = os.path.dirname(args.model_path)
    os.makedirs(model_dir, exist_ok=True)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.finetune_lr)
    
    jobdir = os.path.join(args.log_dir, f"{args.ssl_method}_{args.backbone}_{args.dataset}_finetune")
    writer = SummaryWriter(jobdir)

    for epoch in range(args.finetune_epochs):
        model.train()
        train_loss, n_correct, n_total = 0.0, 0, 0
        
        loop = tqdm(train_loader, desc=f"Finetune Epoch {epoch+1}/{args.finetune_epochs}", leave=False)
        for imgs, labels in loop:
            imgs, labels = imgs.to(device), labels.to(device)

            preds = model(imgs)
            loss = criterion(preds, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * imgs.size(0)
            n_correct += (preds.argmax(dim=1) == labels).sum().item()
            n_total += imgs.size(0)
            loop.set_postfix(loss=loss.item(), acc=n_correct/n_total)

        train_acc = n_correct / n_total
        print(f'Epoch {epoch+1}: Train Loss: {train_loss/n_total:.4f}, Train Acc: {train_acc:.4f}')
        writer.add_scalar('Loss/train', train_loss/n_total, epoch)
        writer.add_scalar('Accuracy/train', train_acc, epoch)
    
    model_name = f"{args.model_path}_{args.ssl_method}_{args.backbone}_{args.dataset}"
    if args.backbone == 'akorn':
        model_name += f"_ch{args.ch}"
    model_name += f"_pretrain{args.pretrain_epochs}"
    save_path = f"{model_name}_finetuned.pth"
    torch.save(model.state_dict(), save_path)
    print(f"Fine-tuned model saved to {save_path}")

def main():

    def set_seed(seed):
        random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    set_seed(0)
    
    """Main execution function."""
    args = get_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Configuration: {args}")

    if args.run_pretraining:
        print("--- Starting Pre-training Phase ---")
        ssl_model = get_ssl_model(args, device)
        pretrain_loader = get_dataloaders(args, for_pretraining=True)
        pretrain(args, ssl_model, pretrain_loader, device)

    if args.run_finetuning:
        print("--- Starting Fine-tuning Phase ---")
        if args.dataset == 'cifar10':
            num_classes = 10
        elif args.dataset == 'cifar100':
            num_classes = 100
        elif args.dataset == 'tiny-imagenet':
            num_classes = 200
        else:
            raise ValueError(f"Unsupported dataset: {args.dataset}")
            
        finetune_model = get_backbone(args.backbone, args.ch, num_classes)
        
        if args.load_from_pretrain:
            model_name = f"{args.model_path}_{args.ssl_method}_{args.backbone}_{args.dataset}"
            if args.backbone == 'akorn':
                model_name += f"_ch{args.ch}"
            model_name += f"_pretrain{args.pretrain_epochs}"
            load_path = f"{model_name}_pretrained.pth"
            try:
                finetune_model.load_state_dict(torch.load(load_path, map_location=device))
                print(f"Loaded pre-trained backbone from {load_path}")
            except FileNotFoundError:
                print(f"ERROR: Pre-trained model not found at {load_path}. Exiting.")
                return
        
        # Attach a new linear classifier for fine-tuning
        if hasattr(finetune_model, 'fc'): # For ResNet-style models
             num_ftrs = finetune_model.fc.in_features
             finetune_model.fc = nn.Linear(num_ftrs, num_classes)
        elif hasattr(finetune_model, 'classifier'): # For ConvNeXt
             num_ftrs = finetune_model.classifier.in_features
             finetune_model.classifier = nn.Linear(num_ftrs, num_classes)
        elif hasattr(finetune_model, 'linear'): # For other models
             num_ftrs = finetune_model.linear.in_features
             finetune_model.linear = nn.Linear(num_ftrs, num_classes)
        
        finetune_model = finetune_model.to(device)

        train_loader, test_loader = get_dataloaders(args, for_pretraining=False)
        finetune(args, finetune_model, train_loader, test_loader, device)

if __name__ == '__main__':
    main()