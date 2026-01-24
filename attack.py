import torch
import torchvision
import argparse
import random
import os
from torchvision import transforms

# Local imports from your project structure
from source.models.classification.knet import AKOrN
from source.evals.classification.adv_attacks import autoattack
from tiny_imagenet_dataset import TinyImageNetDataset

def get_backbone(backbone_name, ch=128, n=2, L=3, T=3, num_classes=10, randomness=True):
    """Return the backbone model, similar to train_cifar10.py."""
    if backbone_name == 'akorn':
        return AKOrN(n=n, ch=ch, out_classes=num_classes, L=L, T=T, J="conv", ksizes=[9,7,5], ro_ksize=3, ro_N=2,
                     norm="bn", c_norm="gn", gamma=1.0, use_omega=True, init_omg=1.0, global_omg=True,
                     learn_omg=True, ensemble=1, randomness=randomness)
    else:
        raise ValueError(f"Unknown backbone: {backbone_name}")

class LogWrapper(torch.nn.Module):
    """Wraps a model to return log probabilities, as expected by AutoAttack."""
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        out = self.net(x)
        return torch.log(out)

class EnsembleWrapper(torch.nn.Module):
    """Wraps a model to create an ensemble, averaging predictions."""
    def __init__(self, nets, apply_softmax=False):
        super().__init__()
        self.nets = torch.nn.ModuleList(nets)
        self.apply_softmax = apply_softmax

    def forward(self, x):
        out = 0
        for net in self.nets:
            pred = net(x)
            if self.apply_softmax:
                pred = torch.softmax(pred, dim=1)
            out += pred
        out /= len(self.nets)
        return out

def main():

    def set_seed(seed):
        random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    #set_seed(0)
    
    parser = argparse.ArgumentParser(description='Evaluate a fine-tuned model with AutoAttack.')
    parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'cifar100', 'tiny-imagenet'], 
                        help='Dataset to use (cifar10, cifar100, or tiny-imagenet).')
    parser.add_argument('--tiny_imagenet_path', type=str, default='/work/YamadaU/mhabibi/tiny-imagenet-200',
                        help='Path to Tiny ImageNet dataset directory.')
    parser.add_argument('--pretrain_epochs', type=int, default=400, help='Number of epochs for pre-training.')
    parser.add_argument('--ssl_method', type=str, required=True, choices=['phinet', 'simclr', 'simsiam', 'byol'], help='SSL method used for pre-training.')
    parser.add_argument('--backbone', type=str, required=True, choices=['akorn'], help='Backbone architecture.')
    parser.add_argument('--model_path', type=str, default='/work/YamadaU/mhabibi/models/model', help='Base path where the fine-tuned model is saved.')
    parser.add_argument('--ch', type=int, default=128, help='Channels for AKOrN backbone.')
    parser.add_argument('--batch_size', type=int, default=512, help='Batch size for loading data.')
    parser.add_argument('--attack_bs', type=int, default=100, help='Batch size for the AutoAttack.')
    parser.add_argument('--randomness', type=bool, default=True, help='Random noise in the forward of AKOrN')
    parser.add_argument('--n', type=int, default=2, help='occilator dimensions')
    parser.add_argument('--T', type=int, default=3, help='timesteps')
    parser.add_argument('--L', type=int, default=3, help='num of layers')
    parser.add_argument('--finetune_epochs', type=int, default=400, help='Number of epochs for fine-tuning.')

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Determine number of classes based on dataset
    if args.dataset == 'cifar10':
        num_classes = 10
    elif args.dataset == 'cifar100':
        num_classes = 100
    elif args.dataset == 'tiny-imagenet':
        num_classes = 200
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    # Construct the model path from arguments
    model_name = f"{args.model_path}_{args.ssl_method}_{args.backbone}_{args.dataset}"
    if args.backbone == 'akorn':
        model_name += f"_ch{args.ch}_n{args.n}_T{args.T}_L{args.L}"
    model_name += f"_pretrain{args.pretrain_epochs}"
    model_filename = f"{model_name}_finetuned.pth"

    # --- Model Loading ---
    net = get_backbone(args.backbone, ch=args.ch, n=args.n, L=args.L, 
                       T=args.T, num_classes=num_classes, randomness=args.randomness)
    net.load_state_dict(torch.load(model_filename, map_location=device))
    net.randomness = args.randomness
    net.to(device)
    net.eval()
    print(f"Loaded fine-tuned model from {model_filename}")

    # Wrap the model for evaluation
    # AutoAttack expects log-probabilities, so we use LogWrapper after softmax.
    eval_net = LogWrapper(EnsembleWrapper([net] * 8, apply_softmax=True))
    eval_net.eval()

    # --- Data Loading ---
    _transforms = transforms.Compose([transforms.ToTensor()])
    if args.dataset == 'cifar10':
        testset = torchvision.datasets.CIFAR10(
            root="/work/YamadaU/mhabibi/data", train=False, download=True, transform=_transforms
        )
    elif args.dataset == 'cifar100':
        testset = torchvision.datasets.CIFAR100(
            root="/work/YamadaU/mhabibi/data", train=False, download=True, transform=_transforms
        )
    elif args.dataset == 'tiny-imagenet':
        testset = TinyImageNetDataset(
            args.tiny_imagenet_path, split='val', transform=_transforms
        )
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    print(f"Testset length: {len(testset)}")
    if len(testset) == 0:
        print('ERROR: Testset is empty! Check your dataset path and structure.')
        return

    testloader = torch.utils.data.DataLoader(
        testset, batch_size=args.batch_size, shuffle=False, num_workers=1
    )

    # --- Evaluation ---
    print("Loading all test images into memory...")
    images = []
    labels = []
    for img, lbl in testloader:
        print(f"Loaded batch with shape: {img.shape}")
        images.append(img)
        labels.append(lbl)
    if len(images) == 0:
        print('ERROR: No batches loaded from testloader!')
        return
    images = torch.cat(images, 0).to(device)
    labels = torch.cat(labels, 0).to(device)

    print("Running AutoAttack with version='rand'...")
    # Note: AutoAttack standardly uses L_inf norm with eps=8/255 for CIFAR-10/CIFAR-100
    # For Tiny ImageNet, we also use eps=8/255 but note that images are 64x64
    eps = 8/255
    autoattack(eval_net, images, labels, epsilon=eps, version='rand', bs=args.attack_bs)

if __name__ == '__main__':
    main()
