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
from source.data.augs import augmentation_strong

def get_attack_transform(dataset, strategy):
    if strategy == 'finetuning':
        image_size = 64 if dataset == 'tiny-imagenet' else 32
        return augmentation_strong(imsize=image_size)
    if strategy == 'eval':
        return transforms.Compose([transforms.ToTensor()])
    raise ValueError(f"Unknown attack augmentation strategy: {strategy}")

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

    parser = argparse.ArgumentParser(description='Evaluate a fine-tuned model with AutoAttack.')
    parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'cifar100', 'tiny-imagenet'], 
                        help='Dataset to use (cifar10, cifar100, or tiny-imagenet).')
    parser.add_argument('--data_root', type=str, default='data',
                        help='Root directory where datasets and models are stored.')
    parser.add_argument('--tiny_imagenet_path', type=str, default=None,
                        help='Path to Tiny ImageNet dataset directory (defaults to data_root/tiny-imagenet-200).')
    parser.add_argument('--pretrain_epochs', type=int, default=400, help='Number of epochs for pre-training.')
    parser.add_argument('--ssl_method', type=str, required=True, choices=['phinet', 'simclr', 'simsiam', 'byol'],
                        help='SSL method used for pre-training.')
    parser.add_argument('--backbone', type=str, required=True, choices=['akorn'], help='Backbone architecture.')
    parser.add_argument('--model_path', type=str, default=None,
                        help='Base path where the fine-tuned model is saved (defaults to data_root/models/model).')
    parser.add_argument('--ch', type=int, default=128, help='Channels for AKOrN backbone.')
    parser.add_argument('--batch_size', type=int, default=512, help='Batch size for loading data.')
    parser.add_argument('--attack_bs', type=int, default=100, help='Batch size for AutoAttack.')
    parser.add_argument('--randomness', type=bool, default=True, help='Random noise in the forward of AKOrN')
    parser.add_argument('--n', type=int, default=2, help='Oscillator dimensions')
    parser.add_argument('--T', type=int, default=3, help='Timesteps')
    parser.add_argument('--L', type=int, default=3, help='Number of layers')
    parser.add_argument('--finetune_epochs', type=int, default=400, help='Number of epochs for fine-tuning.')
    parser.add_argument('--mse_loss_ratio', type=float, default=None,
                        help='MSE loss ratio used during training (only needed for filename suffixes).')
    parser.add_argument('--ori_loss_ratio', type=float, default=None,
                        help='Orientation loss ratio used during training (only needed for filename suffixes).')
    parser.add_argument('--beta', type=float, default=None,
                        help='EMA beta used during training (only needed for filename suffixes).')
    parser.add_argument('--attack_log', type=str, default=None,
                        help='Optional path to append AutoAttack logs (defaults to attack_logs/<config>.txt).')
    parser.add_argument('--attack_aug', type=str, default='eval', choices=['eval', 'finetuning'],
                        help='Augmentation strategy when loading evaluation data.')
    parser.add_argument('--seed', type=int, default=0, help='Random seed for reproducibility.')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='Number of DataLoader workers (set 0 when multiprocessing is disallowed).')
    parser.add_argument('--max_eval', type=int, default=None,
                        help='Limit number of evaluation samples to speed up smoke tests.')

    args = parser.parse_args()
    set_seed(args.seed)

    args.data_root = os.path.abspath(os.path.expanduser(args.data_root))
    os.makedirs(args.data_root, exist_ok=True)
    if args.tiny_imagenet_path is None:
        args.tiny_imagenet_path = os.path.join(args.data_root, 'tiny-imagenet-200')
    else:
        args.tiny_imagenet_path = os.path.abspath(os.path.expanduser(args.tiny_imagenet_path))
    if args.model_path is None:
        args.model_path = os.path.join(args.data_root, 'models', 'model')
    else:
        args.model_path = os.path.abspath(os.path.expanduser(args.model_path))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if args.dataset == 'cifar10':
        num_classes = 10
    elif args.dataset == 'cifar100':
        num_classes = 100
    elif args.dataset == 'tiny-imagenet':
        num_classes = 200
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    def _format_suffix(value):
        text = str(value)
        if '.' in text:
            text = text.replace('.', 'p')
        return text

    model_name = f"{args.model_path}_{args.ssl_method}_{args.backbone}_{args.dataset}"
    if args.backbone == 'akorn':
        model_name += f"_ch{args.ch}_n{args.n}_T{args.T}_L{args.L}"
    if args.mse_loss_ratio is not None:
        model_name += f"_mse{_format_suffix(args.mse_loss_ratio)}"
    if args.ori_loss_ratio is not None:
        model_name += f"_ori{_format_suffix(args.ori_loss_ratio)}"
    if args.beta is not None:
        model_name += f"_beta{_format_suffix(args.beta)}"
    model_name += f"_pretrain{args.pretrain_epochs}"
    model_filename = f"{model_name}_finetuned{args.finetune_epochs}.pth"

    attack_id = (
        f"{args.ssl_method}_{args.backbone}_{args.dataset}"
        f"_ch{args.ch}_n{args.n}_T{args.T}_L{args.L}"
    )
    if args.mse_loss_ratio is not None:
        attack_id += f"_mse{_format_suffix(args.mse_loss_ratio)}"
    if args.ori_loss_ratio is not None:
        attack_id += f"_ori{_format_suffix(args.ori_loss_ratio)}"
    if args.beta is not None:
        attack_id += f"_beta{_format_suffix(args.beta)}"
    attack_id += f"_pretrain{args.pretrain_epochs}_finetune{args.finetune_epochs}"

    project_root = os.path.dirname(os.path.abspath(__file__))
    if args.attack_log is None:
        log_dir = os.path.join(project_root, 'attack_logs')
        os.makedirs(log_dir, exist_ok=True)
        args.attack_log = os.path.join(log_dir, f"{attack_id}.txt")
    else:
        args.attack_log = os.path.abspath(os.path.expanduser(args.attack_log))
    print(f"AutoAttack log will be stored at: {args.attack_log}")

    net = get_backbone(args.backbone, ch=args.ch, n=args.n, L=args.L, 
                       T=args.T, num_classes=num_classes, randomness=args.randomness)
    net.load_state_dict(torch.load(model_filename, map_location=device))
    net.randomness = args.randomness
    net.to(device)
    net.eval()
    print(f"Loaded fine-tuned model from {model_filename}")

    eval_net = LogWrapper(EnsembleWrapper([net] * 8, apply_softmax=True))
    eval_net.eval()

    attack_transform = get_attack_transform(args.dataset, args.attack_aug)
    if args.dataset == 'cifar10':
        testset = torchvision.datasets.CIFAR10(
            root=args.data_root, train=False, download=True, transform=attack_transform
        )
    elif args.dataset == 'cifar100':
        testset = torchvision.datasets.CIFAR100(
            root=args.data_root, train=False, download=True, transform=attack_transform
        )
    elif args.dataset == 'tiny-imagenet':
        testset = TinyImageNetDataset(
            args.tiny_imagenet_path, split='val', transform=attack_transform
        )
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    dataset_len = len(testset)
    if args.max_eval is not None:
        subset_len = min(dataset_len, args.max_eval)
        if subset_len <= 0:
            print('ERROR: max_eval is non-positive.')
            return
        testset = torch.utils.data.Subset(testset, list(range(subset_len)))
        print(f"Restricting evaluation to first {subset_len} samples (original {dataset_len}).")
    else:
        print(f"Testset length: {dataset_len}")

    if len(testset) == 0:
        print('ERROR: Testset is empty! Check your dataset path and structure.')
        return

    testloader = torch.utils.data.DataLoader(
        testset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )

    print("Loading all test images into memory...")
    images = []
    labels = []
    for img, lbl in testloader:
        images.append(img)
        labels.append(lbl)
    if len(images) == 0:
        print('ERROR: No batches loaded from testloader!')
        return
    images = torch.cat(images, 0).to(device)
    labels = torch.cat(labels, 0).to(device)

    print("Running AutoAttack with version='rand'...")
    eps = 8/255
    autoattack(
        eval_net,
        images,
        labels,
        epsilon=eps,
        version='rand',
        bs=args.attack_bs,
        log_file=args.attack_log,
    )

if __name__ == '__main__':
    main()
