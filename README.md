# Neuroscience-Inspired Encoding and Learning: A Path to Robust Representation Learning

## Train and Evaluate an AKOrN encoder

This repository contains a comprehensive framework for training an [AKOrN](https://github.com/autonomousvision/akorn) backbone architecture with different SSL architectures. The framework supports multiple datasets and SSL methods. An evaluation framework corresponding to the one of [RobustBench](https://robustbench.github.io/index.html) is also available and ready to run.

## Results obtained 

<img width="1028" height="334" alt="results" src="https://github.com/user-attachments/assets/187b5100-92d0-4521-a133-b1a91b0ecb33" />

## 🚀 Quick Start


### Environment Setup

#### Option 1: Using Conda (Recommended)
```bash
# Create a new conda environment with Python 3.9
conda create -n robust_akorn python=3.10 -y

# Activate the environment
conda activate robust_akorn

# Install packages from requirements.txt using pip
pip install -r requirements.txt
```

#### Option 2: Using Existing Environment
```bash
# If you already have a conda environment, just activate it and install
conda activate your_env_name
pip install -r requirements.txt
```

### 📁 Directory Structure
```
akorn-main/
├── train.py          # Main training script
├── attack.py  # Evaluation script with adversarial attacks
├── requirements.txt          # Python dependencies
├── ssl_architectures/        # SSL method implementations
├── source/                   # Core model implementations
├── augmentations/            # Data augmentation strategies
└── models/                   # Saved model checkpoints (created automatically)
```

## 🔬 Running Experiments

### Step 1: Training a Model

The main training script `train.py` supports:
- **Datasets**: CIFAR-10, CIFAR-100, Tiny ImageNet.
- **SSL Methods**: PhiNet, SimCLR, SimSiam, BYOL.
- **Backbone**: AKOrN (Artificial Kuramoto Oscillatory Network).

#### Basic Training Command
```bash
python3 train.py \
    --dataset cifar10 \
    --run_pretraining \
    --run_finetuning \
    --load_from_pretrain \
    --ssl_method phinet \
    --backbone akorn \
    --pretrain_epochs 50 \
    --finetune_epochs 50 \
    --pretrain_bs 512 \
    --finetune_bs 512 \
    --pretrain_lr 1e-4 \
    --finetune_lr 1e-4 \
    --out_dim 2048 \
    --ch 128 \
    --randomness True \
    --n 4 \
    --T 5 \
    --L 3
```

### Step 2: Evaluating Model Robustness

After training, evaluate the model's adversarial robustness using AutoAttack:

```bash
python -u attack.py \
    --ssl_method phinet \
    --backbone akorn \
    --attack_bs 100 \
    --pretrain_epochs 50 \
    --finetune_epochs 50 \
    --ch 128 \
    --dataset cifar10 \
    --randomness True \
    --n 4 \
    --T 5 \
    --L 3
```

## 🎛️ Key Parameters

### Training Parameters
- `--dataset`: Dataset choice (`cifar10`, `cifar100`, `tiny-imagenet`).
- `--ssl_method`: SSL method (`phinet`, `simclr`, `simsiam`, `byol`).
- `--backbone`: Neural network backbone (`akorn`).
- `--pretrain_epochs`: Number of self-supervised pre-training epochs.
- `--finetune_epochs`: Number of supervised fine-tuning epochs.
- `--pretrain_bs`/`--finetune_bs`: Batch sizes for pre-training/fine-tuning.
- `--ch`: Number of channels in AKOrN backbone.
- `--out_dim`: Output dimension of projection head.

## 📊 Expected Outputs

### Training Output
- **Pre-trained model**: `models/model_{ssl_method}_{backbone}_{dataset}_ch{ch}_pretrain{epochs}_pretrained.pth`
- **Fine-tuned model**: `models/model_{ssl_method}_{backbone}_{dataset}_ch{ch}_pretrain{epochs}_finetuned.pth`

### Evaluation Output
The evaluation script will output:
- **Clean Accuracy**: Model accuracy on unperturbed test data.
- **Robust Accuracy**: Model accuracy under AutoAttack adversarial examples.

Example output:
```
Loaded fine-tuned model from models/model_phinet_akorn_cifar10_ch64_pretrain100_finetuned.pth
Loading all test images into memory...
Running AutoAttack with version='rand'...
Clean accuracy: 85.2%
Robust accuracy: 67.8%
```

## 🛠️ Troubleshooting

### Common Issue

**Dataset Loading Error**
- For Tiny ImageNet, ensure correct path: `--tiny_imagenet_path /path/to/tiny-imagenet-200`
- In `train.py` and `attack.py`, change the current path `/work/YamadaU/mhabibi/` into the correct one.
