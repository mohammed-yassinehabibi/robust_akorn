import copy
import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.transforms.v2 import AugMix
import torch.optim as optim
from tqdm import tqdm
from torchvision.datasets import ImageFolder
import random
from torchvision.datasets import CIFAR10
from source.models.classification.knet import AKOrN
from ssl_architectures.phinet import XPhiNetTF
from torch.utils.tensorboard import SummaryWriter
from source.data.augs import augmentation_strong, simclr_augmentation, PhiNetTransform
from source.evals.classification.adv_attacks import (
    fgsm_attack,
    pgd_linf_attack,
    autoattack,
    random_attack,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size=512
print("device :", device)

def simclr(zs, temperature=1.0, normalize=True, loss_type="ip"):
    if normalize:
        zs = [F.normalize(z, p=2, dim=-1) for z in zs]
    if zs[0].dim() == 3:
        zs = [z.flatten(1, 2) for z in zs]
    m = len(zs)
    n = zs[0].shape[0]
    device = zs[0].device
    mask = torch.eye(n * m, device=device)
    label0 = torch.fmod(n + torch.arange(0, m * n, device=device), n * m)
    z = torch.cat(zs, 0)
    if loss_type == "euclid":
        sim = -torch.cdist(z, z)
    elif loss_type == "sq":
        sim = -(torch.cdist(z, z) ** 2)
    elif loss_type == "ip":
        sim = torch.matmul(z, z.transpose(0, 1))
    else:
        raise NotImplementedError
    logit_zz = sim / temperature
    logit_zz += mask * -1e8
    loss = nn.CrossEntropyLoss()(logit_zz, label0)
    return loss

def evaluate_model(net, testloader, criterion, device="cuda", eps=0.0, attack_method="fgsm"):
    """Evaluate model accuracy with optional adversarial attacks"""
    correct = 0
    total = 0
    net.eval()
    
    for data in testloader:
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        
        if eps > 0:
            if attack_method == "fgsm":
                inputs = fgsm_attack(net, inputs, labels, eps, criterion=criterion)
            elif attack_method == "random":
                inputs = random_attack(inputs, eps)
            elif attack_method == "pgd":
                inputs = pgd_linf_attack(
                    net,
                    inputs,
                    labels,
                    eps,
                    alpha=eps / 3,
                    num_iter=20,
                    criterion=criterion,
                )
            elif attack_method == "autoattack":
                inputs = autoattack(net, inputs, labels, eps)
            else:
                raise NotImplementedError(f"Attack method {attack_method} not implemented")
        
        with torch.no_grad():
            outputs = net(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    acc = 100 * correct / total
    if eps > 0:
        if attack_method == "fgsm":
            print(f"FGSM Adversarial Accuracy: {acc:.2f}%, eps:{255*eps:.1f}/255")
        elif attack_method == "random":
            print(f"Random Noise Accuracy: {acc:.2f}%, eps:{255*eps:.1f}/255")
        elif attack_method == "pgd":
            print(f"PGD Adversarial Accuracy: {acc:.2f}%, eps:{255*eps:.1f}/255")
        elif attack_method == "autoattack":
            print(f"Autoattack Adversarial Accuracy: {acc:.2f}%, eps:{eps:.1f}/255")
    else:
        print(f"Accuracy of the network on the test images: {acc:.2f}%")
    return acc

def adversarial_evaluation(writer, epoch, net, testloader, criterion, eps=8/255, eval_pgd=False, prefix="model/", device="cuda"):
    """Comprehensive adversarial evaluation"""
    # Clean accuracy
    writer.add_scalar(prefix+"test accuracy", evaluate_model(net, testloader, criterion, device), epoch)
    
    # Random noise
    writer.add_scalar(
        prefix+"Random noise test accuracy",
        evaluate_model(net, testloader, criterion, device, 64/255, attack_method="random"),
        epoch,
    )
    
    # FGSM attack
    writer.add_scalar(
        prefix+"FGSM test accuracy", 
        evaluate_model(net, testloader, criterion, device, eps, attack_method="fgsm"), 
        epoch
    )
    
    # PGD attack (less frequent)
    if eval_pgd:
        writer.add_scalar(
            prefix+"PGD test accuracy", 
            evaluate_model(net, testloader, criterion, device, eps, attack_method="pgd"), 
            epoch
        )


#########################
#########################
#########################
#######PRETRAINING#######
#########################
#########################
#########################
run_pretraining = False
load_model = True
load_phinet = False
pretraining_dataset="cifar10"
backbone="akorn"
load_model_name = "models/phinet_akorn_verification" #"/home/m/mohammed-habibi/.cache/phinet-cifar10-akorn_exp_0710044227" #
load_model_file = f"{load_model_name}.pth"
save_model_name = f"models/phinet_akorn_verification"
save_model_file = f"{save_model_name}.pth"
out_classes=10
out_dim=2048
ch=128
transform_train = augmentation_strong(imsize=32)
transform_test = transforms.Compose([transforms.ToTensor()])
transform_aug = simclr_augmentation(imsize=32)
transform_phinet = PhiNetTransform(image_size=32)
if pretraining_dataset == "tiny-imagenet":#Load tiny-imagenet dataset
    data_dir = '/work/YamadaU/mhabibi/tiny-imagenet-200'
    train_dir = f'{data_dir}/train'
    val_dir = f'{data_dir}/val'
    trainset = ImageFolder(train_dir, transform=transform_phinet)
elif pretraining_dataset == "cifar10":# Load CIFAR-10 dataset
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_phinet)

train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True)

if run_pretraining:
    # Model setup
    if backbone == "akorn":
        model = XPhiNetTF(backbone=AKOrN(
            n=2,
            ch=ch,
            out_classes=out_classes,
            L=3,
            T=3,
            J="conv",
            ksizes=[9,7,5],
            ro_ksize=3,
            ro_N=2,
            norm="bn",
            c_norm="gn",
            gamma=1.0,
            use_omega=True,
            init_omg=1.0,
            global_omg=True,
            learn_omg=True,
            ensemble=1,
        ), out_dim=out_dim).to(device)
    elif backbone == "convnext":
        model = XPhiNetTF(backbone=convnext_tiny_cifar(), out_dim=out_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4) #torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4) # 
    #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
    epochs = 400
    # Training Loop
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        n=0
        loop = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}/{epochs}", leave=False)

        for batch_idx, (x_ori, _) in loop:
            print(f"Epoch {epoch+1}/{epochs} - Batch {batch_idx+1}/{len(train_loader)}")
            x1, x2, x_ori = x_ori
            x1, x2, x_ori = x1.to(device), x2.to(device), x_ori.to(device)

            # Forward pass
            outputs = model(x1, x2, x_ori)
            loss = outputs['loss'].mean()
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * x_ori.shape[0]
            n += x_ori.shape[0]
        #scheduler.step()
        avg_loss = running_loss / n
        
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")

    # Save the model
    torch.save(model.slow_encoder.state_dict(), save_model_file)

#########################
#########################
#########################
#######FINE-TUNING#######
#########################
#########################
#########################

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
testset = torchvision.datasets.CIFAR10(root='./data',train=False,download=True,transform=transform_test)

train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False)

def vector_to_class(x):
  y = torch.argmax(F.softmax(x,dim=1),axis=1)
  return y

def prediction_accuracy(predict,labels):
  accuracy = (predict == labels).sum()/(labels.shape[0])
  return accuracy

if load_model:
    if load_phinet:
        if backbone == "akorn":
            model = XPhiNetTF(backbone=AKOrN(n=2,
                    ch=ch,out_classes=out_classes,L=3,T=3,J="conv",
                    ksizes=[9,7,5],ro_ksize=3,ro_N=2,
                    norm="bn",c_norm="gn",gamma=1.0,
                    use_omega=True,init_omg=1.0,global_omg=True,
                    learn_omg=True,ensemble=1,), out_dim=out_dim).to(device)
        elif backbone == "convnext":
            model = XPhiNetTF(backbone=convnext_tiny_cifar(), out_dim=out_dim).to(device)
        model.load_state_dict(torch.load(load_model_file)['state_dict'])
        model = model.slow_encoder.to(device)
    else:
        if backbone == "akorn":
            model = XPhiNetTF(backbone=AKOrN(n=2,
                        ch=ch,out_classes=out_classes,L=3,T=3,J="conv",
                        ksizes=[9,7,5],ro_ksize=3,ro_N=2,
                        norm="bn",c_norm="gn",gamma=1.0,
                        use_omega=True,init_omg=1.0,global_omg=True,
                        learn_omg=True,ensemble=1,), out_dim=out_dim).slow_encoder.to(device)
        elif backbone == "convnext":
            model = XPhiNetTF(backbone=convnext_tiny_cifar(), out_dim=out_dim).slow_encoder.to(device)
        model.load_state_dict(torch.load(load_model_file))
    
model = model[0] #Extract only AKoRN without the projection layer (if output dim of AKoRN is 10, it is perfect)

print(model)
# Training parameters
learning_rate = 1e-4
batch_size = 512

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.0) #torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
#scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

# Logging
jobdir = f"runs/phinet_akorn/"
writer = SummaryWriter(jobdir)
writer_ema = SummaryWriter(os.path.join(jobdir, "ema"))

# Training:
n_epochs =400

for epoch in range(0,n_epochs):

    model.train()

    train_loss=0.0
    n=0
    all_labels = []
    all_predicted = []

    with tqdm(train_loader, unit="batch") as tepoch:
        for imgs, labels in tepoch:
            tepoch.set_description(f"Epoch {epoch}")

            # Put data on device
            imgs = imgs.to(device)
            labels = labels.to(device)

            predict = model(imgs) # predicted logits
            loss = criterion(predict, labels) # compute loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Compute the loss
            train_loss += loss.item() * imgs.shape[0]
            n += imgs.shape[0]
            # Store labels and class predictions
            all_labels.extend(labels.tolist())
            all_predicted.extend(vector_to_class(predict).tolist())
        avg_loss = train_loss / n
    #scheduler.step()

    print('Epoch {}: Train Loss: {:.4f}'.format(epoch, avg_loss))
    print('Epoch {}: Train Accuracy: {:.4f}'.format(epoch, prediction_accuracy(np.array(all_predicted),np.array(all_labels))))

    # Adversarial evaluation
    if ((epoch + 1) % 20) == 0:
        # Evaluate original model
        print(f"Evaluating original model at epoch {epoch}")
        adversarial_evaluation(
            writer,
            epoch,
            model,
            test_loader,
            criterion,
            8/255,
            eval_pgd=True if ((epoch + 1) % 60) == 0 else False,
            device=device,
        )
    if (epoch + 1) % 100 == 0:
        # Save the model
        torch.save(model.state_dict(), f'{save_model_name}_{epoch}epochs.pth')

  # Save the model
torch.save(model.state_dict(), f'{save_model_name}_{n_epochs}epochs.pth')