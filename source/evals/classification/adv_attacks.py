import torch
import torch.nn as nn
import torch.nn.functional as F
from autoattack import AutoAttack


def random_attack(image, epsilon):   
    sign_data_grad = torch.randn_like(image).sign()
    perturbed_image = image + epsilon * sign_data_grad
    # pixel values are in [0, 1]
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    return perturbed_image

def fgsm_attack(net, image, target, epsilon, criterion=nn.CrossEntropyLoss()):
    image.requires_grad = True

    output = net(image)
    loss = criterion(output, target)
    net.zero_grad()
    loss.backward()
    data_grad = image.grad.data

    sign_data_grad = data_grad.sign()
    perturbed_image = image + epsilon * sign_data_grad
    # pixel values are in [0, 1]
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    return perturbed_image

def pgd_l2_attack(net, image, target, epsilon, alpha, num_iter, criterion=nn.CrossEntropyLoss()):
    original_image = image.clone()
    image.requires_grad = True

    for _ in range(num_iter):
        output = net(image)
        loss = criterion(output, target)
        net.zero_grad()
        loss.backward()
        data_grad = image.grad.data

        # Normalize the gradients to L2 norm
        norm_data_grad = data_grad / data_grad.view(data_grad.size(0), -1).norm(p=2, dim=1).view(-1, 1, 1, 1)
        perturbed_image = image + alpha * norm_data_grad

        # Project back to L2 norm ball of radius epsilon
        delta = perturbed_image - original_image
        delta = delta.view(delta.size(0), -1)
        mask = delta.norm(p=2, dim=1) > epsilon
        scaling_factor = (epsilon / delta.norm(p=2, dim=1))
        scaling_factor[mask] = 1.0
        delta = delta * scaling_factor.view(-1, 1)

        image = original_image + delta.view(*original_image.shape)
        image = torch.clamp(image, 0, 1)
        image = image.detach()
        image.requires_grad = True

    return image

def pgd_linf_attack(net, image, target, epsilon, alpha, num_iter, criterion=nn.CrossEntropyLoss(), targeted=False, eot=1):
    original_image = image.clone().detach()
    image.requires_grad = True

    for _ in range(num_iter):
        
        data_grad = torch.zeros_like(image)
        for _ in range(eot):
            output = net(image)
            loss = criterion(output, target)
            net.zero_grad()
            loss.backward()
            data_grad += image.grad.data / eot  

        # Apply FGSM step
        if targeted:
            perturbed_image = image - alpha * data_grad.sign()
        else:
            perturbed_image = image + alpha * data_grad.sign()

        # Project back to the L∞ norm ball
        perturbed_image = torch.clamp(perturbed_image, original_image - epsilon, original_image + epsilon)
        image = torch.clamp(perturbed_image, 0, 1).detach()
        image = image.detach()
        image.requires_grad = True

    return image


def autoattack(net, image, target, epsilon, version='standard', bs=100, norm="Linf"):
    adversary = AutoAttack(net, norm=norm, eps=epsilon, version=version)
    image_adv = adversary.run_standard_evaluation(image, target, bs=bs)
    return image_adv
