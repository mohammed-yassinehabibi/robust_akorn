import math
import sys
from contextlib import ExitStack, redirect_stderr, redirect_stdout

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from autoattack import AutoAttack as _ExternalAutoAttack
except ModuleNotFoundError:  # pragma: no cover
    _ExternalAutoAttack = None


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


class _Tee:
    """Minimal tee stream to duplicate writes to multiple targets."""

    def __init__(self, *streams):
        self.streams = [stream for stream in streams if stream is not None]

    def write(self, data):
        for stream in self.streams:
            stream.write(data)

    def flush(self):
        for stream in self.streams:
            stream.flush()


def autoattack(net, image, target, epsilon, version='standard', bs=100, norm="Linf", log_file=None):
    if _ExternalAutoAttack is not None:
        return _external_autoattack(net, image, target, epsilon, version, bs, norm, log_file)
    return _pgd_fallback(net, image, target, epsilon, bs, log_file)


def _external_autoattack(net, image, target, epsilon, version, bs, norm, log_file):
    if log_file is None:
        adversary = _ExternalAutoAttack(net, norm=norm, eps=epsilon, version=version)
        return adversary.run_standard_evaluation(image, target, bs=bs)

    with ExitStack() as stack:
        log_handle = stack.enter_context(open(log_file, "a"))
        tee_out = _Tee(sys.stdout, log_handle)
        tee_err = _Tee(sys.stderr, log_handle)
        with redirect_stdout(tee_out), redirect_stderr(tee_err):
            adversary = _ExternalAutoAttack(net, norm=norm, eps=epsilon, version=version)
            return adversary.run_standard_evaluation(image, target, bs=bs)


def _pgd_fallback(net, image, target, epsilon, bs, log_file):
    """Simple PGD attack so evaluation works offline without AutoAttack."""

    def _log(line, handle):
        print(line)
        if handle is not None:
            handle.write(line + "\n")
            handle.flush()

    alpha = epsilon / 4
    num_iter = max(20, int(40 * (epsilon / (8 / 255))))
    criterion = nn.NLLLoss()

    with ExitStack() as stack:
        log_handle = stack.enter_context(open(log_file, "a")) if log_file else None
        _log("AutoAttack package missing; using PGD fallback.", log_handle)

        with torch.no_grad():
            clean_preds = net(image)
        clean_correct = clean_preds.argmax(dim=1).eq(target).sum().item()
        clean_acc = 100.0 * clean_correct / max(1, target.numel())
        _log(f"initial accuracy: {clean_acc:.2f}%", log_handle)

        total = target.numel()
        if total == 0:
            _log("No samples supplied for attack.", log_handle)
            return

        bs = max(1, min(bs or total, total))
        num_batches = math.ceil(total / bs)
        total_perturbed = 0

        for batch_idx in range(num_batches):
            start = batch_idx * bs
            end = min(start + bs, total)
            batch = image[start:end].clone().detach()
            labels = target[start:end].clone().detach()

            adv = pgd_linf_attack(
                net,
                batch,
                labels,
                epsilon,
                alpha=alpha,
                num_iter=num_iter,
                criterion=criterion,
            )

            with torch.no_grad():
                adv_logits = net(adv)
            adv_correct = adv_logits.argmax(dim=1).eq(labels).sum().item()
            perturbed = labels.size(0) - adv_correct
            total_perturbed += perturbed

            _log(
                f"pgd-linf - {batch_idx + 1}/{num_batches} - {perturbed} out of {labels.size(0)} successfully perturbed",
                log_handle,
            )

        robust_acc = 100.0 * (1 - total_perturbed / total)
        _log(f"robust accuracy: {robust_acc:.2f}%", log_handle)
