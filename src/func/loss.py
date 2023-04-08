import torch


def mse(output: torch.Tensor, target: torch.Tensor):
    return ((output - target)**2).mean()


def abse(output: torch.Tensor, target: torch.Tensor):
    return (output - target).abs().mean()
