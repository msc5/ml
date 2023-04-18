import torch


def mse(output: torch.Tensor, target: torch.Tensor):
    target = target.to(output.device)
    return ((output - target)**2).mean()


def abse(output: torch.Tensor, target: torch.Tensor):
    target = target.to(output.device)
    return (output - target).abs().mean()
