import torch


def normal_prob(mean: torch.Tensor, std: torch.Tensor, sample: torch.Tensor):
    """
    Computes likelihood of a sample given the mean and std of a normal distribution.
    """

    prob = ((1 / (torch.tensor(2 * torch.pi).sqrt() * std))
            * torch.exp(- (sample - mean)**2 / (2 * std)**2))

    return prob


def normal_log_prob(mean: torch.Tensor, std: torch.Tensor, sample: torch.Tensor):
    """
    Computes log-likelihood of a sample given the mean and std of a normal distribution.
    """

    prob = normal_prob(mean, std, sample)
    log_prob = torch.log(prob.clamp(1e-30, None))

    return log_prob


def normal_entropy(mean: torch.Tensor, std: torch.Tensor, sample: torch.Tensor):
    """
    Computes entropy of a sample given the mean and std of a normal distribution.
    """

    entropy = - normal_log_prob(mean, std, sample)
    entropy = entropy.flatten(1).mean()

    return entropy
