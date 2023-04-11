from __future__ import annotations

from typing import Optional
import torch

LOG_MIN = 1e-30


def normal_prob(mean: torch.Tensor, std: torch.Tensor, sample: torch.Tensor):
    """
    Computes likelihood of a sample given the mean and std of a normal
    distribution.
    """

    assert mean.shape == std.shape == sample.shape

    prob = ((1 / (torch.tensor(2 * torch.pi).sqrt() * std))
            * torch.exp(- (sample - mean)**2 / (2 * std)**2))

    return prob


def normal_log_prob(mean: torch.Tensor, std: torch.Tensor, sample: torch.Tensor):
    """
    Computes log-likelihood of a sample given the mean and std of a normal distribution.
    """

    prob = normal_prob(mean, std, sample)
    log_prob = torch.log(prob.clamp(LOG_MIN, None))

    return log_prob


def normal_entropy(mean: torch.Tensor, std: torch.Tensor, sample: torch.Tensor):
    """
    Computes entropy of a sample given the mean and std of a normal distribution.
    """

    entropy = - normal_log_prob(mean, std, sample)
    entropy = entropy.flatten(1).mean()

    return entropy


class MultivariateGaussian:

    def __init__(self, logits: torch.Tensor):
        self.logits = logits
        self.mean, self.logvar = self.logits.chunk(2, dim=-1)
        self.std = torch.exp(self.logvar / 2)
        self.var = torch.exp(self.logvar)

    def sample(self):
        noise = torch.randn_like(self.mean)
        sample = self.mean + self.std * noise
        return sample

    def prob(self, sample: torch.Tensor):
        """
        Computes probability p(x) of sample x with respect to this
        distribution.
        """
        return normal_prob(self.mean, self.std, sample)

    def log_prob(self, sample: torch.Tensor):
        """
        Computes log probability p(x) of sample x with respect to this
        distribution.
        """
        return normal_log_prob(self.mean, self.std, sample)

    def kl(self, other: Optional[MultivariateGaussian] = None, batch_dims: int = 1):
        """
        Computes KL-divergence of this distribution and another distribution.
        If unspecified, assume other distribution is standard normal.
        """
        except_batch = list(range(len(self.mean.shape)))[batch_dims:]
        if other is None:
            return 0.5 * torch.sum(self.mean**2 + self.var - 1.0 - self.logvar, dim=except_batch)
        else:
            return 0.5 * torch.sum((self.mean - other.mean)**2 / other.var
                                   + self.var / other.var - 1.0
                                   + other.logvar - self.logvar,
                                   dim=except_batch)

    def nll(self, sample: torch.Tensor, batch_dims: int = 1):
        except_batch = list(range(len(self.mean.shape)))[batch_dims:]
        log_two_pi = torch.log(2.0 * torch.tensor(torch.pi))
        return 0.5 * torch.sum(log_two_pi + self.logvar
                               + (sample - self.mean)**2 / self.var, dim=except_batch)
