from typing import Optional
import torch
import torch.distributions as td

from dataclasses import dataclass

from .cli import console


@dataclass
class Distribution:

    logits: torch.Tensor

    dist: td.Distribution

    def __init__(self, logits: torch.Tensor, batch_dims: int = 1):
        self.batch_dims = batch_dims
        self.logits = logits
        norm = td.Normal(self.logits, 1)
        dist = td.Independent(norm, batch_dims)
        self.dist = dist

    def __rich_repr__(self):
        yield 'logits', self.logits.shape
        yield 'dist', self.__class__.__name__

    def __repr__(self):
        return f'{self.__class__.__name__} -> {self.logits.shape}'

    @property
    def shape(self):
        return self.logits.shape

    def sample(self, *args, **kwargs):
        z = self.dist.sample()
        return z

    def log_prob(self, sample: torch.Tensor):
        sample = sample.reshape(*self.logits.shape)
        log_prob = self.dist.log_prob(sample)
        return log_prob

    def detach(self):
        return self.__class__(self.logits.detach(),
                              batch_dims=self.batch_dims)

    @classmethod
    def stack(cls, distributions: list, dim: int = 0):
        logits = torch.stack([d.logits for d in distributions], dim=dim)
        return cls(logits)


class Continuous (Distribution):
    pass


class Bernoulli (Distribution):

    probs: torch.Tensor

    def __init__(self, logits: torch.Tensor, batch_dims: int = 1):
        self.logits = logits
        norm = td.Bernoulli(logits=self.logits)
        dist = td.Independent(norm, batch_dims)
        self.dist = dist
        if isinstance(norm.probs, torch.Tensor):
            self.probs = norm.probs
        else:
            self.probs = torch.tensor(norm.probs)

    def sample(self, straight_through: bool = False):
        z = self.dist.sample()
        if straight_through:
            z = z + self.probs - self.probs.detach()
        return z


class Categorical (Distribution):

    probs: torch.Tensor

    def __init__(self, logits: torch.Tensor, independent: Optional[int] = None):
        self.logits = logits
        self.dist = td.OneHotCategorical(logits=self.logits)
        self.independent = independent

        if independent is not None:
            self.dist = td.Independent(self.dist, independent)
            d = self.dist.base_dist
            if isinstance(d.probs, torch.Tensor):
                self.probs = d.probs
            else:
                self.probs = torch.tensor(d.probs)
        else:
            if isinstance(self.dist.probs, torch.Tensor):
                self.probs = self.dist.probs
            else:
                self.probs = torch.tensor(self.dist.probs)

    def sample(self, straight_through: bool = False):
        z = self.dist.sample()
        if straight_through:
            z = z + self.probs - self.probs.detach()
        return z

    def detach(self):
        return self.__class__(self.logits.detach(),
                              independent=self.independent)

    def to(self, device):
        logits = self.logits.to(device)
        return Categorical(logits, self.independent)

    @classmethod
    def stack(cls, distributions: list, dim: int = 0):
        logits = torch.stack([d.logits for d in distributions], dim=dim)
        independent = distributions[0].independent - 1
        return Categorical(logits, independent=independent)


if __name__ == "__main__":

    x = torch.rand(5, 3)

    # dist = Bernoulli(x)
    # console.log(x.equal(dist.logits))
    # console.log(dist.logits)
    # console.log(dist.probs)
    #
    # y = dist.sample()
    # console.log(y)
    #
    # lp = dist.log_prob(y)
    # console.log(lp)
    # console.log(-lp.mean())
    #
    # dist = Categorical(x)
    # console.log(x.equal(dist.logits))
    # console.log(dist.logits)
    # console.log(dist.probs)
    #
    # y = dist.sample()
    # console.log(y)
    #
    # lp = dist.log_prob(y)
    # console.log(lp)
    # console.log(-lp.mean())

    D = []
    for s in x:
        x = torch.rand(3)
        d = Categorical(x)
        console.log(d.probs)
        D.append(d)

    console.log(D)

    D = Categorical.stack(D)

    console.log(D.probs)
    console.log(D.dist.entropy())
