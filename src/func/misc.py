import torch
import torch.nn.functional as F

EPS = 1e-30


def expand(x: torch.Tensor, dim: int = 0, n: int = 1):
    """
    Expands tensor along specified dimension n times.
    """

    x = x.unsqueeze(dim)
    dims = [-1] * len(x.shape)
    dims[dim] = n
    x = x.expand(*dims)
    return x


def normalize(x: torch.Tensor, batch_dims: int = 1, low: float = 0.0, high: float = 1.0):
    """
    Normalizes x to [low, high]
    Inputs:
        x: [ batch, * ]
    Outputs:
        x: [ batch, * ]
    """

    shape = x.shape
    x = x.view(*shape[:batch_dims], -1)

    # Scale to [0, 1]
    lows, highs = x.min(dim=1, keepdim=True).values, x.max(dim=1, keepdim=True).values
    x = (x - lows) / (highs - lows)

    # Scale to requested range
    x = high * x - low

    x = x.view(*shape)
    return x


def onehot(x: torch.Tensor, vocab: int):
    """
    Promote indices to one-hot.
    Inputs:
        x: [ *, size ] (int)
    Outputs:
        x: [ *, size, masked ]
    """

    # if x.shape[-1] == vocab:
    #     x = nohot(x)
    x = F.one_hot(x, vocab).to(torch.float32)
    return x


def log_onehot(x: torch.Tensor, vocab: int):
    x = onehot(x, vocab)
    x = torch.log(x.float().clamp(min=EPS))
    return x


def nohot(x: torch.Tensor):
    """
    Reduce one-hot vector to indices.
    Inputs:
        x: [ *, size, masked ]
    Outputs:
        x: [ *, size ]
    """

    x = x.argmax(-1)
    return x


def sample_categorical(logits: torch.Tensor):
    """
    Samples a log-one-hot vector from logits using Gumbel noise.
    Inputs:
        logits: [ batch, seq, size, vocab ]
    Outputs:
        sample: [ batch, seq, size, vocab ]
    """

    vocab = logits.shape[-1]
    uniform = torch.rand_like(logits)
    gumbel = - torch.log(- (uniform + EPS).log() + EPS)
    sample = (gumbel + logits).argmax(dim=-1)
    sample = onehot(sample, vocab)
    return sample


def log_sample_categorical(logits: torch.Tensor):
    """
    Samples a log-one-hot vector from logits using Gumbel noise.
    Inputs:
        logits: [ batch, seq, *size ]
    Outputs:
        log_x:  [ batch, seq, *size ]
    """

    vocab = logits.shape[-1]
    uniform = torch.rand_like(logits)
    gumbel = - torch.log(- (uniform + EPS).log() + EPS)
    sample = (gumbel + logits).argmax(dim=-1)
    onehot = log_onehot(sample, vocab)

    return onehot


def log_sample_st(logits: torch.Tensor, tau: float = 1.0):
    """
    Samples a log-one-hot vector from logits using Gumbel noise.
    Inputs:
        logits: [ batch, seq, *size ]
    Outputs:
        log_x:  [ batch, seq, *size ]
    """

    vocab = logits.shape[-1]

    # Generate gumbel noise
    gumbel = - torch.log(- (torch.rand_like(logits) + EPS).log() + EPS)

    # Softmax logits using tau (has gradient)
    soft = ((logits + gumbel) / tau).softmax(-1)

    # Hard onehot logits + noise (no gradient)
    hard = onehot((logits + gumbel).argmax(-1), vocab)

    # Straight-through trick
    # Result is the same as "hard", but with gradients of "soft"
    sample = hard - soft.detach() + soft

    # Log sample
    sample = (sample.float().clamp(min=EPS)).log()

    return sample


def log_sample_gumbel(logits: torch.Tensor):
    sample = F.gumbel_softmax(logits, tau=1, hard=True)
    sample = torch.log(sample.float().clamp(min=EPS))
    return sample


def deterministic_gumbel(logits: torch.Tensor, tau: float = 0.1):

    vocab = logits.shape[-1]

    # Softmax logits using tau temperature (has gradient)
    soft = (logits / tau).softmax(dim=-1)

    # Hard onehot logits using argmax (no gradient)
    hard = onehot(logits.argmax(-1), vocab)

    # Straight-through trick
    # Result is the same as "hard", but with gradients of "soft"
    sample = hard - soft.detach() + soft

    # # Log
    # sample = torch.log(sample.float().clamp(min=EPS))

    return sample


def truncate(x: torch.Tensor, truncation: float):

    # Sort each distribution by probability
    sorted, indices = torch.sort(x, dim=-1, descending=True)
    cumsum = sorted.exp().cumsum(dim=-1) < truncation

    # Keep highest probability and drop lowest
    highest = torch.full_like(cumsum[..., [0]], True)
    x_new = torch.cat((highest, cumsum), dim=-1)[..., :-1]

    # Re-arrange cumulative probabilities to same configuration as x
    x_new = x_new.gather(-1, indices.argsort(-1))

    # Linear interpolation between original and modified distributions
    probs = x_new.float() * x + (1 - x_new.float()) * (-70)
    return probs


