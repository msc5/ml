import torch


def pos_embed(x: torch.Tensor, embed_dim: int = -1, seq_dim: int = -2, max_len: int = 1000):
    """
    Generates a sinusoidal positional embedding in 'embed_dim' along 'seq_dim'.
    Inputs:
        x: [ *, seq, size ]
    """

    seq_len = x.shape[seq_dim]
    size = x.shape[embed_dim]

    # x : [ batch, seq_len (t), size (i) ]
    t = torch.arange(0, seq_len, device=x.device)
    i = torch.arange(0, size, device=x.device)
    k = i.div(2, rounding_mode='floor')

    evens = (i % 2) == 0
    odds = ~evens
    w = 1 / (max_len**(2 * k / size))

    p = torch.zeros(seq_len, size, device=x.device)
    p[:, evens] = (w[None, evens] * t[:, None]).sin()
    p[:, odds] = (w[None, odds] * t[:, None]).cos()

    shape = [1] * len(x.shape)
    shape[seq_dim], shape[embed_dim] = seq_len, size
    p = p.reshape(*shape)
    p = p.expand(*x.shape)

    return p
