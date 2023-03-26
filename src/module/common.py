from typing import Any, Type, Union

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

from .module import Module


class Chain (Module):

    links: list[Type[Module]]

    def build(self):
        self._links = []
        for i, link in enumerate(self.links):
            module = link(self.opts)._build()
            self.add_module(module.__class__.__name__ + str(i), module)
            self._links += [module]

    def forward(self, *args, **kwargs):

        for link in self._links:
            args = link(*args, **kwargs)

        return args


class TransformerBlock (nn.Module):

    def __init__(self, size: int, heads: int):
        super().__init__()
        self.attn = nn.MultiheadAttention(size, heads, batch_first=True)
        self.lin = nn.Linear(size, size)
        self.norm_a = nn.LayerNorm(size)
        self.norm_b = nn.LayerNorm(size)

        self.test_enc = nn.TransformerEncoderLayer(d_model=size, nhead=heads, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        # attn, _ = self.attn(x, x, x)     # Compute Attention
        # x = self.norm_a(x + attn)        # Add and norm
        #
        # lin = self.lin(x)                # Linear layer
        # x = self.norm_b(x + lin)         # Add and norm

        x = self.test_enc(x)

        return x


class Params (Module):

    shape: Union[int, tuple[int]]

    def build(self):
        self.weight = Parameter(torch.rand(self.shape))
        # self.__getitem__ = self.weight.__getitem__

    def __getitem__(self, *args, **kwargs):
        return self.weight.__getitem__(*args, **kwargs)

    def forward(self):
        return self.weight


class Transformer (Module):

    size: int
    heads: int
    layers: int

    act: Any = None

    def build(self):
        self.block = nn.Transformer(self.size, self.heads, self.layers, self.layers, batch_first=True)
        self.activation = self.act() if self.act is not None else nn.Identity()

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        """ 
        Inputs: 
            q: [ batch, seq, size ]
            k: [ batch, seq, size ]
            v: [ batch, seq, size ]
        """
        x = self.block(q, k, v)
        x = self.activation(x)
        return x


class Embedding (Module):

    count: int
    size: int

    # _hide_grads: bool = True

    def build(self):
        self.embeddings = nn.Embedding(self.count, self.size)
        self.weight = self.embeddings.weight

    def forward(self, *args, **kwargs):
        return self.embeddings(*args, **kwargs)


class PatchEmbed (Module):

    in_chan: int
    hid_size: int
    patch_size: int

    def build(self):
        self.conv = nn.Conv2d(self.in_chan, self.hid_size,
                              kernel_size=self.patch_size, stride=self.patch_size)

    def forward(self, x):
        x = self.conv(x)
        return x


class Embed (Module):

    in_sizes: list[int]
    out_size: int
    act: Any = nn.ELU

    def build(self):
        layers = [nn.Linear(sum(self.in_sizes), self.out_size)]
        if self.act is not None:
            layers += [self.act()]
        self.block = nn.Sequential(*layers)

    def forward(self, *x, dim: int = -1):
        x = torch.cat(x, dim=dim)
        x = self.block(x)
        return x
