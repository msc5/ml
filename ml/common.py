from typing import Any, Literal, Union

import torch
import torch.nn as nn

from .options import Options
from .module import Module


def conv_layer(dimension: Literal['1d', '2d', '3d'], deconv: bool):
    if dimension == '1d':
        conv = nn.ConvTranspose1d if deconv else nn.Conv1d
        norm = nn.BatchNorm1d
    elif dimension == '2d':
        conv = nn.ConvTranspose2d if deconv else nn.Conv2d
        norm = nn.BatchNorm2d
    elif dimension == '3d':
        conv = nn.ConvTranspose3d if deconv else nn.Conv3d
        norm = nn.BatchNorm3d
    else:
        raise Exception('dimension must be \'2d\', \'1d\', or \'3d\'.')
    return conv, norm


class ConvBlock (Module):

    in_chan: int
    out_chan: int
    kernel_size: int = 3
    stride: int = 1
    deconv: bool = False
    act: Any = nn.ELU
    norm: bool = True
    padding: Union[Literal['same'], int] = 'same'
    dimension: Literal['1d', '2d', '3d'] = '2d'

    def build(self):
        conv, norm = conv_layer(self.dimension, self.deconv)
        padding = self.kernel_size // 2 if self.padding == 'same' else self.padding
        layers = [conv(self.in_chan, self.out_chan,
                       kernel_size=self.kernel_size, stride=self.stride,
                       padding=padding)]
        if self.norm:
            layers += [norm(self.out_chan)]
        if self.act is not None:
            layers += [self.act()]
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor):
        if len(x.shape) == 3 and self.dimension == '2d':
            x = x[None]
        x = self.block(x)
        return x


class ConvBlocks (Module):

    chans: list[int]
    kernel_size: int = 3
    stride: int = 1
    deconv: bool = False
    final_act: bool = True
    act: Any = nn.ELU
    norm: bool = True
    padding: Union[Literal['same'], int] = 'same'
    dimension: Literal['1d', '2d', '3d'] = '2d'

    def build(self):
        conv, norm = conv_layer(self.dimension, self.deconv)
        padding = self.kernel_size // 2 if self.padding == 'same' else self.padding
        layers = []
        for i in range(len(self.chans) - 1):
            layers += [conv(self.chans[i], self.chans[i + 1],
                            kernel_size=self.kernel_size, stride=self.stride,
                            padding=padding)]
            if self.norm:
                layers += [norm(self.chans[i + 1])]
            if self.act is not None:
                layers += [self.act()]
        # if self.act is not None and self.final_act:
        #     layers += [self.act()]
        if not self.final_act:
            layers = layers[:-1]
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        if len(x.shape) == 3 and self.dimension == '2d':
            x = x[None]
        x = self.block(x)
        return x


class LinearBlock (Module):

    in_size: int
    out_size: int
    act: Any = nn.ELU

    def build(self):
        layers = [nn.Linear(self.in_size, self.out_size)]
        if self.act is not None:
            layers += [self.act()]
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        x = self.block(x)
        return x


class LinearBlocks (Module):

    sizes: list[int]
    final_act: bool = True
    act: Any = nn.ELU

    def build(self):
        layers = []
        for i in range(len(self.sizes) - 1):
            layers += [nn.Linear(self.sizes[i], self.sizes[i + 1])]
            if self.act is not None:
                layers += [self.act()]
        if not self.final_act:
            layers = layers[:-1]
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        x = self.block(x)
        return x


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


# class Transformer (Module):
#
#     size: int
#     heads: int
#     layers: int
#
#     final_act: bool = True
#     act: Any = nn.ELU
#
#     def build(self):
#         layers = []
#         for _ in range(self.layers - 1):
#             layer = nn.TransformerEncoderLayer
#             layers += [layer(self.size, self.heads, batch_first=True)]
#             if self.act is not None:
#                 layers += [self.act()]
#         if self.act is not None and self.final_act:
#             layers += [self.act()]
#         self.block = nn.Sequential(*layers)
#
#     def forward(self, x):
#         x = self.block(x)
#         return x

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


if __name__ == "__main__":

    o = Options()
    o.size = 200
    o.heads = 10
    o.layers = 32

    m = Transformer(o)
    m._build()

    breakpoint()
