from typing import Any, Literal, Union

from torch import nn
import torch

from .module import Module
from ..options import Options


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

    # Required
    in_chan: int
    out_chan: int

    # Optional
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
        block = conv(self.in_chan, self.out_chan,
                     kernel_size=self.kernel_size, stride=self.stride,
                     padding=padding)
        layers = [block]
        if self.norm:
            layers += [norm(self.out_chan)]
        if self.act is not None:
            layers += [self.act()]
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor):
        x = x.to(self.device)

        # [ b, h, w ] -> [ b, 1, h, w ]
        if len(x.shape) == 3 and self.dimension == '2d':
            x = x[None]

        x = self.block(x)

        return x


class ConvBlocks (Module):

    # Required
    chans: list[int]

    # Optional
    kernel_size: int = 3
    stride: int = 1
    deconv: bool = False
    final_act: bool = True
    act: Any = nn.ELU
    norm: bool = True
    padding: Union[Literal['same'], int] = 'same'
    dimension: Literal['1d', '2d', '3d'] = '2d'

    def build(self):

        # Construct blocks
        self._layers = []
        for i in range(len(self.chans) - 1):
            opts = self.opts(in_chan=self.chans[i], out_chan=self.chans[i + 1])
            block = ConvBlock(opts)
            block._hide_grads = True
            self.add_module(f'_block{i}', block)
            self._layers += [block]

    def forward(self, x):
        x = x.to(self.device)

        # [ b, h, w ] -> [ b, 1, h, w ]
        if len(x.shape) == 3 and self.dimension == '2d':
            x = x[None]

        for layer in self._layers:
            x = layer(x)

        return x


if __name__ == "__main__":

    from ml import console

    model = ConvBlocks({'chans': [3, 32, 1]})
    model._build()

    inp = torch.rand(5, 3, 10, 10)
    out = model(inp)

    console.log(model)
    console.log(inp.shape)
    console.log(out.shape)
