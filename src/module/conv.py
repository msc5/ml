from typing import Any, Literal, Union

from torch import nn
import torch

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
