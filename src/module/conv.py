from typing import Any, Literal, Optional, Union

from torch import nn
import torch

from ..cli import console
from .module import Module


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

        # Select conv and norm layer
        if self.dimension == '1d':
            conv = nn.ConvTranspose1d if self.deconv else nn.Conv1d
            norm = nn.BatchNorm1d
        elif self.dimension == '2d':
            conv = nn.ConvTranspose2d if self.deconv else nn.Conv2d
            norm = nn.BatchNorm2d
        elif self.dimension == '3d':
            conv = nn.ConvTranspose3d if self.deconv else nn.Conv3d
            norm = nn.BatchNorm3d
        else:
            raise Exception('dimension must be \'2d\', \'1d\', or \'3d\'.')

        # Compute padding
        if self.padding == 'same':
            padding = self.kernel_size // 2
        elif isinstance(self.padding, int):
            padding = self.padding
        else:
            raise Exception('padding must be \'same\' or int')

        # Construct layers
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
            block._hide_module = True
            self.add_module(f'_block{i}', block)
            self._layers += [block]

    def forward(self, x: torch.Tensor):
        x = x.to(self.device)

        for layer in self._layers:
            x = layer(x)

        return x


class DownConvBlock (ConvBlocks):

    chans: list[int]
    layers: int = 1

    c_kernel_size: int = 4
    c_stride: int = 2
    c_padding: int = 1

    def build(self):

        # Construct blocks
        self._layers = []
        for i in range(len(self.chans) - 1):

            opts = self.opts(in_chan=self.chans[i], out_chan=self.chans[i + 1])

            # Downsample last layer
            if i == len(self.chans) - 2:
                opts = opts(kernel_size=self.c_kernel_size, stride=self.c_stride, padding=self.c_padding)

            block = ConvBlock(opts)
            self.add_module(f'_block{i}', block, hide=True)
            self._layers += [block]


class UpConvBlock (ConvBlocks):

    chans: list[int]
    layers: int = 1
    residual: bool = False

    c_kernel_size: int = 4
    c_stride: int = 2
    c_padding: int = 1

    def build(self):

        # Reverse channels
        chans = list(reversed(self.chans))

        if self.residual:
            chans[0] = 2 * chans[0]

        # Construct blocks
        self._layers = []
        for i in range(len(chans) - 1):

            opts = self.opts(in_chan=chans[i], out_chan=chans[i + 1])

            # Upsample last layer
            if i == len(chans) - 2:
                opts = opts(kernel_size=self.c_kernel_size, stride=self.c_stride,
                            padding=self.c_padding, deconv=True)

            block = ConvBlock(opts)
            self.add_module(f'_block{i}', block, hide=True)
            self._layers += [block]

    def forward(self, x: torch.Tensor, residual: Optional[torch.Tensor] = None):
        x = x.to(self.device)

        # Concatenate residual
        if residual is not None:
            residual = residual.to(self.device)
            x = torch.cat([x, residual], dim=-3)

        for layer in self._layers:
            x = layer(x)

        return x


class Unet (Module):

    in_chan: int
    out_chan: int

    chans: list[tuple[int, int]]
    residual: bool = True

    c_kernel_size: int = 4
    c_stride: int = 2
    c_padding: int = 1

    def build(self):

        self._downs, self._ups = [], []

        # In size -> first conv chan
        chans = [self.in_chan, self.chans[0][0]]
        block = ConvBlock(self.opts(chans=chans))
        self.add_module(f'_down_chan{0}', block)
        self._downs += [block]

        # Construct downconv
        for i, (chan, layers) in enumerate(self.chans):

            # Add main conv blocks
            chans = [chan] * layers
            block = DownConvBlock(self.opts(chans=chans))
            self.add_module(f'_down{i}', block)
            self._downs += [block]

        # Construct upconv
        for i, (chan, layers) in enumerate(reversed(self.chans)):

            # Add main conv blocks
            chans = [chan] * layers
            block = UpConvBlock(self.opts(chans=chans))
            self.add_module(f'_up{i}', block)
            self._ups += [block]

            # If not first step
            if i > 0:
                chans = [chan, self.chans[-(i + 1)][0]]
                block = ConvBlock(self.opts(chans=chans))
                self.add_module(f'_up_chan{i}', block)
                self._ups += [block]

    def dry_run(self, size: Union[int, float] = 64, chan: int = 3):

        shape = [chan, size, size]
        shapes = [shape]

        # Downconv
        for group in self.chans:
            size = (size - self.c_kernel_size + 2 * self.c_padding) / self.c_stride + 1
            shape = [group[-1], size, size]
            shapes.append(shape)

        hid_shape = shape

        # Upconv
        for group in reversed(self.chans):
            size = (size - 1) * self.c_stride - 2 * self.c_padding + (self.c_kernel_size - 1) + 1
            shape = [group[0], size, size]
            shapes.append(shape)

        return hid_shape, shapes

    def downsample(self, x: torch.Tensor):

        h = []

        for down in self._downs:
            x = down(x)
            h.append(x)

        return x, h

    def upsample(self, x: torch.Tensor, h: Optional[list[torch.Tensor]] = None):

        for up in self._ups:
            if self.residual and h is not None:
                x = up(x, h.pop(-1))
            else:
                x = up(x)

        return x

    def forward(self, x: torch.Tensor):

        z, h = self.downsample(x)
        y = self.upsample(z, h)

        return y
