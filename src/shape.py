from __future__ import annotations
from dataclasses import asdict, astuple, dataclass
from typing import Optional

import torch

from .cli import console


@dataclass
class Shape:

    # batch fields
    batch: Optional[int] = None
    sequence: Optional[int] = None

    # value fields
    vector: Optional[int] = None
    chan: Optional[int] = None
    height: Optional[int] = None
    width: Optional[int] = None

    def __repr__(self):
        values = self.items()
        batch = ', '.join([str(v) for (k, v) in values.items()
                           if k in ['sequence', 'batch']])
        value = ', '.join([str(v) for (k, v) in values.items()
                           if k not in ['sequence', 'batch']])
        return f'({batch} | {value})'

    def __iter__(self):
        for v in astuple(self):
            if v is not None:
                yield v

    def items(self):
        return {k: v for k, v in asdict(self).items() if v is not None}

    def set(self, key, value) -> Shape:
        values = self.items()
        values[key] = value
        return Shape(**values)

    def image(self) -> Shape:
        return Shape(chan=self.chan, height=self.height, width=self.width)

    def event(self) -> Shape:
        values = self.items()
        if all([v in values for v in ['chan', 'height', 'width']]):
            return self.image()
        elif 'vector' in values:
            return Shape(vector=self.vector)
        else:
            raise Exception()

    def unsequence(self) -> Shape:
        """ 
        Collapse sequence and batch dimensions of Shape into one.
        """
        values = self.items()
        if 'sequence' in values:
            values['batch'] = values['batch'] * values['sequence']
            values['sequence'] = None
        return Shape(**values)

    def flat(self) -> Shape:
        values = self.items()
        others = {k: v for (k, v) in values.items()
                  if k not in ['sequence', 'batch']}
        product = 1
        for (k, v) in others.items():
            product *= v
            values[k] = None
        values['vector'] = product
        return Shape(**values)

    def conv(self,
             kernel_size: int,
             stride: int = 1,
             padding: int = 0,
             n: int = 1,
             out_chan: Optional[int] = None) -> Shape:
        """ 
        Returns the shape of an image after 2D convolution.
        Parameters:
            kernel_size (int): size of convolution kernel
            stride      (int): stride of convolution operation
            padding     (int): padding of convolution operation
        Output:
            out_shape   (int): shape of input after 2D convolution
        """
        assert self.height is not None
        assert self.width is not None
        width, height = self.width, self.height
        for _ in range(n):
            width = width - kernel_size + 2 * padding // stride + 1
            height = height - kernel_size + 2 * padding // stride + 1
        values = self.items()
        values['height'], values['width'] = height, width
        if out_chan is not None:
            values['chan'] = out_chan
        return Shape(**values)

    def deconv(self,
               kernel_size: int,
               stride: int = 1,
               padding: int = 0,
               n: int = 1,
               out_chan: Optional[int] = None) -> Shape:
        """ 
        Returns the shape of an image after 2D deconvolution.
        Parameters:
            kernel_size (int): size of deconvolution kernel
            stride      (int): stride of deconvolution operation
            padding     (int): padding of deconvolution operation
        Output:
            out_shape   (int): shape of input after 2D deconvolution
        """
        assert self.height is not None
        assert self.width is not None
        width, height = self.width, self.height
        for _ in range(n):
            height = (stride * (height - 1) + kernel_size - 2 * padding)
            width = (stride * (width - 1) + kernel_size - 2 * padding)
        values = self.items()
        values['height'], values['width'] = height, width
        if out_chan is not None:
            values['chan'] = out_chan
        return Shape(**values)

    def reshape(self, x: torch.Tensor) -> torch.Tensor:
        return x.reshape(*self)


if __name__ == '__main__':

    s = Shape(batch=5, sequence=12, chan=3, height=64, width=64)
    console.log(s)

    for n in s:
        console.log(s)

    s = s.unsequence()
    console.log(s)

    s = s.conv(3)
    console.log(s)

    s = s.deconv(3)
    console.log(s)

    s = s.flatten()
    console.log(s)

    console.log([*s])

    s = Shape(batch=5, sequence=12, chan=3, height=64, width=64)
    console.log(s)

    x = torch.rand(*s)
    console.log(x.shape)

    x = x.reshape(*s.flatten())
    console.log(x.shape)
