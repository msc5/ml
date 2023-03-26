from typing import Any

from torch import nn
from .module import Module


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
