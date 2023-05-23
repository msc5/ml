from typing import Union
import torch


class EMA (torch.Tensor):

    def update(self, value: Union[float, torch.Tensor], p: float = 0.005):
        self.data = (1 - p) * self.data + p * value
