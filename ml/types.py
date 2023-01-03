import torch
from torchtyping import TensorType, patch_typeguard
from typing import Any, Union, Literal

Mask = TensorType['batch', 'seq', torch.bool]
MaskRatio = Union[float, torch.Tensor]
MaskStrategy = Literal['uniform', 'middle', 'forward', 'none']

Unmasked = TensorType['batch', 'seq', Any]
Masked = TensorType['batch', 'masked_seq', Any]

Extras = dict[str, Any]

DataDict = dict[Any, Union['DataDict', torch.Tensor]]

patch_typeguard()
