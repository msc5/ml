from typing import Optional

import torch
from typeguard import typechecked

from .module import Module

from . import types as tt


class Masker (Module):

    mask_ratio: float = 0.8
    mask_strategy: tt.MaskStrategy = 'uniform'

    @typechecked
    def mask(self,
             x: tt.Unmasked,
             mask: Optional[tt.Mask] = None,
             mask_ratio: Optional[tt.MaskRatio] = None,
             strategy: Optional[tt.MaskStrategy] = None,
             keep_zeros: bool = False
             ) -> tuple[tt.Masked, tt.Mask]:
        """
        Masks input tensor x (keeps true values). Returns the masked tensor as well as boolean mask.
        Inputs:
            x:      [ batch, seq_len, size ]
            mask:   [ batch, seq_len ] 
            strategy:
                'uniform': Masks all tokens of x with constant uniform likelihood
                'middle': Masks entire middle of x, excluding initial and final states in sequence
                'forward': Masks all tokens of x except for first token
                'none': Masks no tokens
        Outputs:
            masked: [ batch, masked_seq_len, size ]
            mask:   [ batch, seq_len ]
        """

        mask_ratio = mask_ratio or self.mask_ratio
        strategy = strategy or self.mask_strategy

        batch, seq_len, size = x.shape
        n = int(seq_len * (1 - mask_ratio))

        if mask is None:
            if strategy == 'none':
                _mask = torch.ones((batch, seq_len), device=x.device, dtype=torch.bool)
            elif strategy == 'uniform':
                _mask = torch.zeros((batch, seq_len), device=x.device, dtype=torch.bool)
                choice = torch.ones((batch, seq_len), device=x.device).multinomial(n)
                _mask.scatter_(1, choice, True)
            elif strategy == 'middle':
                _mask = torch.zeros((batch, seq_len), device=x.device, dtype=torch.bool)
                _mask[:, [0, -1]] = True
            elif strategy == 'forward':
                _mask = torch.zeros((batch, seq_len), device=x.device, dtype=torch.bool)
                _mask[:, 0] = True
            else:
                raise Exception(f'Invalid choice of mask strategy: {strategy}')
            mask = _mask

        masked = x[mask].reshape(batch, -1, size)

        if keep_zeros:
            masked = self.unmask(masked, mask)
            return masked, mask
        else:
            return masked, mask

    @typechecked
    def unmask(self, x: tt.Masked, mask: tt.Mask) -> tt.Unmasked:
        """
        Unmasks input tensor x, filling in masked values with zeros.
        Returns the masked tensor as well as boolean mask.
        Inputs:
            x:        [ batch, masked_seq_len, size ]
            mask:     [ batch, seq_len ]
        Outputs:
            unmasked: [ batch, seq_len, size ]
        """

        batch, _, size = x.shape
        _, seq_len = mask.shape

        unmasked = torch.zeros((batch, seq_len, size), device=x.device)
        unmasked[mask] = x.flatten(0, 1)

        return unmasked
