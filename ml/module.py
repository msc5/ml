from typing import Literal, Optional, cast
from rich.console import group
import torch
import torch.nn as nn

from .options import Dot, Options, OptionsModule

from .dist import Distribution


class Param:

    count: int = 0
    grad: Optional[torch.Tensor] = None

    @group()
    def __rich__(self):
        msg = f'[magenta]({self.count:,})[reset]'
        if self.grad is not None:
            norm, zero = self.grad.norm(), self.grad.abs().sum() == 0
            if not zero:
                msg += ' <- ' + f'[green]𝝯 {norm.item():5.2e}'
        yield msg


def get_device():
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    # if torch.backends.mps.is_available():
    #     device = 'mps'
    return device


class Module (OptionsModule, nn.Module):

    _hide_grads: bool = False

    device: Literal['cpu', 'cuda', 'mps'] = get_device()
    samples: torch.Tensor

    def __init__(self, opts: Optional[Options] = None):
        super(Module, self).__init__(opts)

        self.metrics = Dot()
        self.params = Param()

        super().register_full_backward_hook(self._get_grads)

    def _params(self, freeze: list[str] = []):
        params = {}
        for name, child in self.named_children():
            if not any([frozen in name for frozen in freeze]):
                if isinstance(child, Module) and not child._hide_grads:
                    # params[f'{name} [blue]({child.params.count:,})[reset]'] = child.params
                    p = child._params(freeze)
                    if len(p) != 0:
                        params[f'{name} [blue]({child.params.count:,})[reset]'] = p
                    else:
                        params[name] = child.params
        return Dot(params)

    def _get_grads(self, _: nn.Module, in_grad, out_grad):
        in_grad = in_grad[0] if isinstance(in_grad, tuple) else None
        out_grad = out_grad[0] if isinstance(out_grad, tuple) else None
        if out_grad is not None:
            self.params.grad = out_grad.detach()

    def _build(self):
        built = super()._build()

        count = 0
        for p in self.parameters():
            count += torch.tensor(p.shape).prod().item()
        self.params.count = cast(int, count)

        # for k, v in self.named_children():
        #     if isinstance(v, Module):
        #         if not hasattr(v, '_hide_grads'):
        #             self.params[k] = v.params

        return built


class ProbabilisticModule (Module):

    def dist(self, x: torch.Tensor, *args, **kwargs) -> Distribution:
        return Distribution(x, *args, **kwargs)


class Optimizers (Dot):

    _freeze: list[str] = []

    def zero_grad(self):
        for key, o in self._items():
            if key != '_freeze':
                o.value.zero_grad()

    def optimize(self, key: str, loss: torch.Tensor, **kwargs):
        if not key in self._freeze:
            loss.backward(**kwargs)
            self[key].step()
