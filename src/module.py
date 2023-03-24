from __future__ import annotations
from typing import Optional

from rich import box
from rich.columns import Columns
from rich.columns import Columns
from rich.console import Group, group
from rich.padding import Padding
from rich.panel import Panel
from rich.text import Text
import torch
import torch.nn as nn

from .cli import console
from .options import Dot, Options, OptionsModule
from .renderables import Alive, Table


def get_device(dev: Optional[str] = None):
    """
    When initializing modules, prefer cuda if it is available.
    """
    if dev is None:
        if torch.cuda.is_available():
            return torch.device('cuda')
        else:
            return torch.device('cpu')
    else:
        return torch.device(dev)


class Module (OptionsModule, nn.Module):
    """
    A Module extends the option-inheritance function of the OptionsModule, but
    has added functionality for use with the PyTorch Library.
    """

    _hide_grads: bool = False
    _selected: bool = False

    _param_count: int = 0
    _out_grad_norm: Optional[float] = None
    _in_grad_norm: Optional[float] = None

    device: torch.device = get_device()

    def __init__(self, opts: Optional[Options] = None, glob: Optional[Options] = None):
        super(Module, self).__init__(opts, glob)
        self.metrics = Dot()

    def _build(self):
        built = super()._build()

        # Check if module is common (Contains no children Modules)
        self._is_common = sum([isinstance(c, Module) for c in self.children()]) == 0

        # Register gradient hook
        super().register_full_backward_hook(self._grad_update_hook)

        # Count parameters
        self._param_count = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return built

    def _grad_update_hook(self, module: nn.Module, in_grad, out_grad):
        if any([p.requires_grad for p in module.parameters()]):

            out_grad = out_grad[0] if isinstance(out_grad, tuple) and len(out_grad) > 0 else None
            if out_grad is not None:
                self._out_grad_norm = out_grad.detach().norm().item()
            else:
                self._out_grad_norm = None

            in_grad = in_grad[0] if isinstance(in_grad, tuple) and len(in_grad) > 0 else None
            if in_grad is not None:
                self._in_grad_norm = in_grad.detach().norm().item()
            else:
                self._in_grad_norm = None

    # ---------------------------------------- Public Methods ---------------------------------------- #

    def select(self):

        # Select all children
        self._selected = True
        for c in self.children():
            if isinstance(c, Module):
                c.select()

    def polyak(self, learner: Module, p: float = 0.005):
        """
        Polyak Averaging
        Updates own weights with those of learner module with identical architecture.
        """

        for target_param, learner_param in zip(self.parameters(), learner.parameters()):
            updated_param = p * learner_param.data + (1.0 - p) * target_param.data
            target_param.data.copy_(updated_param)

    # ---------------------------------------- Rendering ---------------------------------------- #

    @group()
    def _render_device(self):
        msg = Text(self.device.type, style='magenta')
        if self.device.index is not None:
            msg += Text(' : ') + Text(str(self.device.index))
        yield msg

    @group()
    def _render_params(self):

        high, low = 1e-3, 1e-6
        msg = f'[blue]( {self._param_count:,} )[reset]'

        if self._out_grad_norm is not None:
            if self._out_grad_norm > high:
                color = '[green]'
            elif high > self._out_grad_norm > low:
                color = '[yellow]'
            else:
                color = '[red]'
            msg += ' <- ' + color + f'𝝯 {self._out_grad_norm:5.2e}'

        # if self._in_grad_norm is not None:
        #     if self._in_grad_norm > high:
        #         color = '[green]'
        #     elif high > self._in_grad_norm > low:
        #         color = '[yellow]'
        #     else:
        #         color = '[red]'
        #     msg = color + f'𝝯 {self._out_grad_norm:5.2e}' + ' <- ' + msg

        yield msg

    @group()
    def _render(self):

        table = Table(box=box.ROUNDED, style='black')
        uncommons = []

        for name, child in self.named_children():
            if isinstance(child, Module) and not child._hide_grads:
                if child._is_common:
                    heading = Text(name, style='yellow')
                    # selected = Alive(child._selected, true_label='selected')
                    table.add_row(heading, child._render_device(), child._render())
                else:
                    heading = Columns([Text(name, style='bold cyan'), child._render_params()])
                    heading = Padding(heading, (0, 1))
                    panel = Group(heading, child._render())
                    uncommons.append(panel)

        has_uncommons = len(uncommons) > 0
        has_commons = table.row_count > 0

        if has_commons and not has_uncommons:
            yield table
        elif not has_commons and has_uncommons:
            yield Panel(Group(*uncommons), border_style='black')
        elif has_commons and has_uncommons:
            yield Panel(Group(table, *uncommons), border_style='black')
        else:
            yield self._render_params()

    def __rich__(self):
        return self._render()

    def __str__(self) -> str:
        return self.__class__.__name__

    def __repr__(self) -> str:
        return self.__class__.__name__
