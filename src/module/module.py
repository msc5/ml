from __future__ import annotations
from abc import abstractmethod
from contextlib import contextmanager
from typing import Any, Optional, Union

from rich import box
from rich.columns import Columns
from rich.columns import Columns
from rich.console import Group, group
from rich.panel import Panel
from rich.text import Text
import torch
import torch.nn as nn
import os

from ..cli import console
from ..dot import Dot
from ..options import Options, OptionsModule
from ..renderables import Table, check, section
from ..func import ema


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

    _hide_module: bool = False
    _hide_grads: bool = False
    _selected: bool = False

    _commons: int
    _uncommons: int
    _is_common: bool

    _param_count: int = 0
    _out_grad_norm: Optional[float] = None
    _in_grad_norm: Optional[float] = None

    device: torch.device = get_device()

    # -------------------- Private Methods -------------------- #

    def __init__(self, opts: Optional[Union[Options, dict]] = None):
        if isinstance(opts, dict) and opts is not None:
            super(Module, self).__init__(Options(opts))
        else:
            super(Module, self).__init__(opts)

        self.metrics = Dot()
        self.ranges = Dot()

    def _build(self):
        built = super()._build()

        # Check if module is common (Contains no children Modules)
        self._is_common = True
        self._commons = self._uncommons = 0
        for module in self.children():
            if isinstance(module, Module) and not module._hide_module:
                self._uncommons += 1
                self._is_common = False
            else:
                self._commons += 1

        # Register gradient hook
        super().register_full_backward_hook(self._grad_update_hook)

        # Count parameters
        self._param_count = sum(p.numel() for p in self.parameters() if p.requires_grad)

        # Add child metrics to self
        for name, child in self.named_children():
            if isinstance(child, Module):
                self.metrics[name] = child.metrics
                self.ranges[name] = child.ranges

        # Initialize optimizers
        self.optimizers = Dot(self.init_optimizers())

        from ..trainer import CurrentTrainer
        if CurrentTrainer is not None:
            self.progress = CurrentTrainer.progress
        else:
            raise Exception('No Current Trainer!')

        # Move self to device
        self.to(self.device)

        return built

    def _grad_update_hook(self, module: nn.Module, in_grad, out_grad):
        if any([p.requires_grad for p in module.parameters()]):

            out_grad = out_grad[0] if isinstance(out_grad, tuple) and len(out_grad) > 0 else None
            if out_grad is not None:
                # self._out_grad_norm = float(ema(self._out_grad_norm, out_grad.detach().norm().item()))
                self._out_grad_norm = out_grad.detach().norm().item()
            # else:
            #     self._out_grad_norm = None

            in_grad = in_grad[0] if isinstance(in_grad, tuple) and len(in_grad) > 0 else None
            if in_grad is not None:
                # self._in_grad_norm = float(ema(self._in_grad_norm, in_grad.detach().norm().item()))
                self._in_grad_norm = in_grad.detach().norm().item()
            # else:
            #     self._in_grad_norm = None

    # def __setattr__(self, key: str, val: Any):
    #     if isinstance(val, nn.Module):
    #         self.add_module(key, val)
    #     object.__setattr__(self, key, val)

    # -------------------- Abstract Methods -------------------- #

    @abstractmethod
    def init_optimizers(self):
        return iter(())

    # -------------------- Public Methods -------------------- #

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
            target_param.data.copy_((1.0 - p) * target_param.data + p * learner_param.data)

    def add_module(self, name: str, module: Module | nn.Module, hide: bool = False):

        # Add a torch module
        super().add_module(name, module)

        # Add to _children for OptionsModule to initialize
        if isinstance(module, Module):
            self._children[name] = module
            if not module._is_built:
                module._build()
                module._hide_module = hide

    @contextmanager
    def freeze(self):
        """
        Context manager that freezes module's parameters and returns them to
        previous condition when exited.
        """

        states: dict[str, bool] = {}

        # Freeze parameters
        try:
            for name, param in self.named_parameters():
                states[name] = param.requires_grad
                param.requires_grad_(False)
            yield None

        finally:
            for name, param in self.named_parameters():
                param.requires_grad_(states[name])

    def get_mlmodule(self, key: str):
        if self.__class__.__name__ == key:
            return self
        else:
            for child in self._children.values():
                if isinstance(child, Module):
                    if (result := child.get_mlmodule(key)) is not None:
                        return result

    # --------------------  Rendering --------------------  #

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

        yield msg

    @group()
    def _render(self):

        table = Table(box=box.ROUNDED, style='black')
        uncommons = []

        for name, child in self.named_children():
            if isinstance(child, Module) and not child._hide_grads and not child._hide_module:
                if child._is_common:
                    heading = Text(name, style='yellow')
                    module = Text(child.__class__.__name__, style='italic yellow')
                    table.add_row(heading, module, child._render_device(), child._render())
                else:
                    heading = [Text(name, style='bold cyan')]
                    heading += [Text(child.__class__.__name__, style='italic cyan')]
                    heading = Columns([*heading, child._render_params()], padding=(0, 2))
                    panel = Group(heading, child._render())
                    uncommons.append(panel)

        has_uncommons = len(uncommons) > 0
        has_commons = table.row_count > 0

        if has_commons and not has_uncommons:
            yield table
        elif not has_commons and has_uncommons:
            if len(uncommons) == 1:
                yield Group(*uncommons)
            else:
                yield Panel(Group(*uncommons), border_style='black')
        elif has_commons and has_uncommons:
            yield Panel(Group(table, *uncommons), border_style='black')
        else:
            yield self._render_params()

    def __rich__(self):
        return self._render()

    # def __str__(self) -> str:
    #     return self.__class__.__name__

    # def __repr__(self) -> str:
    #     return self.__class__.__name__
