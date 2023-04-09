from __future__ import annotations
import abc
import argparse
from inspect import isclass
import sys
from typing import Any, Optional, cast, get_origin
from rich.columns import Columns
from rich.console import Group
from rich.panel import Panel
from rich.text import Text

from torch import nn


from ..cli import console
from ..dot import Dot, DotItem
from ..renderables import Status


class Options (Dot):
    pass


class OptionsItem (DotItem):
    pass


def is_primitive(v: type):

    if v in (int, float, str, bool):
        return True
    if get_origin(v) in (list, tuple):
        return True

    return False


class OptionsModule:

    opts: Options

    _children: dict
    _is_built: bool = False
    _is_building: bool = False

    def __init__(self, opts: Optional[Options] = None):
        """ 
        Recursively initialize all annotated and inherited OptionsModules
            Calls pre() hook on the way down 
        """
        super().__init__()
        self.apply_opts(opts)

    @abc.abstractmethod
    def pre(self, o: Options):
        return o

    @abc.abstractmethod
    def build(self):
        return

    def apply_opts(self, opts: Optional[Options] = None):

        self.opts = opts if opts is not None else Options()
        self.opts = self._get_defaults(self.opts)
        self.pre(self.opts)
        self._bind_opts(self.opts)

        self._children = {}
        for k, child in self._get_annotations().items():
            if isclass(child) and issubclass(child, OptionsModule):
                if k in self.opts:
                    # child = child(self.opts._dotitems()._soft_update(self.opts[k]))
                    child = child(self.opts._dotitems()._update(self.opts[k]))
                else:
                    child = child(self.opts._dotitems())
                self._children[k] = child

    @classmethod
    def _get_annotations(cls):
        """ 
        Collects class annotations by combining own annotations with inherited
        annotations
        """

        # Collect annotated
        annotated = {**cls.__annotations__}

        # Collect inherited OptionsModules annotations
        for parent in cls.__bases__:
            if (issubclass(parent, OptionsModule)
                    and parent is not OptionsModule
                    and parent is not cls):
                for k, p in parent._get_annotations().items():
                    if not k in annotated:
                        annotated[k] = p

        return annotated

    @classmethod
    def _get_defaults(cls, opts: Options):
        """ 
        Goes through class annotations and adds default values (if they exist)
        to opts.
            -> Primitive defaults are added to opts
            -> OptionsModules are instantiated
        """

        for (k, v) in cls._get_annotations().items():
            if k[0] != '_' and not k in opts:
                if is_primitive(v):
                    attr = getattr(cls, k, None)
                    if attr is not None:
                        opts[k] = attr

        return opts

    @classmethod
    def parse(cls):
        """
        Uses class annotations in order to build an argument parser which
        accepts all primitive annotations as arguments. Parses from sys.argv
        and returns an Options dictionary of provided arguments.
        """

        parser = argparse.ArgumentParser(prog='Options')

        def add_argument(k: str, v: Any):
            tag = f'--{k}'
            if v == bool:
                default = getattr(cls, k, None)
                parser.add_argument(tag, action='store_true', default=default)
            elif v == str:
                parser.add_argument(tag, type=v, default=None)
            elif v in [int, float]:
                parser.add_argument(tag, type=v, default=None)
            elif v == list[str]:
                parser.add_argument(tag, nargs='+', default=None)

        def recur(module: type[OptionsModule], prefix: str = ''):
            p = prefix + '.' if prefix != '' else ''
            for k, v in module._get_annotations().items():
                if k[0] != '_':
                    if isclass(v) and (issubclass(v, OptionsModule)
                                       and v is not OptionsModule
                                       and v is not module):
                        recur(v, p + k)
                    else:
                        add_argument(p + k, v)

        recur(cls)
        args = parser.parse_args(sys.argv[2:])
        args = {k: v for k, v in args.__dict__.items() if v is not None}
        args = Options(args)
        return args

    @classmethod
    def required(cls) -> Options:
        """
        Uses class annotations to build an Options dictionary representing all
        required parameters of OptionsModule.
        """

        annotations = cls._get_annotations()

        required = Options()
        for k, v in annotations.items():
            if isclass(v) and (issubclass(v, OptionsModule)
                               and v is not OptionsModule
                               and v is not cls):
                required[k] = v.required()
            else:
                required[k] = DotItem(k, getattr(cls, k, None), v)

        return required

    def _bind_opts(self, opts: Options):
        """ 
        Binds opts to self and children
        """

        for (k, v) in vars(opts).items():
            if k != '_parent':
                if isinstance(v, DotItem):
                    setattr(self, k, v.value)
                    self.__dict__[k] = v.value

    def _gather_opts(self):
        opts = self.opts()
        for name, child in self._children.items():
            opts[name] = child._gather_opts()
        return opts

    def _gather_params(self):
        params = Dot()
        for key, val in vars(self).items():
            if key[0] != '_':
                if isinstance(val, OptionsModule):
                    params[key] = val._gather_params()
                # elif type(val) in PRIMITIVES:
                elif is_primitive(type(val)):
                    params[key] = val
        return params

    def _render_building(self):

        children = []
        for child in self._children.values():
            children.append(child._render_building())

        if self._is_built:
            state = 'built'
        elif self._is_building:
            state = 'working'
        else:
            state = 'waiting'
        name = self.__class__.__name__

        if len(children) > 0:
            return Panel(Group(*children), border_style='black', title=name, width=50)
        else:
            return Columns([Status(status=state), Text(name)], padding=(0, 3))  # type: ignore

    def _build(self):
        """ 
        Recursively build all annotated and inherited OptionsModules
            Calls build() hook on the way up
        """

        self._is_building = True

        for k, child in self._children.items():
            child._build()
            if isinstance(self, nn.Module) and isinstance(child, nn.Module):
                self.add_module(k, child)
            setattr(self, k, child)
            self.__dict__[k] = child

        self = cast(OptionsModule, self)
        self.build()
        self._is_built = True
        self._is_building = False

        return self
