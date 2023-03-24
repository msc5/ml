from __future__ import annotations
from abc import abstractmethod
import argparse
from inspect import isclass
import sys
from typing import Any, Optional

from torch import nn

from ..cli import console
from .dot import Dot, DotItem


class Options (Dot):
    pass


class OptionsItem (DotItem):
    pass


class OptionsModule:

    opts: Options
    glob: Options

    _children: dict
    _is_built: bool = False

    def __init__(self, opts: Optional[Options] = None, glob: Optional[Options] = None):
        """ 
        Recursively initialize all annotated and inherited OptionsModules
            Calls pre() hook on the way down 
        """
        super().__init__()
        self.apply_opts(opts, glob)

    # ---------------------------------------- Abstract Methods ---------------------------------------- #

    @abstractmethod
    def pre(self, o: Options):
        return o

    @abstractmethod
    def build(self):
        return

    # ---------------------------------------- Public Methods ---------------------------------------- #

    def apply_opts(self, opts: Optional[Options] = None, glob: Optional[Options] = None):
        """ 
        Recursively applies provided opts and glob to self and all children.
        """

        breakpoint()

        # Initialize opts and glob if they do not exist
        self.opts = opts if opts is not None else Options()
        self.glob = glob if glob is not None else Options()

        # Initialize Options
        # 1. Collect default options from annotations into opts
        # 2. Call pre(o, g) function and possibly mutate opts, glob
        # 3. Bind opts to self
        self.opts = self._get_defaults(self.opts)
        self.pre(self.opts)
        self._bind_opts(self.opts)

        # Collect and initialize annotated OptionsModules
        self._children = {}
        for name, child in self._get_submodules().items():

            # Initialize child and add to cache
            child = child(self.opts, self.glob)
            self._children[name] = child

    # ---------------------------------------- Static Methods ---------------------------------------- #

    @classmethod
    def _get_submodules(cls):
        """ 
        Collects annotated OptionsModules by checking annotations.
        """

        submodules = {}
        for name, child in cls._get_annotations().items():
            if isclass(child) and issubclass(child, OptionsModule):
                submodules[name] = child

        return submodules

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
                if v in [int, float, str, bool, list]:
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

    # ---------------------------------------- Private Methods ---------------------------------------- #

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
        """ 
        Recursively traverse OptionsModule and collects opts.
        """
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
                elif type(val) in [bool, int, float, str, list]:
                    params[key] = val
        return params

    def _build(self):
        """ 
        Recursively build all annotated and inherited OptionsModules
            Calls build() hook on the way up
        """

        for k, child in self._children.items():
            child._build()
            if isinstance(self, nn.Module) and isinstance(child, nn.Module):
                self.add_module(k, child)
            setattr(self, k, child)
            self.__dict__[k] = child

        self.build()
        self._is_built = True
        return self
