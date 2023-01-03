import abc
from inspect import isclass
from typing import Optional

from torch import nn

from .dot import Dot, DotItem
from ..cli import console


class Options (Dot):
    pass


class OptionsModule:

    _children: dict

    def __init__(self, opts: Optional[Options] = None):
        """ 
        Recursively initialize all annotated and inherited OptionsModules
            Calls pre() hook on the way down 
        """
        super().__init__()

        self.opts = opts if opts is not None else Options()
        self.opts = self._add_defaults(self.opts)
        self.pre(self.opts)

        self._bind_opts(self.opts)
        self._children = {}
        for k, child in self._get_annotations().items():
            if isclass(child) and issubclass(child, OptionsModule):
                if k in self.opts:
                    child = child(self.opts[k])
                    self._children[k] = child

    @classmethod
    def _get_annotations(cls):

        # Collect annotated
        annotated = {**cls.__annotations__}

        # Collect inherited OptionsModules
        for parent in cls.__bases__:
            if (issubclass(parent, OptionsModule)
                    and parent is not OptionsModule
                    and parent is not cls):
                for k, p in parent._get_annotations().items():
                    if isclass(p) and issubclass(p, OptionsModule):
                        annotated[k] = p

        return annotated

    @classmethod
    def _add_defaults(cls, opts: Options):

        for (k, v) in cls._get_annotations().items():
            if (v in [int, float, str] and not k in opts and k[0] != '_'):
                attr = getattr(cls, k, None)
                if attr is not None:
                    opts[k] = attr

        return opts

    def _bind_opts(self, opts: Options):
        """ 
        Binds opts to self
        """

        for (k, v) in vars(opts).items():
            if k != '_parent':
                if isinstance(v, DotItem):
                    setattr(self, k, v.value)
                    self.__dict__[k] = v.value

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
        return self

    @abc.abstractclassmethod
    def pre(self, o: Options):
        return o

    @abc.abstractclassmethod
    def build(self):
        return

    @classmethod
    def required(cls, opts: Optional[Options] = None) -> Options:

        annotated = cls._get_annotations()

        required = Options()
        for k, child in annotated.items():
            if isclass(child) and issubclass(child, OptionsModule):
                required[k] = child.required()
            else:
                if k != 'self':
                    if (k in cls.__dict__ and cls.__dict__[k] is not None):
                        value = cls.__dict__[k]
                    else:
                        value = None
                    required[k] = DotItem(key=k, value=value, type=child)

        if opts is not None:
            for (k_req, v_req) in required:
                for (k_opt, v_opt) in opts:
                    if k_req == k_opt:
                        v_req.value = v_opt.value
                        break

        if isinstance(cls, type) and hasattr(cls, 'opts'):
            for (k_req, v_req) in required:
                for (k_opt, v_opt) in cls.__dict__['opts']:
                    if k_req == k_opt:
                        v_req.value = v_opt.value

        return required
