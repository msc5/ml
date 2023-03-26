from dataclasses import dataclass
from inspect import isclass
from typing import Any, Iterable, Optional
from rich.console import RenderableType

import torch

from ..renderables import Status


@dataclass
class DotItem:

    key: str
    value: Any

    _type: Optional[Any]
    _default: Optional[Any]
    _none_allowed: bool = False

    _order: int = 0

    def __init__(self,
                 key: str,
                 value: Any,
                 type: Optional[Any] = None,
                 default: Optional[Any] = None):

        if isinstance(value, DotItem):
            raise Exception('Tried to create nested DotItem!')
        self.key = key
        self.value = value
        self._type = type
        self._default = default

    def _format_value(self):
        if hasattr(self.value, '__rich__'):
            v = self.value
        else:
            v = self.value
            if self.value is None:
                v = ''
            elif isinstance(self.value, str):
                v = f'\'{self.value}\''
            elif isinstance(self.value, float):
                v = f'{self.value:5.5f}'
            elif isinstance(self.value, torch.Tensor):
                if len(self.value.shape) == 0:
                    v = self.value.item()
                else:
                    v = self.value.shape
            elif isinstance(self.value, torch.nn.Module):
                v = self.value.__class__.__name__
            elif isclass(self.value):
                v = f'{self.value.__name__}'
            v = f'[green]{v}'
        return v

    def _format_type(self):

        def _format_args(args: Iterable):
            t = []
            for arg in args:
                if isclass(arg):
                    t.append(arg.__name__)
                elif hasattr(arg, '__args__'):
                    t.append(_format_args(arg.__args__))
                elif isinstance(arg, str):
                    t.append(f'\'{arg}\'')
                else:
                    t.append(str(arg))
            t = '(' + ' | '.join(t) + ')'
            return t

        t = None
        if self._type is not None:
            t = self._type
            if isclass(self._type):
                t = self._type.__name__
            elif hasattr(self._type, '__module__'):
                if self._type.__module__ == 'typing':
                    if hasattr(self._type, '__args__'):
                        if type(None) in t.__args__:
                            self._none_allowed = True
                        t = _format_args(t.__args__)
                    elif hasattr(self._type, '_name'):
                        t = self._type._name
            t = f'[blue]{t}'
        return t

    def _is_active(self):
        is_active = True
        if self.value is None:
            if self._none_allowed:
                is_active = True
            else:
                is_active = False
        if is_active:
            self._order = 1
        return is_active

    def _row(self) -> Iterable[RenderableType]:
        _key = f'[yellow]{self.key}'
        _value = self._format_value()
        _type = self._format_type()
        # _active = Status('closed' if self._is_active() else 'open')
        if _type is not None:
            return _key, _type, _value
        else:
            return _key, _value

    def __deepcopy__(self, *_):
        return self

    def __copy__(self):
        return DotItem(self.key, self.value)
