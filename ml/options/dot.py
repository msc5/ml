from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Optional

from rich import box
from rich.table import Table
from rich.text import Text
import torch


@dataclass
class DotItem:

    key: str
    value: Any
    type: Optional[Any]
    default: Optional[Any]

    def __init__(self,
                 key: str,
                 value: Any,
                 type: Optional[Any] = None,
                 default: Optional[Any] = None):

        if isinstance(value, DotItem):
            raise Exception('Tried to create nested DotItem!')
        self.key = key
        self.value = value
        self.type = type
        self.default = default

    def __rich__(self):
        key = f'[dot.key]{self.key}'
        type = f'[dot.type]{self.type}'
        if isinstance(self.value, torch.Tensor):
            if len(self.value.shape) <= 1:
                v = self.value
            else:
                v = self.value.shape
        else:
            v = self.value
        if isinstance(v, float):
            v = f'{v:5.5f}'
        value = (f'[dot.complete]{v}' if v is not None else f'[dot.missing]{v}')
        if self.type is not None:
            return f'{key} {type} : {value}'
        else:
            return f'{key} : {value}'

    def row(self):
        key = f'[dot.key]{self.key}'
        type = f'[dot.type]{self.type}'
        if hasattr(self.value, '__rich__'):
            value = self.value
        else:
            if isinstance(self.value, torch.Tensor):
                if len(self.value.shape) <= 1:
                    v = self.value
                else:
                    v = self.value.shape
            elif isinstance(self.value, float):
                v = f'{self.value:5.5f}'
            else:
                v = self.value
            value = (f'[dot.complete]{v}' if v is not None else f'[dot.missing]{v}')
        if self.type is not None:
            return key, type, value
        else:
            return key, value

    def __deepcopy__(self, *_):
        return self

    def __copy__(self):
        return DotItem(self.key, self.value)


class Dot (object):
    """
    Implements a self-referencing dot-dict. Attributes can be set and accessed
    using dictionary or dot notation.
    """

    def __init__(self, *args, **kwargs):
        super().__init__()
        object.__setattr__(self, '_size', 0)
        object.__setattr__(self, '_parent', self)
        self._update(*args, **kwargs)

    def _update(self, *args, **kwargs):
        args = list(args)
        for a in args:
            if isinstance(a, Dot):
                for (key, value) in self._items():
                    self.__setattr__(key, value)
                args.remove(a)
        input = {**dict(*args), **kwargs}
        for (key, value) in input.items():
            self.__setattr__(key, value)

    def _items(self):
        return {k: v for k, v in vars(self).items() if k[0] != '_'}.items()

    def _dict(self):
        obj = {}
        for key, val in self._items():
            if isinstance(val, DotItem):
                obj[key] = val.value
            elif isinstance(val, Dot):
                obj[key] = val._dict()
        return obj

    def __call__(self, *args, **kwargs):
        map = deepcopy(self)
        map._update(*args, **kwargs)
        return map

    def __iter__(self):

        def dfs(dot, parent):
            keys = []
            for (k, v) in dot._items():
                if isinstance(v, self.__class__):
                    keys += dfs(v, parent + '/' + k)
                elif isinstance(v, DotItem):
                    keys += [(parent + '/' + v.key, v)]
            return keys

        return iter(dfs(self, ''))

    def _table(self, key: Optional[str] = None):

        is_child = key is not None

        if is_child:
            table = Table(title=key, title_style='dot.title', box=None,
                          padding=(0, 0, 0, 2), title_justify='left',
                          show_header=False, style='dot.title')
        else:
            table = Table(title=None, title_style='white', box=box.ROUNDED,
                          show_header=False, style='dot.border', expand=True)

        items, dots = [], []
        for (k, v) in sorted(self._items()):
            if isinstance(v, Dot):
                dots += [v._table(k)]
            elif isinstance(v, DotItem):
                items += [v.row()]

        if items != []:
            subtable = Table(show_header=False, box=None, pad_edge=False, padding=(0, 2))
            for row in items:
                subtable.add_row(*row)
            table.add_row(subtable)
        for dot in dots:
            table.add_row(dot)

        if is_child:
            if items == [] and dots == []:
                return Text(key, style='dot.title')
            else:
                return table
        else:
            if items == [] and dots == []:
                return ''
            else:
                return table

    def __rich__(self):
        return self._table()

    def __contains__(self, key):
        return key in vars(self)

    def __len__(self):
        return self._size

    # Set Methods --------------------------------------------------------------

    def __setitem__(self, key, value):
        self.__setattr__(key, value)

    def __setattr__(self, key, value):

        if isinstance(key, int):
            key = str(key)

        # For replacements
        if key in self:
            v = self._get_map_item(key)
            if isinstance(v, DotItem):
                v.value = value
            elif isinstance(v, self.__class__):
                object.__setattr__(self, key, value)
            return

        if isinstance(value, DotItem):
            v = value
        elif isinstance(value, dict):
            v = self.__class__(value)
        elif isinstance(value, self.__class__):
            v = value
            if v._parent == v:
                object.__setattr__(v, '_parent', self)
        else:
            v = DotItem(key, value)
        object.__setattr__(self, key, v)
        object.__setattr__(self, '_size', self._size + 1)

    # Get Methods --------------------------------------------------------------

    def __getitem__(self, key):
        return self.__getattribute__(key)

    def _get_map_item(self, key):
        v = object.__getattribute__(self, key)
        if isinstance(v, DotItem) or isinstance(v, Dot):
            return v
        else:
            raise Exception(f'_get_map_item returned {v.__class__}')

    def __getattribute__(self, key):
        if isinstance(key, int):
            key = str(key)
        v = object.__getattribute__(self, key)
        if isinstance(v, DotItem):
            return v.value
        return v
