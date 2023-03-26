from __future__ import annotations
from copy import deepcopy
from dataclasses import dataclass
from inspect import isclass
import time
from typing import Any, Callable, Iterable, Optional

from rich import box
from rich.console import group
from rich.table import Table
from rich.text import Text
import torch


from ..cli import console
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

    def _row(self):
        _key = f'[yellow]{self.key}'
        _value = self._format_value()
        _type = self._format_type()
        _active = Status('closed' if self._is_active() else 'open')
        if _type is not None:
            return _key, _type, _active, _value
        else:
            return _key, _value

    def __deepcopy__(self, *_):
        return self

    def __copy__(self):
        return DotItem(self.key, self.value)


class Dot (object):
    """
    Implements a dot dictionary. Attributes can be set and accessed using
    dictionary or dot notation.
    """

    _renderable: Any = None
    _order: int = 0

    def __init__(self, *args, **kwargs):
        super().__init__()
        object.__setattr__(self, '_size', 0)
        object.__setattr__(self, '_parent', self)
        self._update(*args, **kwargs)

    def _update(self, *args, **kwargs):
        """ 
        Updates dot dictionary with provided args and kwargs.
        """
        args = list(args)
        for a in args:
            if isinstance(a, Dot):
                for (key, value) in a._items():
                    self.__setattr__(key, value)
                args.remove(a)
        input = {**dict(*args), **kwargs}
        for (key, value) in input.items():
            self.__setattr__(key, value)
        return self

    def _soft_update(self, *args, **kwargs):
        """ 
        Updates dot dictionary with provided args and kwargs. Does not replace
        existing values.
        """
        def soft(key: str):
            return not key in self or self[key] == None
        args = list(args)
        for a in args:
            if isinstance(a, Dot):
                for (key, value) in a._items():
                    if soft(key):
                        self.__setattr__(key, value)
                args.remove(a)
        input = {**dict(*args), **kwargs}
        for (key, value) in input.items():
            if soft(key):
                self.__setattr__(key, value)
        return self

    def _items(self):
        """
        Returns shallow iterator over public key-value pairs in Dot.
        """
        return {k: v for k, v in vars(self).items() if k[0] != '_'}.items()

    def _dotitems(self):
        """
        Returns shallow iterator over public DotItems in Dot.
        """
        return self.__class__({k: v.value for k, v in vars(self).items()
                               if (k[0] != '_' and isinstance(v, DotItem))})

    def _dict(self):
        """
        Returns a deep dictionary of public key-value pairs in Dot.
        """
        obj = {}
        for key, val in self._items():
            if isinstance(val, Dot):
                obj[key] = val._dict()
            elif isinstance(val, DotItem):
                obj[key] = val.value
        return obj

    def _dfs(self,
             preorder: Optional[Callback],
             inorder: Optional[Callback],
             postorder: Optional[Callback],
             data: dict = {}):
        """
        Performs Depth-First Search starting from self.
        """

        if preorder is not None: preorder(data)
        for (_, v) in self._items():
            if isinstance(v, Dot):
                if inorder is not None: inorder(data)
                v.dfs(v, preorder=preorder, postorder=postorder)
        if postorder is not None: postorder(data)

    def _set_renderable(self, renderable: Any):
        object.__setattr__(self, '_renderable', renderable)

    def _table(self, key: Optional[str] = None):

        is_child = key is not None

        if is_child:
            title = self._renderable or key
            table = Table(title=title, title_style='cyan', box=None,
                          padding=(0, 0, 0, 2), title_justify='left',
                          show_header=False, style='cyan', expand=True)
        else:
            table = Table(title=None, title_style='white', box=box.ROUNDED,
                          show_header=False, style='black', expand=True)

        items, dots = [], []
        items_order, dots_order = [], []
        for (k, v) in sorted(self._items()):
            if isinstance(v, Dot):
                dots += [v._table(k)]
                dots_order += [v._order]
            elif isinstance(v, DotItem):
                items += [v._row()]
                items_order += [v._order]

        # dots order (need to refactor this whole thing)
        dots = [x for _, x in sorted(zip(dots_order, dots), key=lambda x: x[0])]
        items = [x for _, x in sorted(zip(items_order, items), key=lambda x: x[0])]

        if items != []:
            subtable = Table(show_header=False, box=None, pad_edge=False, padding=(0, 2))
            for row in items:
                subtable.add_row(*row)
            table.add_row(subtable)
        for dot in dots:
            table.add_row(dot)

        if is_child:
            if items == [] and dots == []:
                if self._renderable is not None:
                    return self._renderable
                else:
                    return Text(key, style='cyan')
            else:
                return table
        else:
            if items == [] and dots == []:
                return ''
            else:
                return table

    def __call__(self, *args, **kwargs):
        map = deepcopy(self)
        map._update(*args, **kwargs)
        return map

    def __iter__(self):
        """
        Returns an iterable over key-value pairs of all items in Dot. Items are
        returned in DFS postorder, and each level is separated by "/".
        """

        def dfs(dot, parent):
            keys = []
            for (k, v) in dot._items():
                if isinstance(v, Dot):
                    keys += dfs(v, parent + '/' + k)
                elif isinstance(v, DotItem):
                    keys += [(parent + '/' + v.key, v)]
            return keys

        return iter(dfs(self, ''))

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

        if key[0] == '_':
            object.__setattr__(self, key, value)
            return

        # If key has a dot in it (e.g., 'a.b'), then wrap in another Dot
        if '.' in key:
            splits = key.split('.')
            if len(splits) > 1:
                if splits[0] in self:
                    val = self[splits[0]]
                else:
                    val = self.__class__()
                val.__setattr__('.'.join(splits[1:]), value)
                value = val
                key = splits[0]

        # For replacements
        if key in self:
            # _get_dotitem()
            v = object.__getattribute__(self, key)
            if not isinstance(v, DotItem) and not isinstance(v, self.__class__):
                raise Exception(f'_get_dotitem returned {v.__class__}')
            if isinstance(v, DotItem):
                v.value = value
                if isinstance(value, DotItem):
                    v.value = value.value
            elif isinstance(v, self.__class__):
                if isinstance(value, dict):
                    object.__setattr__(self, key, self.__class__(value))
                else:
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

    def __getattribute__(self, key):
        if isinstance(key, int):
            key = str(key)
        v = object.__getattribute__(self, key)
        if key[0] == '_':
            return v
        elif isinstance(v, DotItem):
            return v.value
        return v


Callback = Callable[[dict], None]
