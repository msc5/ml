from __future__ import annotations
from copy import deepcopy
from typing import Any, Callable, Optional
from rich.columns import Columns

from rich.console import Group
from rich.padding import Padding
from rich.panel import Panel
from rich.text import Text

from ..renderables import Table
from ..cli import console
from .dotitem import DotItem


class Dot (object):
    """
    Implements a dot dictionary. Attributes can be set and accessed using
    dictionary or dot notation.
    """

    _name: Optional[str] = None
    _parent: Optional[Dot] = None

    def __init__(self, *args, **kwargs):
        super().__init__()
        self._update(*args, **kwargs)

    def _update(self, *args, **kwargs):
        """
        Updates dot dictionary with provided args and kwargs.
        (Overwrites existing data.)
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
        Updates dot dictionary with provided args and kwargs.
        (Does not overwrite existing data.)
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
            if isinstance(val, Dot) and len(val) > 0:
                obj[key] = val._dict()
            elif isinstance(val, DotItem):
                obj[key] = val.value
        return obj

    def _list(self, root: str = '', delimiter: str = '/') -> list[tuple[str, Any]]:
        """
        Returns a deep list of public key-value pairs in Dot.
        """
        items = []
        for key, val in self._items():
            if isinstance(val, Dot) and len(val) > 0:
                items += val._list(root=(root + key))
            elif isinstance(val, DotItem):
                items += [(root + delimiter + val.key, val.value)]
        return items

    def __iter__(self):
        """
        Returns an iterable over deep key-value pairs of all items in Dot.
        """
        yield from self._list()

    def __call__(self, *args, **kwargs):
        map = deepcopy(self)
        map._update(*args, **kwargs)
        return map

    def __contains__(self, key):
        return key in vars(self)

    def __len__(self):
        count = 0
        for _, child in self._items():
            if isinstance(child, Dot):
                count += len(child)
            elif isinstance(child, DotItem):
                count += 1
        return count

    # -------------------- Rendering -------------------- #

    def _render(self, key: Optional[str] = None):

        is_child = key is not None

        table = Table(box=None, style='black', expand=False)

        render = []
        dots = []
        for (k, v) in sorted(self._items()):

            if isinstance(v, Dot) and len(v) > 0:
                r = [Columns([Text(k, style='bold blue'), Text(f'({len(v)})', style='magenta')], padding=(0, 2))]
                r += [Padding(v._render(k), (0, 2))]
                dots += [Group(*r)]

            elif isinstance(v, DotItem):
                table.add_row(*v._row())

        render += dots
        if table.row_count > 0:
            render.append(table)

        if is_child:
            return Group(*render)
        else:
            return Panel(Group(*render), border_style='black')

    def __rich__(self):
        return self._render()

    # -------------------- Set Methods -------------------- #

    def __setitem__(self, key, value):
        self.__setattr__(key, value)

    def __setattr__(self, key, value):

        key = str(key)

        # Set a private variable (directly, with no wrapping DotItem)
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
            if v._parent == None:
                object.__setattr__(v, '_parent', self)
        else:
            v = DotItem(key, value)
        object.__setattr__(self, key, v)

    # -------------------- Get Methods -------------------- #

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


Callback = Callable[[Dot, dict], None]
