from __future__ import annotations
from copy import deepcopy
from typing import Callable, Optional

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

    _size: int = 0

    _name: Optional[str] = None
    _parent: Optional[Dot] = None

    def __init__(self, *args, **kwargs):
        super().__init__()
        object.__setattr__(self, '_size', 0)
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

    def __contains__(self, key):
        return key in vars(self)

    def __len__(self):
        return self._size

    # -------------------- Rendering -------------------- #

    def _render(self, key: Optional[str] = None):

        is_child = key is not None

        table = Table(box=None, style='black', expand=False)

        render = []
        dots = []
        for (k, v) in sorted(self._items()):

            if isinstance(v, Dot) and len(v) > 0:
                title = Text(k, style='bold blue')
                dots += [Group(title, Padding(v._render(k), (0, 2)))]

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
            object.__setattr__(self, '_size', self._size + v._size)
        else:
            v = DotItem(key, value)
        object.__setattr__(self, key, v)
        object.__setattr__(self, '_size', self._size + 1)

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


Callback = Callable[[dict], None]


if __name__ == "__main__":

    o = Dot()

    o.env = 'halfcheetah-expert-v2'
    o.size = (5, 12)
    o.seq_len = 50

    o['m.u'] = 3

    a = o.agent = Dot()
    a.name = 'random-walker-2'
    a.seq_len = 500

    d = o.dataset = Dot()
    g = d.general = Dot(name='Matthew')
    p = o.post = Dot()

    console.log(o)
