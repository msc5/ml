import os
from collections.abc import Callable
from collections import defaultdict

from rich.tree import Tree


class Tree:

    def __init__(self, key: str):
        self.key = key
        self.children = {}

    def get(self, key):
        keys = path.split(os.sep)
        curr = self
        while keys:
            key = keys.pop(0)
            curr = curr.children[key]
        return curr

    def add_child(self, key):
        child = Tree(path.name, path)
        if child.key in self.children:
            raise KeyError
        self.info['n_children'] += 1
        self.children[child.key] = child
        return child

    def add_file(self, path):
        self.info['n_files'] += 1
        self.files.append(Path(path))

    def bfs(
        self,
        callback: Callable[['Tree', any], None] = None,
        params: any = None,
    ):
        """ Performs breadth-first search starting from this node """
        queue = [self]
        while queue:
            curr = queue.pop(0)
            if callback is not None:
                callback(curr, params)
            children = curr.children.items()
            for key, child in children:
                queue.append(child)

    def dfs(
        self,
        callback: Callable[['Tree', any], None] = None,
        params: any = None,
        order: 'pre' or 'post' = 'post',
    ):
        """ Performs depth-first search starting from this node """
        children = self.children.items()
        if callback is not None and order == 'pre':
            callback(self, params)
        for key, child in children:
            child.dfs(callback, None if not params else params.copy(), order)
        if callback is not None and order == 'post':
            callback(self, params)
