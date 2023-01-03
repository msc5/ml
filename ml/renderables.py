from collections.abc import Callable
from multiprocessing.sharedctypes import SynchronizedBase
from typing import Optional, Union

from rich import box
from rich.console import group
from rich.padding import Padding
import rich.progress as progress
import rich.table
import torch.multiprocessing as mp
import torch.multiprocessing.queue as tmpq
import multiprocessing.managers as mpman

from src.ml.options.dot import Dot

from .cli import console


class Progress (progress.Progress):

    def __init__(self, tasks: list[Union[str, int]] = [], total: int = 100, transient: bool = False, *args, **kwargs) -> None:
        super().__init__(progress.TextColumn('{task.description}'),
                         progress.BarColumn(),
                         progress.TaskProgressColumn(), *args, **kwargs)
        # super().__init__(*args, **kwargs)
        self.task_map = {}
        for task in tasks:
            self.add_task(task, total=total, transient=transient)

    def add_task(self, task: Union[str, int], description: Optional[str] = None, *args, **kwargs) -> progress.TaskID:
        if task in self.task_map:
            return self.task_map[task]
        else:
            id = super().add_task(description or '', *args, **kwargs)
            self.task_map[task] = id
            return id

    def remove_task(self, task: Union[str, int]) -> None:
        if task in self.task_map:
            return super().remove_task(self.task_map[task])
        else:
            return None

    def update(self, task: Union[str, int], completed: Optional[int] = None, *args, **kwargs) -> None:
        assert task in self.task_map
        return super().update(task_id=self.task_map[task], completed=completed, *args, **kwargs)


class Process (mp.Process):

    daemon: bool = True

    def __rich__(self):
        if self.is_alive():
            return '[green]⬤  alive'
        else:
            return '[red] stopped'


class Queue (tmpq.Queue):

    def __init__(self, maxsize: int = 10, *args, **kwargs):
        ctx = mp.get_context()
        self._size = ctx.Value('i', 0)
        super(Queue, self).__init__(*args, **kwargs, maxsize=maxsize, ctx=ctx)

    def put(self, *args, **kwargs):
        with self._size.get_lock():
            self._size.value += 1
        return super().put(*args, **kwargs)

    def get(self, *args, **kwargs):
        with self._size.get_lock():
            self._size.value -= 1
        return super().get(*args, **kwargs)

    def size(self):
        return self._size.value


class Manager (mpman.SyncManager):
    Queue = Queue
    Dot = Dot


Manager.register('Queue', Queue)
Manager.register('Dot', Dot, exposed=['__setitem__', '__getitem__', '__call__', '__rich__', '_table'])


class Alive:

    _alive: bool = True
    callback: Optional[Callable]

    def __init__(self, state: bool = True, callback: Optional[Callable] = None) -> None:
        self.callback = callback
        self._alive = state

    def alive(self):
        self._alive = True

    def stopped(self):
        self._alive = False

    def __rich__(self):
        if self.callback is not None and self._alive:
            self._alive = self.callback()
        if self._alive:
            return '[green]⬤  alive'
        else:
            return '[red] stopped'


class Table (rich.table.Table):

    def __init__(self, *args, **kwargs):
        defaults = {'expand': True,
                    'show_header': False,
                    'box': box.SIMPLE,
                    'style': 'blue'}
        super().__init__(*args, **{**defaults, **kwargs})


def section(message: str, module: str = 'Trainer', color: str = 'blue'):
    msg = Padding(f'[ {module} ] {message}', (1, 0), style=color)
    console.print(msg)
