from __future__ import annotations
import multiprocessing.managers as mpman
import threading
from typing import Callable, Iterable, Optional

from rich import box
from rich.columns import Columns
from rich.console import Group, group
from rich.panel import Panel
import rich.progress as progress
import torch.multiprocessing as mp
import torch.multiprocessing.queue as tmpq

from .cli import console
from .io import generate_name
from .options.dot import Dot
from .renderables import Alive, Progress, Table

COLUMN_WIDTH = 20


class ProcessInfo:
    """
    Maintains references to current process objects.
    """

    process: Process | None = None
    manager: Manager | None = None


class ThreadInfo (threading.local):
    """
    Maintains references to current thread objects.
    """

    thread: Thread | None = None
    queues: dict[str, dict[str, ManagedQueue]] = {}
    n_procs: int = 0

    update_thread: Thread | None = None
    update_queue: ManagedQueue | None = None
    update_thread_start: threading.Event = threading.Event()


_pinfo: ProcessInfo = ProcessInfo()
_tinfo: ThreadInfo = ThreadInfo()


class ManagedQueue (tmpq.Queue):
    """
    Subclass of torch multiprocessing Queue which tracks throughput.
    """

    def __init__(self, maxsize: int = 100, *args, **kwargs):
        ctx = mp.get_context()
        self._maxsize = maxsize
        self._size = ctx.Value('i', 0)
        self._total_in = ctx.Value('i', 0)
        self._total_out = ctx.Value('i', 0)
        super(ManagedQueue, self).__init__(*args, **kwargs, maxsize=maxsize, ctx=ctx)

    def put(self, *args, **kwargs):
        super().put(*args, **kwargs)
        with self._size.get_lock():
            self._size.value += 1  # type: ignore
        with self._total_in.get_lock():
            self._total_in.value += 1  # type: ignore

    def get(self, *args, **kwargs):
        item = super().get(*args, **kwargs)
        with self._size.get_lock():
            self._size.value -= 1  # type: ignore
        with self._total_out.get_lock():
            self._total_out.value += 1  # type: ignore
        return item

    def size(self):
        data = []
        with self._size.get_lock():
            data.append(self._size.value)  # type: ignore
        with self._total_in.get_lock():
            data.append(self._total_in.value)  # type: ignore
        with self._total_out.get_lock():
            data.append(self._total_out.value)  # type: ignore
        return data


class Manager (mpman.SyncManager):

    Queue: type[ManagedQueue]
    Dot: type[Dot]

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        if _pinfo.manager is not None:
            raise Exception('Manager already running!')
        else:
            _pinfo.manager = self


Manager.register('Queue', ManagedQueue, exposed=['get', 'put', 'size'])
Manager.register('Dot', Dot, exposed=['__setitem__', '__getitem__', '__call__', '__rich__', '_table'])


@group()
def _render_children(children: dict):
    table = Table(box=box.ROUNDED, style='black')
    for child in children.values():
        if hasattr(child, '_render'):
            table.add_row(child._render())
    yield table


class Thread (threading.Thread):

    children: dict

    _is_main: bool = False

    def __init__(self, target: Optional[Callable] = None, main: bool = False, *args, **kwargs):
        super().__init__(target=target, *args, **kwargs)

        self._is_main = main
        self.name = generate_name()
        self.target = target
        self.alive = Alive(state=self._is_main)

        self.children = {}

        # If thread is running, append self to that thread's children
        if _tinfo.thread is None:
            _tinfo.thread = self
        else:
            _tinfo.thread.children[self.name] = self

    def start(self) -> None:
        self.alive.alive()
        return super().start()

    def run(self) -> None:
        """
        Runs target method in new thread.
        """
        _tinfo.thread = self
        result = super().run()
        return result

    @group()
    def _render(self):

        title = [f'[bold magenta]Thread[reset]', f'[magenta]{self.name}[reset]']
        if self.isDaemon():
            title += [f'[red][daemon][reset]']
        if self.target is not None:
            title += [f'[magenta]{self.target.__name__}[reset]()']

        render = Group('', Columns([*title, self.alive], padding=(0, 3)))
        if self.children:
            render = Group(render, _render_children(self.children))
        if self._is_main:
            render = Panel(render, style='black')
        yield render

    def __rich__(self):
        return self._render()


class Queue:

    size: int = 0
    total_in: int = 0
    total_out: int = 0

    queues: dict[str, ManagedQueue]

    def __init__(self, group: str = 'default', maxsize: int = 100) -> None:
        self.name = group or generate_name()
        self.maxsize = maxsize
        self.group = group

        if _pinfo.manager is not None:
            self.manager = _pinfo.manager
        else:
            raise Exception('No Manager Running!')

        if group in _tinfo.queues:
            self.queues = _tinfo.queues[group]
        else:
            self.queues = {'in': self.manager.Queue(maxsize),
                           'out': self.manager.Queue(maxsize)}
            _tinfo.queues.update({group: self.queues})

        format = '[progress.percentage]{task.completed} / {task.total}'
        columns = (progress.TextColumn(format), progress.BarColumn(bar_width=15))
        self.progress = Dot({'in': Progress(['size'], total=maxsize, columns=columns),
                             'out': Progress(['size'], total=maxsize, columns=columns)})

    @group()
    def _render(self):

        table = Table(expand=False)

        def bar(key: str):
            _, _, total_out = self.queues[key].size()
            self.progress[key].update('size', completed=self.size)
            table.add_row(f'[yellow]{key}', self.progress[key], f'[yellow] -> {total_out}')

        bar('in')
        bar('out')

        yield table

    def __rich__(self):
        return self._render()


class Process:

    children: dict

    queues: dict[str, ManagedQueue]

    def __init__(self,
                 args,
                 target: Callable | None = None,
                 name: str | None = None,
                 queue: Queue | None = None,
                 hidden: bool = False, *arguments, **kwargs):

        self.target = target
        self.name = name or generate_name()
        self.alive = Alive(state=False)

        self.children = {}

        if queue is not None:
            self.queue = queue
            self.queues = queue.queues
        else:
            self.queue = Queue(group=self.name)
            self.queues = self.queue.queues

        # If thread is running, append self to that thread's children
        if _tinfo.thread is not None and not hidden:
            _tinfo.thread.children[self.name] = self

        queues = {**self.queues}
        self.process = mp.Process(target=target, args=[queues, *args], *arguments, **kwargs)

    def join(self):
        self.process.join()

    def start(self):

        # Start Process
        self.process.start()

        _tinfo.n_procs += 1
        self.alive.alive()

    def close(self):

        # Close Process
        self.queues['in'].put((None, None))
        self.queues['out'].put((None, None))
        self.process.join()

        _tinfo.n_procs -= 1
        self.alive.stopped()

    @group()
    def _render(self):

        title = [f'[bold blue]Process[reset]', f'[blue]{self.name}[reset]']
        if self.target is not None:
            title += [f'[blue]{self.target.__name__}[reset]()']

        render = Group('', Columns([*title, self.alive], padding=(0, 3)))
        if self.children:
            render = Group(render, _render_children(self.children))
        yield render
        yield self.queue

    def __rich__(self):
        return self._render()


class Pool:

    children: dict

    def __init__(self, size: int, name: str | None = None) -> None:
        self.size = size
        self.name = name or generate_name()
        self.queue = Queue(group=self.name)

        self.children = {}

    def apply_async(self, target: Callable, parameters: Iterable, *args, **kwargs):

        kwargs.update(target=target, args=parameters, queue=self.queue, hidden=True)
        self.processes = [Process(*args, **kwargs) for _ in range(self.size)]

        # Add processes to renderable
        for process in self.processes:
            self.children[process.name] = process

        # Add processes to renderable
        if _tinfo.thread is not None:
            _tinfo.thread.children[self.name] = self

        for process in self.processes:
            process.start()

    def close(self):

        # Close processes
        for process in self.processes:
            self.queue.queues['in'].put((None, None))
            self.queue.queues['out'].put((None, None))
        for process in self.processes:
            process.join()
            _tinfo.n_procs -= 1
            process.alive.stopped()
            del self.children[process.name]

    @group()
    def _render(self):

        title = [f'[green]{self.name}']
        cols = Columns(title)
        yield cols
        yield self.queue

        if self.children:
            yield _render_children(self.children)

    def __rich__(self):
        return self._render()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def get_current_manager():
    if _pinfo.manager is not None:
        return _pinfo.manager
    else:
        manager = Manager()
        manager.start()
        return manager
