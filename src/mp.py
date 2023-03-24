from __future__ import annotations
import multiprocessing.managers as mpman
import threading
from typing import Callable, Iterable, Optional

from rich import box
from rich.columns import Columns
from rich.console import RenderableType, group
import rich.progress as progress
import torch.multiprocessing as mp
import torch.multiprocessing.queue as tmpq
import wandb

from .cli import console
from .io import generate_name
from .options.dot import Dot
from .renderables import Alive, Progress, Table, section

COLUMN_WIDTH = 20


class ProcessInfo:

    process: Process | None = None
    manager: Manager | None = None


class ThreadInfo (threading.local):

    thread: Thread | None = None
    queues: dict[str, dict[str, ManagedQueue]] = {}
    n_procs: int = 0

    update_thread: Thread | None = None
    update_queue: ManagedQueue | None = None
    update_thread_start: threading.Event = threading.Event()


_pinfo: ProcessInfo = ProcessInfo()
_tinfo: ThreadInfo = ThreadInfo()


class ManagedQueue (tmpq.Queue):

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


class Thread (threading.Thread):

    children: dict

    _is_main: bool = False
    _renderable: RenderableType

    def __init__(self, target: Optional[Callable] = None, main: bool = False, *args, **kwargs):
        super().__init__(target=target, *args, **kwargs)

        self._is_main = main
        self.name = generate_name()
        self.target = target

        self.children = {}
        self._renderable = self._render()

        if _tinfo.thread is None:
            _tinfo.thread = self
        else:
            _tinfo.thread.children[self.name] = self.children
            _tinfo.thread._renderable = _tinfo.thread._render()

    def run(self) -> None:
        """
        Runs target method in new thread.
        """
        _tinfo.thread = self
        result = super().run()
        return result

    @group()
    def _render(self):

        alive = Alive(state=self._is_main, callback=self.is_alive)
        title = [f'[magenta]{self.name}']
        if self.target is not None:
            daemon = f'[red]d: [reset]' if self.isDaemon() else ''
            title += [f'[magenta]({daemon}[reset]{self.target.__name__}[magenta])']

        cols = Columns([*title, alive], width=COLUMN_WIDTH)

        table = Table(box=box.ROUNDED, style='black')
        for name, child in self.children.items():
            if hasattr(child, '_render'):
                table.add_row(name, child._render())

        yield cols
        yield table

    def __rich__(self):
        return self._renderable


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
        self.progress = Dot({'in': Progress(['size'], total=maxsize, columns=columns, expand=True),
                             'out': Progress(['size'], total=maxsize, columns=columns, expand=True)})

    @group()
    def _render(self):

        def bar(key: str):
            _, _, total_out = self.queues[key].size()
            self.progress[key].update('size', completed=self.size)
            return Columns([f'[yellow]{key:<18}', self.progress[key], f'[yellow] -> {total_out}'])

        yield bar('in')
        yield bar('out')

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
        # self.children = Dot()
        # self.children._order = 1

        if queue is not None:
            self.queue = queue
            self.queues = queue.queues
        else:
            self.queue = Queue(group=self.name)
            self.queues = self.queue.queues

        self.children['queue'] = {}
        # self.children.queue = Dot()
        # self.children.queue._set_renderable(self.queue)
        # self.children.queue._order = -1

        if _tinfo.thread is not None and not hidden:
            _tinfo.thread.children[self.name] = self.children

        # # Initialize Update Loop Thread
        # if _threaded.update_thread is None and _processed.manager is not None:
        #     _threaded.update_queue = _processed.manager.Queue()
        #     _threaded.update_thread = Thread(target=update_loop, args=[_threaded.update_queue], daemon=True)

        # queues = {**self.queues, 'io': _threaded.update_queue}
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

        title = [f'[blue]{self.name}']
        if self.target is not None:
            title += [f'[blue]([reset]{self.target.__name__}[blue])[reset]']
        cols = Columns([*title, self.alive], width=COLUMN_WIDTH)

        yield cols

    def __rich__(self):
        return self._render()


class Pool:

    children: dict

    def __init__(self, size: int, name: str | None = None) -> None:
        self.size = size
        self.name = name or generate_name()
        self.queue = Queue(group=self.name)

        self.children = {}

        # self.children.queue = Dot()
        # self.children.queue._set_renderable(self.queue)
        # self.children.queue._order = -1

    def apply_async(self, target: Callable, parameters: Iterable, *args, **kwargs):

        kwargs.update(target=target, args=parameters, queue=self.queue, hidden=True)
        self.processes = [Process(*args, **kwargs) for _ in range(self.size)]

        for process in self.processes:
            self.children[process.name] = process.children

        # self.children._set_renderable(self.renderable())
        # self.children._order = 2

        if _tinfo.thread is not None:
            _tinfo.thread.children[self.name] = self.children

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
        # title.append(f'[black]process')
        cols = Columns(title, width=COLUMN_WIDTH)
        yield cols

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
        # raise Exception('No Manager Running!')
