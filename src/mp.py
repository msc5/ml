from __future__ import annotations
import multiprocessing.managers as mpman
import threading
from typing import Callable, Iterable, Optional

from rich.columns import Columns
from rich.console import group
import rich.progress as progress
from rich.text import Text
import torch.multiprocessing as mp
import torch.multiprocessing.queue as tmpq
import wandb

from .cli import console
from .options.dot import Dot
from .renderables import Alive, Progress, section
from .io import generate_name

COLUMN_WIDTH = 20


class Processed:
    process: Process | None = None
    manager: Manager | None = None


class Threaded (threading.local):
    thread: Thread | None = None
    queues: dict[str, dict[str, ManagedQueue]] = {}
    n_procs: int = 0

    update_thread: Thread | None = None
    update_queue: ManagedQueue | None = None
    update_thread_start: threading.Event = threading.Event()


_processed: Processed = Processed()
_threaded: Threaded = Threaded()


def update_loop(queue: ManagedQueue):
    while True:
        cmd, msg = queue.get()
        try:
            if cmd == None and msg == None:
                return
            elif cmd == 'log video':
                video = wandb.Video(msg['video'], caption=msg['name'], fps=msg['fps'], format=msg['format'])
                wandb.log({msg['name']: video}, step=msg['step'])
            elif cmd == 'exception':
                section('Exception', module=msg.get('module'), color='red')
                console.log(msg.get('traceback'))
        except:
            section('Exception', module='Update Loop', color='red')
            console.print_exception()


class Thread (threading.Thread):

    children: Dot
    _is_main: bool = False

    def __init__(self, target: Optional[Callable] = None, main: bool = False, *args, **kwargs):
        super().__init__(target=target, *args, **kwargs)

        self._is_main = main
        self.name = generate_name()
        self.alive = Alive(state=self._is_main, callback=self.is_alive)
        self.target = target

        self.children = Dot()
        self.children._set_renderable(self.renderable())

        if _threaded.thread is None:
            _threaded.thread = self
        else:
            _threaded.thread.children[self.name] = self.children

    def run(self) -> None:
        _threaded.thread = self
        self.alive.alive()
        result = super().run()
        self.alive.stopped()
        # if _threaded.update_thread is not None:
        #     _threaded.update_thread.join()
        return result

    def renderable(self):
        title = [f'[magenta]{self.name}']
        if self.target is not None:
            daemon = f'[red]d: [reset]' if self.isDaemon() else ''
            title += [f'[magenta]({daemon}[reset]{self.target.__name__}[magenta])']
        return Columns([*title, self.alive], width=COLUMN_WIDTH)

    def __rich__(self):
        return self.children


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

        if _processed.manager is not None:
            raise Exception('Manager already running!')
        else:
            _processed.manager = self


Manager.register('Queue', ManagedQueue, exposed=['get', 'put', 'size'])
Manager.register('Dot', Dot, exposed=['__setitem__', '__getitem__', '__call__', '__rich__', '_table'])


class Queue:

    size: int = 0
    total_in: int = 0
    total_out: int = 0

    queues: dict[str, ManagedQueue]

    def __init__(self, group: str = 'default', maxsize: int = 100) -> None:
        self.name = group or generate_name()
        self.maxsize = maxsize
        self.group = group

        if _processed.manager is not None:
            self.manager = _processed.manager
        else:
            raise Exception('No Manager Running!')

        if group in _threaded.queues:
            self.queues = _threaded.queues[group]
        else:
            self.queues = {'in': self.manager.Queue(maxsize),
                           'out': self.manager.Queue(maxsize)}
            _threaded.queues.update({group: self.queues})

        format = '[progress.percentage]{task.completed} / {task.total}'
        columns = (progress.TextColumn(format), progress.BarColumn(bar_width=15))
        self.progress = Dot({'in': Progress(['size'], total=maxsize, columns=columns, expand=True),
                             'out': Progress(['size'], total=maxsize, columns=columns, expand=True)})

    @group()
    def __rich__(self):

        def bar(key: str):
            _, _, total_out = self.queues[key].size()
            self.progress[key].update('size', completed=self.size)
            return Columns([f'[yellow]{key:<18}', self.progress[key], f'[yellow] -> {total_out}'])

        yield bar('in')
        yield bar('out')


class Process:

    children: Dot
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

        self.children = Dot()
        self.children._set_renderable(self.renderable())
        self.children._order = 1

        if queue is not None:
            self.queue = queue
            self.queues = queue.queues
        else:
            self.queue = Queue(group=self.name)
            self.queues = self.queue.queues

        self.children.queue = Dot()
        self.children.queue._set_renderable(self.queue)
        self.children.queue._order = -1

        if _threaded.thread is not None and not hidden:
            _threaded.thread.children[self.name] = self.children

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

        # if (
        #         _threaded.update_thread is not None
        #         and not _threaded.update_thread.is_alive()
        #         and not _threaded.update_thread_start.is_set()
        # ):
        #     _threaded.update_thread_start.set()
        #     _threaded.update_thread.start()

        # Start Process
        self.process.start()

        _threaded.n_procs += 1
        self.alive.alive()

    def close(self):

        # Close Process
        self.queues['in'].put((None, None))
        self.queues['out'].put((None, None))
        self.process.join()

        _threaded.n_procs -= 1
        # # Close Update Loop Thread
        # if (
        #     _threaded.n_procs == 0
        #     and _threaded.update_thread is not None
        #     and _threaded.update_thread_start.is_set()
        #     and _threaded.update_queue is not None
        # ):
        #     _threaded.update_thread_start.clear()
        #     _threaded.update_queue.put((None, None))
        #     _threaded.update_thread.join()
        #     _threaded.update_thread = Thread(target=update_loop, args=[_threaded.update_queue], daemon=True)

        self.alive.stopped()

    def renderable(self):
        title = [f'[blue]{self.name}']
        if self.target is not None:
            title += [f'[blue]([reset]{self.target.__name__}[blue])[reset]']
        return Columns([*title, self.alive], width=COLUMN_WIDTH)

    def __rich__(self):
        return self.children


class Pool:

    def __init__(self, size: int, name: str | None = None) -> None:
        self.size = size
        self.name = name or generate_name()
        self.queue = Queue(group=self.name)
        self.children = Dot()

        self.children.queue = Dot()
        self.children.queue._set_renderable(self.queue)
        self.children.queue._order = -1

    def apply_async(self, target: Callable, parameters: Iterable, *args, **kwargs):

        kwargs.update(target=target, args=parameters, queue=self.queue, hidden=True)
        self.processes = [Process(*args, **kwargs) for _ in range(self.size)]

        for process in self.processes:
            self.children[process.name] = process.children
        self.children._set_renderable(self.renderable())
        self.children._order = 2

        if _threaded.thread is not None:
            _threaded.thread.children[self.name] = self.children

        for process in self.processes:
            process.start()

    def close(self):

        # Close processes
        for process in self.processes:
            self.queue.queues['in'].put((None, None))
            self.queue.queues['out'].put((None, None))
        for process in self.processes:
            process.join()
            _threaded.n_procs -= 1
            process.alive.stopped()
            object.__delattr__(self.children, process.name)

        # # Close Update Loop Thread
        # if (
        #         _threaded.update_thread is not None
        #         and _threaded.update_thread_start.is_set()
        #         and _threaded.update_queue is not None
        # ):
        #     _threaded.update_thread_start.clear()
        #     _threaded.update_queue.put((None, None))
        #     _threaded.update_thread.join()
        #     _threaded.update_thread = Thread(target=update_loop, args=[_threaded.update_queue], daemon=True)

    def renderable(self):
        title = [f'[green]{self.name}']
        # title.append(f'[black]process')
        return Columns(title, width=COLUMN_WIDTH)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def get_current_manager():
    if _processed.manager is not None:
        return _processed.manager
    else:
        manager = Manager()
        manager.start()
        return manager
        # raise Exception('No Manager Running!')
