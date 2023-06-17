from __future__ import annotations
import contextlib as cl
import ctypes
from dataclasses import dataclass
import json
import linecache
import os
import socket
import sys
import termios
import threading
import time
import tracemalloc
from typing import Any, Callable, Iterable, Optional, Union

import GPUtil
import psutil
from pyfzf.pyfzf import FzfPrompt
from rich import box
from rich.console import Console, Group, group
from rich.panel import Panel
from rich.progress import BarColumn, Progress, TaskProgressColumn, TextColumn
from rich.table import Column
from rich.text import Text
import torch
import torch.nn as nn
from torchviz import make_dot

from .database import influxdb
from .cli import console
from .func import ema
from .module import Module
from .renderables import Table


def viz(module: Module, tensor: torch.Tensor, path: str = 'viz'):
    dot = make_dot(tensor, params=dict(module.named_parameters()),
                   show_attrs=True, show_saved=True)
    dot.render(path)


def sin_data(T: int, size: int, batch_size: int = 1):
    t = torch.linspace(0, 1, T)
    a, b = torch.rand(2, batch_size, size)
    data = (a[:, None, :] * t[None, :, None] + b[:, None, :]).sin()
    return data


def makedir(path: str):
    if not os.path.exists(path): os.makedirs(path)


@cl.contextmanager
def quiet():
    with open(os.devnull, 'w') as null:
        with cl.redirect_stdout(null), cl.redirect_stderr(null):
            yield None


class RedirectStream (object):
    # source: https://github.com/bulletphysics/bullet3/discussions/3441

    @staticmethod
    def _flush_c_stream(stream):
        streamname = stream.name[1:-1]
        libc = ctypes.CDLL(None)
        libc.fflush(ctypes.c_void_p.in_dll(libc, streamname))

    def __init__(self, stream=sys.stderr, file=os.devnull):
        self.stream = stream
        self.file = file

    def __enter__(self):
        self.stream.flush()  # ensures python stream unaffected
        self.fd = open(self.file, "w+")
        self.dup_stream = os.dup(self.stream.fileno())
        os.dup2(self.fd.fileno(), self.stream.fileno())  # replaces stream

    def __exit__(self, *_):
        # quiet._flush_c_stream(self.stream)  # ensures C stream buffer empty
        os.dup2(self.dup_stream, self.stream.fileno())  # restores stream
        os.close(self.dup_stream)
        self.fd.close()


def ranges(x: torch.Tensor):
    return (round(x.min().item(), 3),
            round(x.mean().item(), 3) if x.is_floating_point() else None,
            round(x.max().item(), 3))


class Ranges:

    _min: float
    _max: float
    _mean: Optional[float]
    _std: Optional[float]

    def __init__(self, x: Optional[torch.Tensor] = None):
        if x is not None:
            self.update(x)

    def update(self, x: torch.Tensor):
        self._min = x.min().item()
        self._max = x.max().item()
        self._mean = x.mean().item() if x.is_floating_point() else None
        self._std = x.std().item() if x.is_floating_point() else None

    def log(self, key: str):
        influxdb.log(f'{key}.min', self._min)
        influxdb.log(f'{key}.max', self._max)
        if self._mean is not None:
            influxdb.log(f'{key}.mean', self._mean)
        if self._std is not None:
            influxdb.log(f'{key}.std', self._std)

    def __rich__(self):
        msg = Text.from_markup(f'[[green]{self._min:.2f}[/green], [green]{self._max:.2f}[/green]]')
        if self._mean is not None and self._std is not None:
            msg += ' ~ '
            msg += Text.from_markup(f'([blue]{self._mean:.2f}[/blue], [blue]{self._std:.2f}[/blue])')
        return msg


class FreezeParameters:

    def __init__(self, *modules: nn.Module):
        self.modules = modules
        self.parameters, self.states = [], []
        for module in self.modules:
            self.parameters += list(module.parameters())
            self.states += [param.requires_grad for param in module.parameters()]

    def __enter__(self):
        for param in self.parameters:
            param.requires_grad = False

    def __exit__(self, *_):
        for param, state in zip(self.parameters, self.states):
            param.requires_grad = state


class Metadata:

    def __init__(self, path: str, name: str = 'metadata', default: Optional[dict] = None) -> None:
        self.file = f'{name}.json'
        self.path = os.path.join(path, self.file)
        self.data = default or {}

    @classmethod
    def load(cls, dir: str, name: str = 'metadata'):
        file = f'{name}.json'
        path = os.path.join(dir, file)
        if os.path.exists(path):
            with open(path, 'r') as f:
                data = json.load(f)
            return data
        else:
            return {}

    @classmethod
    def save(cls, dir: str, name: str = 'metadata', data: dict = {}):
        file = f'{name}.json'
        path = os.path.join(dir, file)
        if os.path.exists(path):
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=4, default=str)

    def __enter__(self):
        if os.path.exists(self.path):
            with open(self.path, 'r') as f:
                self.data = json.load(f) or self.data
        else:
            dirname = os.path.dirname(self.path)
            if not os.path.exists(dirname):
                os.makedirs(dirname)
        return self

    def __exit__(self, *_):
        with open(self.path, 'w', encoding='utf-8') as f:
            json.dump(self.data, f, ensure_ascii=False, indent=4, default=str)


class Timer:

    def __init__(self, name: Optional[str] = None):
        self._start = self._mark = time.time()
        self._rate = 0.0
        self._name = name

    def reset(self):
        self._start = self._mark = time.time()
        self._rate = 0.0

    def __call__(self, reset: bool = False, step: bool = False):
        now = time.time()
        total = now - self._start
        diff = now - self._mark
        if reset:
            self._start = now
        if step:
            self._mark = now

            # EMA update
            self._rate = ema(self._rate, 1 / (diff / 60**2))

        return total, diff

    def __enter__(self):
        self.reset()

    def __exit__(self, *_):
        console.print(self)

    @group()
    def _render_rate(self):
        if self._rate != 0.0:
            yield f'[cyan]{self._rate:,.2f}[reset] steps/hour'

    def __rich__(self):
        total, _ = self()
        mins = total / 60
        if mins < 1:
            msg = f'[blue]{total:,.5f}[reset] secs'
        elif mins < 60:
            msg = f'[blue]{mins:,.2f}[reset] mins'
        else:
            hrs = mins // 60
            mins = mins - 60 * hrs
            msg = f'[blue]{hrs:,.2f}[reset] hrs [blue]{mins:,.2f}[reset] mins'
        if self._rate != 0.0:
            msg += f' ([cyan]{self._rate:,.2f}[reset] steps/hour)'
        if self._name is not None:
            msg += f'{self._name:>15}'
        return msg


def thread(function: Callable):
    def run(*args, **kwargs):
        t = Thread(target=function, args=args, kwargs=kwargs)
        t.start()
        return t
    return run


def display_top(snapshot, key_type='lineno', limit=5):
    if isinstance(snapshot, tracemalloc.Snapshot):
        snapshot = snapshot.filter_traces((
            tracemalloc.Filter(False, "<frozen importlib._bootstrap>"),
            tracemalloc.Filter(False, "<unknown>"),
        ))
        stats = snapshot.statistics(key_type)
    elif isinstance(snapshot, list):
        stats = [stat for stat in snapshot]
    else:
        raise Exception()

    table = Table('Rank', 'File', 'Size', 'Line', title=f'Top {limit} lines', box=box.ROUNDED)
    for index, stat in enumerate(stats[:limit], 1):
        frame = stat.traceback[0]
        # replace "/path/to/module/file.py" with "module/file.py"
        filename = os.sep.join(frame.filename.split(os.sep)[-2:])
        row = [f'[cyan]{index}',
               f'[magenta]{filename} : [reset]{frame.lineno}',
               f'[yellow]{stat.size / 1024:5.3f}']
        line = linecache.getline(frame.filename, frame.lineno).strip()
        if line: row += [f'[white]{line}']
        table.add_row(*row)

    console.log(table)
    other = stats[limit:]
    if other:
        size = sum(stat.size for stat in other)
        console.log(f'{len(other)} other: [red]{size / 1024:.1f}[reset] KiB')
    total = sum(stat.size for stat in stats)
    console.log(f'Total allocated size: [red]{total / 1024:.1f}[reset] KiB')


class TextBuffer:

    buffer: list[str]

    def __init__(self, prompt: str = ''):
        self.prompt = prompt
        self.buffer = []

    def delete(self):
        self.buffer.pop(-1)

    def append(self, msg: str):
        self.buffer.append(msg)

    def copy(self, buffer: TextBuffer):
        self.buffer = [char for char in buffer.buffer]

    def msg(self):
        return ''.join(self.buffer)

    def __rich__(self):
        return Text(self.prompt) + Text(self.msg())


class Keyboard:

    callbacks: dict

    def __init__(self) -> None:
        self.callbacks = {}

    def __enter__(self):
        self.thread = threading.Thread(target=self.poll, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *_):
        pass

    def readchar(self):
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        new_settings = termios.tcgetattr(fd)
        new_settings[3] = new_settings[3] & ~termios.ECHO & ~termios.ICANON
        termios.tcsetattr(fd, termios.TCSANOW, new_settings)
        try:
            char = sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSAFLUSH, old_settings)
        return char

    def poll(self):
        try:
            while True:
                char = self.readchar()
                callback = self.callbacks.get(char, None)
                if callback is not None:
                    callback()
                # else:
                #     console.log(char)
                #     console.log(ord(char))
        except:
            console.print_exception()


class Pager:

    def __init__(self, renderable: Any) -> None:
        self.renderable = renderable
        from .cli.console import THEME
        self.console = Console(theme=THEME)

    def start(self):
        with self.console.pager():
            # for line in self.renderable:
            #     self.console.print(line)
            self.console.print(self.renderable)

    def stop(self):
        pass


class Fuzzy:

    def __init__(self, console, iterable: Any) -> None:
        self.console = console
        self.iterable = iterable
        self.prompt = FzfPrompt()

    def start(self):
        with self.console.screen():
            self.prompt.prompt(self.iterable)

    def stop(self):
        pass


class Screens:

    screens: dict
    active: Optional[str] = None

    def __init__(self, screens: dict) -> None:
        self.screens = screens

    def select(self, key: str):
        if key in self.screens:
            if self.active is None:
                self.active = key
                if hasattr(self.screens[key], 'start'):
                    self.screens[key].start()
                elif hasattr(self.screens, '__enter__'):
                    self.screens[key].__enter__()
                else:
                    raise Exception('Screen has no start or __enter__ function')
            elif key != self.active:
                if hasattr(self.screens[key], 'stop'):
                    self.screens[key].stop()
                elif hasattr(self.screens, '__exit__'):
                    self.screens[key].__exit__()
                else:
                    raise Exception('Screen has no stop or __exit__ function')
                self.active = key
                self.screens[key].start()

    def clear(self):
        if self.active is not None and self.active in self.screens:
            self.screens[self.active].stop()
        self.active = None


class Terminal:

    inputs: TextBuffer
    message: TextBuffer

    def __init__(self) -> None:
        self.inputs = TextBuffer('-> ')
        self.message = TextBuffer()

    def __enter__(self):
        self.thread = threading.Thread(target=self.poll, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *_):
        pass

    def readchar(self):
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        new_settings = termios.tcgetattr(fd)
        new_settings[3] = new_settings[3] & ~termios.ECHO & ~termios.ICANON
        termios.tcsetattr(fd, termios.TCSANOW, new_settings)
        try:
            char = sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSAFLUSH, old_settings)
        return char

    def poll(self):
        try:
            while True:
                char = self.readchar()
                if char == Key.enter or char == '\n':
                    self.command()
                elif char == Key.backspace or char == '\x7f':
                    self.backspace()
                else:
                    self.inputs.append(char)
        except:
            console.print_exception()

    def command(self):
        self.message.copy(self.inputs)
        self.inputs = TextBuffer('-> ')

        # command = self.message.msg()
        # if command == 'exit':
        #     self.trainer.exit()

    def backspace(self):
        self.inputs.delete()

    @group()
    def __rich__(self):
        yield self.message
        yield self.inputs


def cluster():
    hostname = socket.gethostname()
    return 'cs' in hostname


class Steps:

    counters: dict
    moduli: dict

    def __init__(self, keys: Optional[Iterable[str]] = None) -> None:
        self.counters = {}
        self.moduli = {}
        if keys is not None:
            for key in keys:
                self.add(key)

    def add(self, key: str):
        if not key in self.counters:
            self.counters[key] = {'timer': Timer(), 'steps': 0}

    def add_modulo(self, key: str, every: int = 1, gt: int = -1):
        if not key in self.moduli:
            self.moduli[key] = {'every': every, 'gt': gt, 'steps': 0}

    def get(self, key: str):
        if key in self.counters:
            return self.counters[key]['steps']
        else:
            return 0

    def step(self, key: str):
        if key in self.counters:
            self.counters[key]['steps'] += 1
            self.counters[key]['timer'](step=True)

    def modulo(self, key: str, dest: str):
        if key in self.counters and dest in self.moduli:

            # Get relevant information
            every, gt = self.moduli[dest]['every'], self.moduli[dest]['gt']
            step = self.counters[key]['steps']

            # Check if modulus condition is met
            if step % every == 0 and step > gt:
                self.moduli[dest]['steps'] += 1
                return True

        return False

    def __rich__(self):

        table = Table(Column('Name', ratio=1),
                      Column('Steps', ratio=1),
                      Column('Info', ratio=3),
                      show_header=True, box=box.ROUNDED,
                      style='black', header_style='bold yellow')

        for key, counter in self.counters.items():
            name = Text(key, style='magenta')
            steps = f'[magenta]{counter["steps"]:,}'
            info = counter['timer']._render_rate()
            table.add_row(name, steps, info)

        for key, modulus in self.moduli.items():
            name = Text(key, style='blue')
            steps = f'[blue]{modulus["steps"]:,}'
            info = f'[white]Every [reset][blue]{modulus["every"]:,}[reset][white] steps'
            table.add_row(name, steps, info)

        return table


class System:

    stats: dict[str, dict[str, float]]
    progress: dict[str, Progress]

    def __init__(self) -> None:
        self.stats = {'CPU': {}, 'Memory': {}}
        self.progress = {}
        self._update()
        self._update_progress()

    def _update(self):

        gpus = GPUtil.getGPUs()
        if len(gpus) > 0:
            gpu = gpus[0]
            self.stats['GPU'] = {}
            self.stats['GPU']['Memory'] = gpu.memoryUsed / gpu.memoryTotal
            self.stats['GPU']['Load'] = gpu.load
            memory_used = torch.cuda.memory_allocated(0)
            memory_total = torch.cuda.get_device_properties(0).total_memory
            self.stats['GPU']['Torch'] = memory_used / memory_total

        self.stats['CPU']['Usage'] = psutil.cpu_percent() / 100

        self.stats['Memory']['Usage'] = psutil.virtual_memory().percent / 100

    def _update_progress(self):

        for device, stats in self.stats.items():

            # Initialize
            if not device in self.progress:
                columns = [TextColumn('{task.description:10}', style='yellow'),
                           BarColumn(complete_style='yellow', finished_style='yellow'),
                           TaskProgressColumn()]
                self.progress[device] = Progress(*columns)

            ids = {task.description: task.id for task in self.progress[device].tasks}
            for key, stat in stats.items():

                completed = int(stat * 100)

                # Initialize
                if not key in ids:
                    self.progress[device].add_task(key, total=100, completed=completed)
                else:
                    self.progress[device].update(ids[key], completed=completed)

    @group()
    def __rich__(self):

        self._update()
        self._update_progress()

        panels = []
        for device, progress in self.progress.items():
            panels += [Panel(progress, title=device, title_align='left', border_style='black')]

        group = Group(*panels)
        yield group


class Loss:

    losses: dict

    def __init__(self) -> None:
        self.losses = {}

    def add(self, name: str, loss: torch.Tensor):
        self.losses[name] = loss
