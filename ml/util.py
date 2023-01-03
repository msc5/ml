import contextlib as cl
import json
import linecache
import os
import threading
import time
import tracemalloc
from typing import Callable, Optional

from rich import box
from rich.table import Table
import torch
import torch.nn as nn
from torchviz import make_dot

from src.ml.cli import console

from .io import generate_name
from .module import Module


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


def pos_embed(x: torch.Tensor, max_len: int = 1000, seq_dim: int = -2, size_dim: int = -1):
    """ 
    Inputs: 
        x: [ *, seq, size ]
    """

    seq_len = x.shape[seq_dim]
    size = x.shape[size_dim]

    # x : [ batch, seq_len (t), size (i) ]
    t = torch.arange(0, seq_len, device=x.device)
    i = torch.arange(0, size, device=x.device)
    k = i.div(2, rounding_mode='floor')

    evens = (i % 2) == 0
    odds = ~evens
    w = 1 / (max_len**(2 * k / size))

    p = torch.zeros(seq_len, size, device=x.device)
    p[:, evens] = (w[None, evens] * t[:, None]).sin()
    p[:, odds] = (w[None, odds] * t[:, None]).cos()

    shape = [1] * len(x.shape)
    shape[seq_dim], shape[size_dim] = seq_len, size
    p = p.reshape(*shape)
    p = p.expand(*x.shape)

    return p


@cl.contextmanager
def quiet():
    with open(os.devnull, 'w') as null:
        with cl.redirect_stdout(null), cl.redirect_stderr(null):
            yield None


def ranges(x: torch.Tensor):
    return (round(x.min().item(), 3),
            round(x.mean().item(), 3) if x.is_floating_point() else None,
            round(x.max().item(), 3))


class Ranges:

    _min: float
    _max: float
    _mean: Optional[float]

    def __init__(self, x: torch.Tensor):
        self._min = x.min().item()
        self._max = x.max().item()
        self._mean = x.mean().item() if x.is_floating_point() else None

    def __rich__(self):
        return f'([green]{self._min:.2f}[reset], [yellow]{self._mean:.2f}[reset], [red]{self._max:.2f}[reset])'


# def get_parameters(modules: Iterable[nn.Module]):
#     """
#     Given a list of torch modules, returns a list of their parameters.
#     :param modules: iterable of modules
#     :returns: a list of parameters
#     """
#     model_parameters = []
#     for module in modules:
#         model_parameters += list(module.parameters())
#     return model_parameters


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

    def __init__(self, path: str, default: Optional[dict] = None) -> None:
        self.path = os.path.join(path, 'metadata.json')
        self.data = default or {}

    @classmethod
    def load(cls, path: str):
        if os.path.exists(path):
            with open(path, 'r') as f:
                data = json.load(f)
            return data
        else:
            return {}

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

    def __init__(self):
        self._start = self._mark = time.perf_counter()
        self._rate = 0.0

    def __call__(self, reset: bool = False, step: bool = False):
        now = time.perf_counter()
        total = now - self._start
        diff = now - self._mark
        if reset:
            self._start = now
        if step:
            self._mark = now
            self._rate = 1 / (diff / 60**2)
        return total, diff

    def __rich__(self):
        total, _ = self()
        mins = total / 60
        if mins < 60:
            msg = f'[blue]{mins:.2f}[reset] mins'
        else:
            hrs = mins // 60
            mins = mins - 60 * hrs
            msg = f'[blue]{hrs:.2f}[reset] hrs [blue]{mins:.2f}[reset] mins'
        if self._rate != 0.0:
            msg += f' ([cyan]{self._rate:.2f}[reset] steps/hour)'
        return msg


class Thread (threading.Thread):

    daemon: bool = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        from .trainer import CurrentTrainer
        if CurrentTrainer is not None:
            self.threads = CurrentTrainer.threads
            self.name = generate_name()
            self.threads[self.name] = self

    def run(self) -> None:
        result = super().run()
        del self.threads[self.name]
        return result

    def __rich__(self):
        if self.is_alive():
            return '[green]⬤  alive'
        else:
            return '[red] stopped'


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
