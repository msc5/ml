import torch

from collections.abc import Callable
from typing import Any, Optional, Union, Iterable

from rich import box
from rich.console import Console, ConsoleOptions, RenderResult, RenderableType, group
from rich.measure import Measurement
from rich.padding import Padding
import rich.progress as progress
from rich.spinner import Spinner
from rich.styled import Styled
import rich.table
from rich.text import Text

from .cli import console


class Characters:
    closed_circle: str = ''
    open_circle: str = ''
    arrow_down_right: str = '↘'


class Progress (progress.Progress):

    def __init__(self,
                 tasks: list[Union[str, int]] = [],
                 total: int = 100,
                 transient: bool = False,
                 columns: Optional[Any] = None,
                 *args, **kwargs) -> None:
        cols = columns or (progress.TextColumn('{task.description}'),
                           progress.BarColumn(),
                           progress.TaskProgressColumn())
        super().__init__(*cols, *args, **kwargs)
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


class Memory:

    log: list[tuple]

    def __init__(self):
        self.log = []
        self.check()

    def check(self):
        used_mem = torch.cuda.memory_allocated()
        res_mem = torch.cuda.memory_reserved()
        self.log.append((used_mem, res_mem))

    @group()
    def __rich__(self):
        self.check()
        orig_used_mem, orig_res_mem = self.log[0]
        used_mem, res_mem = self.log[-1]

        def diff(a, b):
            return Text(f'{b - a:+10,d}', style='green' if b > a else 'red')

        yield Text('Current Usage', style='yellow')
        yield Text(f'{100 * (used_mem / res_mem):3,.3f}')
        yield Text(f'{used_mem:10,d} / {res_mem:10,d}')
        if len(self.log) > 1:
            yield Text('Change', style='yellow')
            yield diff(orig_used_mem, used_mem) + ' / ' + diff(orig_res_mem, res_mem)


class Alive:

    _alive: bool
    callback: Optional[Callable]

    def __init__(self,
                 state: bool = True,
                 callback: Optional[Callable] = None,
                 true_label: str = 'active',
                 false_label: str = 'inactive') -> None:
        self.callback = callback
        self._alive = state
        self.true_label = true_label
        self.false_label = false_label

    def alive(self):
        self._alive = True

    def stopped(self):
        self._alive = False

    def __rich_console__(self, console: Console, options: ConsoleOptions) -> RenderResult:
        if self.callback is not None and self._alive:
            self._alive = self.callback()
        if self._alive:
            yield f'[green]{Characters.closed_circle}  {self.true_label}'
        else:
            yield f'[red]{Characters.open_circle}  {self.false_label}'

    def __rich_measure__(self, console: Console, options: ConsoleOptions) -> Measurement:
        max_length = 3 + max(len(self.true_label), len(self.false_label))
        min_length = 3 + max(len(self.true_label), len(self.false_label))
        return Measurement(min_length, max_length)


class Status:

    _status: str
    callback: Optional[Callable]

    states: dict[str, Any] = {
        'active': Text(f'{Characters.closed_circle}  active', style='green'),
        'inactive': Text(f'{Characters.open_circle}  inactive', style='red'),
        'alive': Text(f'{Characters.closed_circle}  alive', style='green'),
        'stopped': Text(f'{Characters.open_circle}  stopped', style='red'),
        'open': Text(f'{Characters.open_circle}', style='red'),
        'closed': Text(f'{Characters.closed_circle}', style='green'),
        'working': Styled(Spinner('dots'), style='cyan'),
    }

    def __init__(self, status: str, callback: Optional[Callable] = None) -> None:
        self._status = status
        self.callback = callback

    def set(self, status: str):
        self._status = status

    def __rich__(self):
        return self.states.get(self._status)


class Table (rich.table.Table):

    def __init__(self, *args, **kwargs):
        defaults = {'expand': True, 'show_header': False, 'box': box.SIMPLE, 'style': 'blue'}
        super().__init__(*args, **{**defaults, **kwargs})


def log(msg: Union[str, RenderableType]):

    # # Log using current trainer
    # from .trainer import get_current_trainer
    # trainer = get_current_trainer()

    # trainer.logs.append(msg)

    console.log(msg)


def section(message: str, module: str = 'Trainer', color: str = 'blue', cons: Optional[Console] = None):
    msg = Padding(f'[ {module} ] {message}', (1, 0), style=color)
    if cons is not None:
        cons.print(msg)
    console.print(msg)


def check(msg: str, color: str = 'blue'):
    console.print(f' [{color}]{Characters.closed_circle}[reset]   {msg}')
