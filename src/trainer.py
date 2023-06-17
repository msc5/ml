import abc
from functools import partial
from numbers import Number
import os
import re
import subprocess
import threading
import time
from typing import Any, Optional
from inspect import getmembers, ismethod


from humanize import naturalsize
from rich import box
from rich.console import Group, group as rgroup
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Column
from rich.text import Text
import torch
import wandb

from .cli import console
from .module import Module
from .mp import Thread
from .options import Options
from .dot import Dot
from .renderables import Alive, Table, check, section
from .util import Keyboard, Metadata, Ranges, Screens, Steps, System, Timer
from .shared import OnlineResults
from .database import influxdb


os.environ["WANDB_CONSOLE"] = "off"
os.environ['D4RL_SUPPRESS_IMPORT_ERROR'] = '1'


class Trainer (Module):

    # Configuration options
    log: bool = False
    debug: bool = False
    online_eval: bool = False
    mode: str = 'train'

    # Loading / Saving
    # e.g. /results/narldiff/{group}/001-spring-green
    load_model: list[str] = []
    save_every: int = 5000
    rollout_every: int = 5000
    max_episodes: Optional[int] = None
    results_dir: str = 'results'
    tags: list[str] = []
    group: str = 'misc'
    wandb_group: str = ''
    wandb_resume: bool = False
    retrain: list[str] = []
    note: str = ''

    # Instance variables
    metrics: Dot
    progress: Steps
    system: System
    _threads: dict
    _loops: list
    _gpus: list

    # Private variables
    _online_results: OnlineResults

    # Renderables
    layout: Layout
    timer: Timer

    # Multiprocessing
    exit_event: threading.Event

    # Modules
    model: Module

    @abc.abstractmethod
    def optimizers(self, *args, **kwargs) -> dict:
        return {}

    def __init__(self, opts: Optional[Options] = None):
        super().__init__(opts)

        # When trainer is initialized, it takes over as the global trainer
        # object for all threads and subthreads.
        global CurrentTrainer
        CurrentTrainer = self

        from .shared import session
        session.trainer = self

        self._init_renderables()

    def _build(self):
        section('Building')
        return super()._build()

    def start(self):
        """
        Initializes module for training loop.
        """

        self.timer = Timer()
        self.progress = Steps(keys=['session'])
        self.progress.add_modulo('save', every=self.save_every)
        self.system = System()
        self._logged = Layout()
        self.agent_table = Layout()
        self._online_results = OnlineResults()

        section('Starting')

        # Initialize shared session
        from .shared import session
        self.session = session
        self.info = session.start(self)

        influxdb.initialize()

        # Build Trainer and models
        with Live(get_renderable=self._render_building, transient=True):
            self._build()
            self.to(self.device)
        check('Built Trainer')

        # Load previous model
        if self.load_model != []:
            for path in self.load_model:
                self.load(path)
                check('Loaded Weights')

        # Debug mode
        if self.debug:
            torch.autograd.set_detect_anomaly(True)  # type: ignore

        # Save options and session info
        with Metadata(self.info.dir) as meta:
            meta.data['opts'] = self.opts._dict()
            meta.data['model'] = self.model.__class__.__name__
        check('Saved Options and Configuration')

        check('Finished')

        section(f'Training Run [cyan3]{self.info.name}')

        # session.threads = {}
        # for name, method in getmembers(self, predicate=ismethod):
        #     prefix = 'loop_'
        #     if name[:len(prefix)] == prefix:
        #         session.threads[name[len(prefix):]] = Thread(target=method, daemon=False)

        self.screens = Screens(self._init_screens())

        def block():
            while all([thread.is_alive() for thread in session.threads.values()]):
                time.sleep(1.0)

        try:
            # [thread.start() for thread in session.threads.values()]
            session.start_threads()
            if self.debug:
                block()
            else:
                self.screens.select('live')
                with Keyboard() as keyboard:
                    keyboard.callbacks['L'] = partial(self.screens.select, 'live')
                    keyboard.callbacks['O'] = partial(self.screens.select, 'opts')
                    keyboard.callbacks['D'] = partial(self.screens.select, 'dataset')
                    keyboard.callbacks['C'] = partial(self.screens.select, 'config')
                    keyboard.callbacks['W'] = partial(self.screens.select, 'logged')
                    keyboard.callbacks['A'] = partial(self.screens.select, 'agent')
                    keyboard.callbacks['\x1b'] = self.screens.clear
                    block()

        except KeyboardInterrupt:
            self.screens.clear()
            section('Shutting Down [red](KeyboardInterrupt)[reset]')

        except Exception:
            self.screens.clear()
            section('Shutting Down [red](Exception)[reset]')
            console.print_exception()

        finally:
            self.screens.clear()
            self.exit()

    def exit(self):

        section('Exiting')

        # Import session
        from .shared import session

        # Save model
        self.save()
        check('Saved')

        # Stop threads
        self.info.exit_event.set()
        for thread in session.threads.values():
            if thread.is_alive():
                thread.join()
        check('Threads stopped')

        influxdb.close()

        # Finish Wandb
        if self.info.wandb is not None:
            console.print()
            self.info.wandb.finish(quiet=True)
            console.print()
        check('Finished Wandb')

        # Print information
        console.print(self.dashboard())

        self.session.manager.shutdown()
        check('Finished')

    def generate_train_loop(self):
        self.train_step_complete()
        if self.info.exit_event.is_set():
            return False
        else:
            return True

    def train_step_complete(self):

        # self.log_metrics()
        self.log_metrics_database()

        if self.progress.modulo('session', 'save'):
            self.save()

        self.progress.step('train')
        self.progress.step('session')

    def log_metrics_database(self):
        influxdb.log_metrics(self.model.metrics)

    def log_metrics(self):

        if self.log:
            log = {}
            for model in self._selected:

                for key, val in model.metrics:
                    key = f'{self.group}-{model.__class__.__name__}{key}'
                    if isinstance(val.value, Number):
                        log[key] = val.value

                for key, val in model.ranges:
                    key = f'{self.group}-{model.__class__.__name__}{key}'
                    if isinstance(val.value, Ranges):
                        log[f'{key}.min'] = val.value._min
                        log[f'{key}.max'] = val.value._max
                        log[f'{key}.mean'] = val.value._mean
                        log[f'{key}.std'] = val.value._std

            step = self.progress.get('train') if self.wandb_resume else self.progress.get('session')
            wandb.log(log, step=step)
            self._logged.update(Dot(log))

            # Log to json
            with Metadata(self.info.dir, name='log') as meta:
                meta.data[step] = log

    def _init_renderables(self):

        empty = Text('None', style='black')

        @rgroup()
        def title():
            table = Table(box=box.ROUNDED, style='black')
            table.add_row('Module', self.opts.sys.module)
            table.add_row('Group', Text(self.group, style='cyan'))
            table.add_row('Run', Text(self.info.name, style='magenta'))
            table.add_row('Loaded', Text(str(self.load_model), style='green') or empty)
            yield table

        @rgroup()
        def info():
            table = Table(box=None)

            if self.note != '':
                note = Text(self.note, style='yellow')
                table.add_row('Note', note)

            # Github
            branch = Text('Branch: ') + Text(self.info.github.branch, style='magenta')
            commit = Text('Commit: ') + Text('\"' + self.info.github.commit + '\"', style='green')
            table.add_row('Github', branch, commit)

            # Influxdb
            table.add_row('Influxdb', '', Alive(self.info.influxdb))
            table.add_row('MySQL', '', Alive(self.info.mysql))

            # Wandb and Slurm
            wandb_id = Text(self.info.wandb.id, style='magenta') if self.info.wandb is not None else empty
            slurm_id = Text(self.info.slurm_id, style='magenta') if self.info.slurm_id != '' else empty
            table.add_row('Wandb', Text('ID: ') + wandb_id, Alive(self.info.wandb is not None))
            table.add_row('Slurm', Text('ID: ') + slurm_id, Alive(self.info.slurm_id != ''))

            yield Panel(table, border_style='black')

        @rgroup()
        def system():
            table = Table(box=box.ROUNDED, style='black')
            size = ''
            try:
                size = int(subprocess.check_output(['du', '-s', self.info.dir]).split()[0])
                size = naturalsize(size * 512)
            except:
                pass
            table.add_row('Time', self.timer)
            table.add_row('Storage', Text(str(size), style='blue'))
            table.add_row('Mode', Text(self.mode, style='green'))
            yield table

        @rgroup()
        def dot(name: str, dot: Dot):
            yield Text.from_markup(f'[bold italic]{name}', justify='center')
            yield dot

        @rgroup()
        def status():
            yield dot(f'Model [magenta]({self._param_count:,})[reset]', self.model._render())

        @rgroup()
        def progress():
            yield dot('Threads', self.session.main_thread)
            yield dot('Progress', self.progress)
            yield dot('System', self.system)

        self.renderables = Dot(title=title, info=info, system=system,
                               dot=dot, progress=progress, status=status)

    def _init_screens(self):
        # config = [f'{k}: {str(v.value)}' for k, v in self._gather_opts()]
        screen_args = {'refresh_per_second': 8, 'screen': True, 'console': console}
        screens = {'live': Live(get_renderable=self.dashboard, **screen_args),
                   'opts': Live(self.opts, **screen_args),
                   'logged': Live(self._logged, **screen_args), }
        # 'config': Fuzzy(console, config)}
        if self.online_eval:
            screens['agent'] = Live(self, **screen_args)
        return screens

    @rgroup()
    def dashboard(self):

        # Layout
        layout = Layout()
        layout.split_column(
            Layout(name='title', size=7),
            Layout(name='content'))
        layout['title'].split_row(
            Layout(name='run', ratio=2),
            Layout(name='info', ratio=3),
            Layout(name='system', ratio=2))
        layout['content'].split_row(
            Layout(name='meta', ratio=2),
            Layout(name='metrics', ratio=3),
            Layout(name='progress', ratio=2))
        layout['content']['meta'].split_column(
            Layout(name='status'),
            Layout(name='agent', visible=False))

        metrics = Group(self.renderables.dot('Metrics', self.metrics), self.renderables.dot('Ranges', self.ranges))

        layout['title']['run'].update(self.renderables.title())
        layout['title']['info'].update(self.renderables.info())
        layout['title']['system'].update(self.renderables.system())
        layout['content']['meta']['status'].update(self.renderables.status())
        layout['content']['metrics'].update(metrics)
        layout['content']['progress'].update(self.renderables.progress())
        self.layout = layout

        if self.online_eval:
            self.layout['content']['metrics'].split_row(
                Layout(name='met'),
                Layout(name='agent'))
            self.layout['content']['metrics']['met'].update(metrics)
            self.layout['content']['metrics']['agent'].update(
                self.renderables.dot('Results', self._render_online_results()))

        yield layout

    def save(self):

        # Save weights
        model_path = os.path.join(self.info.dir, 'model.pt')
        torch.save(self.model.state_dict(), model_path)

        # Save Metadata
        with Metadata(self.info.dir) as meta:
            meta.data['opts'] = self.opts._dict()
            meta.data['model'] = self.model.__class__.__name__

    def load(self, run: str):

        section(f'Loading Model [magenta]{run}')

        metadata = Metadata.load(run)
        selected = metadata['model']
        module = self.get_mlmodule(selected)

        if module is not None:

            model_path = os.path.join(run, 'model.pt')
            state_dict = torch.load(model_path, map_location=self.device)
            for key, val in module.named_parameters():
                if key in state_dict:
                    if any([retrain in key for retrain in self.retrain]):
                        check(f'Retraining: [yellow]{key}[reset]', color='magenta')
                        del state_dict[key]
                    elif (state_dict[key].shape != val.shape):
                        check(f'Shape Mismatch: [red]{key}[reset]', color='magenta')
                        console.print(f'     Expected: {val.shape}')
                        console.print(f'     Loaded:   {state_dict[key].shape}')
                        del state_dict[key]
                else:
                    check(f'Missing Parameter: [yellow]{key}[reset]', color='magenta')
            to_delete = set()
            for key, val in state_dict.items():
                if any([retrain in key for retrain in self.retrain]):
                    check(f'Retraining: [red]{key}[reset]', color='magenta')
                    to_delete.add(key)
                if 'samples' in key:
                    check(f'Ignoring: [red]{key}[reset]', color='magenta')
                    to_delete.add(key)
            for key in to_delete:
                del state_dict[key]
            module.load_state_dict(state_dict, strict=False)

        else:
            raise Exception('Loaded module does not exist in current module')

        check('Finished', color='magenta')
        console.print('')

    def _render_online_table(self, cache: dict, name: str = 'Unnamed Run', style: str = 'black'):

        def format_float(data: Any):
            if type(data) == float:
                return f'{data: 3.3f}'
            elif type(data) == int:
                return f'{data: ,}'
            else:
                return f'{data}'

        def ordered(data):
            data = dict(filter(lambda x: isinstance(x[1], dict), data.items()))
            data = dict(sorted(data.items(), key=lambda x: -x[1]['score']))
            return data

        group = []
        tags = ['score', 'returns', 'steps', 'episode']
        cols = [t.capitalize() for t in tags]
        columns = (Column(col, ratio=1) for col in cols)
        table = Table(*columns, show_header=True, box=None, header_style='bold yellow')
        for run in ordered(cache).values():
            row = (f'{format_float(run[tag])}' for tag in tags)
            table.add_row(*row)
        group += [table]

        if 'mean' in cache and 'std' in cache:
            group += [f'[bold green]{cache["mean"]: 3.3f} ± {cache["std"]:3.3f}']

        group = Group(*group)
        table = Panel(group, border_style=style, title=name, title_align='left')

        return table

    @rgroup()
    def _render_online_results(self):

        if self._online_results is not None:

            if (current := self._online_results.get_current()) != {}:
                yield self._render_online_table(current, name='Current Run', style='yellow')

            if len(history := self._online_results.get_history()) > 0:

                # Sort results by mean score
                caches = sorted(enumerate(history), key=lambda x: -x[1].get('mean', 0))
                for run, cache in caches[:3]:
                    yield self._render_online_table(cache, name=f'Run {run + 1} / {len(history)}')

        else:
            yield Panel('No Results', border_style='red')


CurrentTrainer: Optional[Trainer] = None


def get_current_trainer():
    if CurrentTrainer is not None:
        return CurrentTrainer
    else:
        raise Exception('Trainer not started!')
