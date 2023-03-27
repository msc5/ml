import abc
from functools import partial
from numbers import Number
import os
import subprocess
import threading
import time
from typing import Any, Optional

import git
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
from .data import OfflineDataset
from .io import generate_name
from .module import Module
from .mp import Manager, Thread
from .options import Options
from .dot import Dot
from .renderables import Alive, Table, check, section
from .util import Fuzzy, Keyboard, Metadata, Ranges, Screens, Steps, System, Timer


os.environ["WANDB_CONSOLE"] = "off"
os.environ['D4RL_SUPPRESS_IMPORT_ERROR'] = '1'


class Trainer (Module):

    # Configuration options
    log: bool = False
    debug: bool = False
    mode: str = 'train'
    online_eval: bool = False

    # Loading / Saving
    # e.g. /results/narldiff/{group}/001-spring-green
    load_model: str = ''
    save_model: int = 500
    rollout: int = 500
    max_episodes: Optional[int] = None
    results_dir: str = 'results'
    tags: list[str] = []
    group: str = 'misc'
    slurm_id: str = ''
    wandb_id: str = ''
    wandb_group: str = ''
    wandb_resume: bool = False
    retrain: list[str] = []
    override_opts: bool = False
    note: str = ''

    # Instance variables
    metrics: Dot
    progress: Steps
    system: System
    _selected: list = []
    _threads: dict
    _loops: list
    _gpus: list

    # Renderables
    layout: Layout
    timer: Timer

    # Multiprocessing
    exit_event: threading.Event

    # Modules
    model: Module
    dataset: OfflineDataset

    def _build(self):
        section('Building')
        return super()._build()

    @abc.abstractmethod
    def train_step(self) -> None:
        raise NotImplementedError()

    @abc.abstractmethod
    def test_step(self) -> None:
        raise NotImplementedError()

    @abc.abstractmethod
    def actor_initial(self) -> Optional[torch.Tensor]:
        return None

    @abc.abstractmethod
    def episode_callback(self, *args, **kwargs) -> None:
        pass

    @abc.abstractmethod
    def optimizers(self, *args, **kwargs) -> dict:
        return {}

    def __init__(self, opts: Optional[Options] = None):
        super().__init__(opts)
        global CurrentTrainer
        CurrentTrainer = self
        self._init_renderables()

    def _reset(self):
        self._loops = []
        self.timer = Timer()
        self.progress = Steps(keys=['session', 'train', 'save', 'rollout'])
        self.system = System()
        self.main_thread = Thread(main=True)
        self._logged = Layout()
        self.agent_table = Layout()
        self._online_results = []

        g = Dot()
        g.repo = git.Repo(os.getcwd())  # type: ignore
        g.master = g.repo.head.reference
        g.branch = str(g.master.name)
        g.commit = str(g.master.commit.message).replace('\n', ' ').strip()
        self.github = g

        # Get slurm job name
        name = os.environ.get('SLURM_JOB_NAME')
        id = os.environ.get('SLURM_JOB_ID')
        if name is not None and id is not None:
            self.slurm_id = f'{name}-{id}'

    def start(self):
        """
        Begins training loop.
        """

        self._reset()

        section('Starting')

        # Initialize manager for multiprocessing
        self.manager = Manager()
        self.manager.start()
        self.exit_event = self.manager.Event()

        # Load previous model options
        if self.load_model != '':
            with Metadata(self.load_model) as meta:
                config = Options(meta.data['config'])
                config = config(self.parse())
                # progress = Dot(meta.data['progress'])
                # progress.session = 0
                wandb_id = meta.data.get('wandb', '')
                sys = self.opts.sys
            if self.override_opts:
                self.apply_opts(config)
                self.opts.sys = sys
            self.wandb_id = wandb_id
            # self.progress = progress
            check('Loaded Model Configuration')

        # Metadata
        with Metadata(self.results_dir) as meta:
            meta.data['number'] = number = meta.data.get('number', 0) + 1
        self.name = f'{number:03d}-{self.group if self.group != "misc" else generate_name()}'
        self.dir = os.path.join(self.results_dir, self.opts.sys.module, self.group, self.name)
        check(f'Created Directory [cyan]{self.dir}[reset]')

        self._build()
        self.to(self.device)
        check('Built Trainer')

        # # Debug
        # self._system.cpu = Dot(usage=Progress(tasks=[0]))
        # self._system.memory = Progress(tasks=[0])
        # if self.device == 'cuda':
        #     # import GPUtil
        #     # self._gpus = GPUtil.getGPUs()
        #     # self._system.gpu = Dot(memory=Progress(tasks=list(range(len(self._gpus)))),
        #     #                        load=Progress(tasks=list(range(len(self._gpus)))))
        #     self._system.gpu = Dot(memory=Progress(tasks=['memory']))
        # if self.debug:
        #     torch.autograd.anomaly_mode.set_detect_anomaly(True)
        # check('Initialized System')

        # Load previous model
        if self.load_model != '':
            self.load(self.load_model)
            check('Loaded Weights')

        # Initialize wandb
        self.run = None
        if self.log:
            console.print()
            self.run = wandb.init(project=self.opts.sys.module, name=self.name,
                                  group=self.wandb_group,
                                  tags=[*self.tags, self.group],
                                  # config={k: v.value for k, v in self.opts},
                                  config={k: v.value for k, v in self._gather_params()},
                                  id=self.wandb_id if self.wandb_resume else None)
            console.print()
            if self.wandb_resume:
                check(f'Resumed wandb run [cyan]{self.wandb_id}[reset]')
            else:
                check('Initialized Wandb')

        with Metadata(self.dir) as meta:
            meta.data['opts'] = self.opts._dict()
            # meta.data['progress'] = self.progress._dict()
            meta.data['config'] = self._gather_params()._dict()
            meta.data['parse'] = self.parse()._dict()
            if self.log and self.run is not None:
                meta.data['wandb'] = self.run.id
        check('Saved Options and Configuration')

        if 'train' in self.opts.mode:
            ln_source = os.path.join(os.getcwd(), self.dir)
            ln_dest = os.path.join(self.results_dir, self.opts.sys.module, self.group, 'latest')
            if os.path.exists(ln_dest):
                os.unlink(ln_dest)
            os.symlink(ln_source, ln_dest)
            check(f'Created symlink at [cyan]{ln_dest}[reset] from [cyan]{ln_source}[reset]')

        check('Finished')

    def train(self):
        """
        Training loop.
        """

        self.start()
        section(f'Training Run [cyan3]{self.name}')

        self._threads = {loop.__name__: Thread(target=loop, daemon=False) for loop in self._loops}
        self.screens = Screens(self._init_screens())

        def block():
            while all([thread.is_alive() for thread in self._threads.values()]):
                time.sleep(1.0)

        try:
            starters = list(self._threads.values())
            [thread.start() for thread in starters]
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

        # Save model
        self.save()
        check('Saved')

        # Stop threads
        # with Live(self.main_thread, console=console, transient=True):
        self.exit_event.set()
        for thread in self._threads.values():
            if thread.is_alive():
                thread.join()
        check('Threads stopped')

        # Finish Wandb
        if self.run is not None:
            console.print()
            self.run.finish(quiet=True)
            console.print()
        check('Finished Wandb')

        # Print information
        console.print(self.dashboard())

        self.manager.shutdown()
        check('Finished')

        # except KeyboardInterrupt:
        #     active = mp.active_children()
        #     for child in active:
        #         child.terminate()
        #     sys.exit(0)

    def train_loop(self):
        """
        Trains model on collected actions.
        """

        while True:
            self.train_step()
            self.train_step_complete()
            if self.exit_event.is_set():
                return

    def train_step_complete(self):
        self.log_metrics()
        if self.progress.modulo('session', self.save_model, dest='save', gt=self.save_model):
            self.save()
        self.progress.step('train')
        self.progress.step('session')

    def log_metrics(self):
        if self.log:
            log = {}
            for model in self._selected:
                for key, val in model.metrics:
                    key = f'{model.__class__.__name__}{key}'
                    if isinstance(val.value, Number):
                        log[key] = val.value
                for key, val in model.ranges:
                    if isinstance(val.value, Ranges):
                        log[f'{key}.min'] = val.value._min
                        log[f'{key}.max'] = val.value._max
                        log[f'{key}.mean'] = val.value._mean
            step = self.progress.get('train') if self.wandb_resume else self.progress.get('session')
            wandb.log(log, step=step)
            self._logged.update(Dot(log))

    def test_loop(self):
        """
        Tests model on collected actions.
        """
        # section('Starting Test Loop')
        self.test_step()
        # section('Exiting Test Loop')

    def _init_renderables(self):

        empty = Text('None', style='black')

        @rgroup()
        def title():
            table = Table(box=box.ROUNDED, style='black')
            table.add_row('Module', self.opts.sys.module)
            table.add_row('Group', Text(self.group, style='cyan'))
            table.add_row('Run', Text(self.name, style='magenta'))
            table.add_row('Loaded', Text(self.load_model, style='green') or empty)
            yield table

        @rgroup()
        def info():
            table = Table(box=None)

            if self.note != '':
                note = Text(self.note, style='yellow')
                table.add_row('Note', note)

            # Github
            branch = Text('Branch: ') + Text(self.github.branch, style='magenta')
            commit = Text('Commit: ') + Text('\"' + self.github.commit + '\"', style='green')
            table.add_row('Github', branch, commit)

            # Wandb and Slurm
            wandb_id = Text(self.run.id, style='magenta') if self.run is not None else empty
            slurm_id = Text(self.slurm_id, style='magenta') if self.slurm_id != '' else empty
            table.add_row('Wandb', Text('ID: ') + wandb_id, Alive(self.run is not None))
            table.add_row('Slurm', Text('ID: ') + slurm_id, Alive(self.slurm_id != ''))

            yield Panel(table, border_style='black')

        @rgroup()
        def system():
            table = Table(box=box.ROUNDED, style='black')
            size = ''
            try:
                size = int(subprocess.check_output(['du', '-s', self.dir]).split()[0])
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
            yield dot('Threads', self.main_thread)
            yield dot('Progress', self.progress)
            yield dot('System', self.system)

        self.renderables = Dot(title=title, info=info, system=system,
                               dot=dot, progress=progress, status=status)

    def _init_screens(self):
        config = [f'{k}: {str(v.value)}' for k, v in self._gather_opts()]
        screen_args = {'refresh_per_second': 8, 'screen': True, 'console': console}
        screens = {'live': Live(get_renderable=self.dashboard, **screen_args),
                   'opts': Live(self.opts, **screen_args),
                   'dataset': Live(self.dataset.table(), **screen_args),
                   'logged': Live(self._logged, **screen_args),
                   'config': Fuzzy(console, config)}
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
            Layout(name='run'),
            Layout(name='info'),
            Layout(name='system'))
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
        model_path = os.path.join(self.dir, 'model.pt')
        torch.save(self.state_dict(), model_path)

    def load(self, run: str):
        section(f'Loading Model [magenta]{run}')
        model_path = os.path.join(run, 'model.pt')
        state_dict = torch.load(model_path, map_location=self.device)
        for key, val in self.named_parameters():
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
        check('Finished', color='magenta')
        console.print('')

        self.load_state_dict(state_dict, strict=False)

    def update_online(self, cache: dict):

        def format(data: Any):
            if type(data) == float:
                return f'{data: 3.3f}'
            else:
                return f'{data}'

        def ordered(data):
            data = dict(filter(lambda x: isinstance(x[1], dict), data.items()))
            data = dict(sorted(data.items(), key=lambda x: -x[1]['score']))
            return data

        tags = ['score', 'returns', 'steps']
        cols = [t.capitalize() for t in tags]

        columns = (Column(col, ratio=1) for col in cols)
        table = Table(*columns, show_header=True, box=None, header_style='bold yellow')
        for run in ordered(cache).values():
            row = (f'{format(run[tag])}' for tag in tags)
            table.add_row(*row)
        stats = f'[bold green]{cache["mean"]: 3.3f} ± {cache["std"]:3.3f}'
        table.add_section()
        table.add_row(None, stats)
        table.add_section()
        table = Panel(table, border_style='black', title=f'Run {len(self._online_results)}', title_align='left')

        self._online_results.append((cache['mean'], table))

    @rgroup()
    def _render_online_results(self):

        # Sort results by mean score
        tables = list(sorted(self._online_results, key=lambda x: -x[0]))
        for _, table in tables[:5]:
            yield table


CurrentTrainer: Optional[Trainer] = None


def get_current_trainer():
    if CurrentTrainer is not None:
        return CurrentTrainer
    else:
        raise Exception('Manager not started!')
