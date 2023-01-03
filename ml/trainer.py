import abc
import contextlib
from numbers import Number
import os
import queue
import sys
import time
from typing import Literal, Optional

import psutil
from rich.console import group
from rich.layout import Layout
from rich.live import Live
from rich.progress import track
from rich.text import Text
from rich import box
import torch
import wandb

import multiprocessing as mp
import torch.multiprocessing as tmp

import src.ml as ml

from .agent import Actor, Agent
from .cli import console
from .io import generate_name
from .util import Thread


class Trainer (ml.Module):

    _name: str
    _tag: str

    log: bool = False                           # Log results to wandb
    debug: bool = False                         # Debug with no live display in console
    save_model: int = 500                       # Save model every n steps. No saving if 0.
    load_model: Optional[str] = None            # Path to load model from.
    results_dir: str = 'results'                # Directory to store results
    rollout: int = 500
    max_episodes: Optional[int] = None
    eval: bool = False

    tag: Optional[str] = None
    mode: Optional[str] = None

    retrain: list[str] = []
    freeze: list[str] = []

    layout: Layout

    def __new__(cls, *_):
        ml.section('Building Trainer')
        return super().__new__(cls)

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

    def start(self, project: str):
        """
        Begins training loop.
        """

        self.loops = []
        self.queues = ml.Dot({'episode': queue.Queue()})

        self.metrics = ml.Dot(time=ml.Timer())
        self.steps = ml.Dot({'train': 0, 'test': 0})
        if self.save_model != 0: self.steps.saved = 0

        # Metadata
        with ml.Metadata(self.results_dir) as meta:
            meta.data['number'] = number = meta.data.get('number', 0) + 1
        self.name = f'{number:03d}-{self.tag or generate_name()}'

        self.dir = os.path.join(self.results_dir, self._tag, self.name)
        with ml.Metadata(self.dir) as meta:
            meta.data['opts'] = self.opts._dict()

        self.threads = {}
        self.processes = ml.Dot()

        global CurrentTrainer
        CurrentTrainer = self

        # Load previous model
        if self.load_model is not None:
            load_path = os.path.join(self.results_dir, self.load_model)
            with ml.Metadata(load_path) as meta:
                self.opts.model = ml.Dot(meta.data['opts']['model'])

        self._build()
        self.to(self.device)

        # Debug
        self.system = ml.Dot()
        self.system.cpu = ml.Dot(usage=ml.Progress(tasks=[0]))
        self.system.memory = ml.Progress(tasks=[0])
        if self.device == 'cuda':
            import GPUtil
            self.gpus = GPUtil.getGPUs()
            self.system.gpu = ml.Dot(memory=ml.Progress(tasks=list(range(len(self.gpus)))),
                                     load=ml.Progress(tasks=list(range(len(self.gpus)))))
        if self.debug:
            torch.autograd.anomaly_mode.set_detect_anomaly(True)

        # Initialize model optimizers
        self.optim = ml.Optimizers(self.optimizers())
        if self.freeze is not None: self.optim._freeze = self.freeze

        # Load previous model
        if self.load_model is not None:
            self.load(self.load_model)

        # Initialize wandb
        self.run = None
        if self.log:
            self.run = wandb.init(project=project, name=self.name,
                                  config={k: v.value for k, v in self.opts})

        ml.section('Options')
        console.log(self.opts)

    def train(self, project: str):
        """
        Training loop.
        """

        self.start(project)
        ml.section(f'Training Run [cyan3]{self.name}')

        self.threads = {**self.threads, **{loop.__name__: Thread(target=loop) for loop in self.loops}}

        context = (contextlib.nullcontext() if self.debug else
                   Live(self.dashboard(), console=console, refresh_per_second=8))

        try:
            [thread.start() for thread in self.threads.values()]
            with context as self.live:
                while True: time.sleep(1e5)

        except KeyboardInterrupt:
            self.exit()

    def exit(self):
        ml.section('Shutting Down')
        if self.save_model != 0 and self.steps.train > self.rollout:
            self.save()
        if self.log: wandb.finish(quiet=True)
        # sys.exit()
        os._exit(1)

    def train_loop(self):
        """
        Trains model on collected actions.
        """

        while True:
            self.update_system()
            self.train_step()
            if self.log:

                log = {}
                for key, val in self.metrics:
                    if not any([frozen in key for frozen in self.freeze]):
                        if isinstance(val.value, Number):
                            log[key] = val.value
                        elif isinstance(val.value, ml.Ranges):
                            log[f'{key}.min'] = val.value._min
                            log[f'{key}.max'] = val.value._max
                            log[f'{key}.mean'] = val.value._mean
                wandb.log(log, step=self.steps.train)

            if self.save_model != 0 and self.steps.train != 0 and self.steps.train % self.save_model == 0:
                self.save()
            self.metrics.time(step=True)
            self.steps.train += 1

    def update_system(self):
        self.system.memory.update(0, completed=int(psutil.virtual_memory().percent))
        self.system.cpu.usage.update(0, completed=int(psutil.cpu_percent()))
        if self.device == 'cuda':
            for i, gpu in enumerate(self.gpus):
                memory_used = int((gpu.memoryUsed / gpu.memoryTotal) * 100)
                load = int(gpu.load * 100)
                self.system.gpu.memory.update(i, completed=memory_used)
                self.system.gpu.load.update(i, completed=load)

    def test_loop(self):
        """
        Tests model on collected actions.
        """

        self.test_step()
        self.exit()

    def sync(self):
        self.update_system()
        if self.log:
            wandb.log({k: v.value for k, v in self.metrics if isinstance(v.value, Number)})

    @group()
    def dashboard(self):

        @group()
        def title():
            table = ml.Table()
            name = self.name + ' ([green]Logging ⬤ [reset])' if self.log else self.name
            table.add_row('Model', self._name)
            table.add_row('Run', name)
            if self.load_model is not None:
                table.add_row('Loaded', self.load_model)
            yield table

        @group()
        def program():
            yield dot('Threads', ml.Dot(self.threads))
            yield dot('Processes', self.processes)

        @group()
        def dot(name: str, dot: ml.Dot):
            yield Text.from_markup(f'[italic]{name}', justify='center')
            yield dot

        @group()
        def status():
            yield program()
            yield dot(f'Model [magenta]({self.params.count:,})[reset]', self._params(freeze=self.freeze))

        @group()
        def progress():
            yield dot('Progress', self.steps)
            yield dot('System', self.system)

        # Layout
        layout = Layout()
        layout.split_column(
            Layout(name='title', size=5),
            Layout(name='content'))
        layout['content'].split_row(
            Layout(name='meta'),
            Layout(name='metrics'),
            Layout(name='progress'))
        layout['content']['meta'].split_column(
            Layout(name='status'),
            Layout(name='dataset', visible=False))

        layout['title'].update(title())
        layout['content']['meta']['status'].update(status())
        layout['content']['metrics'].update(dot('Metrics', self.metrics))
        layout['content']['progress'].update(progress())
        self.layout = layout

        yield layout

    def save(self):
        ml.section(f'Saving Model to [magenta]{self.dir}')

        # Save progress
        self.steps.saved = self.steps.train
        with ml.Metadata(self.dir) as meta:
            meta.data['steps'] = self.steps._dict()

        # Save weights
        model_path = os.path.join(self.dir, 'model.pt')
        torch.save(self.state_dict(), model_path)

    def load(self, run: str):
        ml.section(f'Loading Model [magenta]{run}')
        model_path = os.path.join(self.results_dir, run, 'model.pt')
        state_dict = torch.load(model_path, map_location=self.device)
        for key, val in self.named_parameters():
            if key in state_dict:
                if any([retrain in key for retrain in self.retrain]):
                    console.print(f'Retraining: [yellow]{key}[reset]')
                    del state_dict[key]
                elif (state_dict[key].shape != val.shape):
                    console.print(f'Shape Mismatch: [red]{key}[reset]')
                    console.print(f'Expected: {val.shape}')
                    console.print(f'Loaded:   {state_dict[key].shape}')
                    del state_dict[key]
            else:
                console.print(f'Missing Parameter: [yellow]{key}[reset]')
        self.load_state_dict(state_dict, strict=False)


class OnlineTrainer (Trainer):

    agent: Agent
    actor: Actor

    seed_episodes: int = 1000

    def start(self, project: str):
        super().start(project)
        self.metrics.agent = self.agent.metrics
        self.steps.agent = self.agent.steps
        self.processes.agent = self.agent.processes

    def exit(self):
        self.agent.close()
        super().exit()

    @group()
    def dashboard(self):
        super().dashboard()
        if self.eval:

            @group()
            def dot(name: str, dot: ml.Dot):
                yield Text(name, style='italic', justify='center')
                yield dot

            @group()
            def program():
                yield dot('Threads', ml.Dot(self.threads))
                yield dot('Processes', self.processes)

            @group()
            def combine():
                yield self.agent
                layout = Layout()
                layout.split_row(Layout(name='progress'), Layout(name='metrics'))
                layout['progress'].update(program())
                layout['metrics'].update(dot('Metrics', self.metrics))
                yield layout

            self.layout.split_column(self.layout['title'], combine())

        yield self.layout

    def actor_loop(self):
        """
        Evaluate Actor on environment.
        """

        while True:
            kwargs = self.queues.episode.get()
            self.agent.episode(actor=self.actor, render=True, **kwargs)
            self.queues.episode.task_done()

    def test_step(self) -> None:
        """
        Tests the actor on the environment using a batch from an offline
        dataset or replay buffer. 
        """

        self.agent.episode(actor=self.actor, render=True)

    def fill_buffer(self, episodes: int = 1):
        """
        Fill buffer with necessary amount of episodes to begin training. Policy
        used is random.
        """

        ml.section(f'Collecting {self.seed_episodes} seed episodes')

        if self.debug:
            for _ in range(episodes):
                self.agent.episode(actor=self.actor)
        else:
            for _ in track(range(episodes), description='Progress', transient=True):
                self.agent.episode(actor=self.actor)


CurrentTrainer: Optional[Trainer] = None
