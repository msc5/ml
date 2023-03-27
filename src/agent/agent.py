import abc
from collections import defaultdict
import os
import random
import signal
import threading
import traceback
from typing import Any, Optional, cast

import imageio
import numpy as np
import torch
import wandb

from ..util import Metadata, Timer

from .. import plots as pp
from ..cli import console
from ..mp import ManagedQueue, Pool
from ..options import OptionsModule
from ..dot import Dot
from ..renderables import Table, section, check
from .wrappers import Wrapper


class Actor:

    @abc.abstractmethod
    def __call__(self, *args, **kwargs):
        return self

    @abc.abstractmethod
    def initial(self) -> torch.Tensor:
        return None

    @abc.abstractmethod
    def before(self) -> None:
        """
        Function that is run before an episode runs.
        """
        return None

    @abc.abstractmethod
    def act(self, data: dict, cache: dict, envs: dict) -> torch.Tensor:
        """
        Function that is called at each step of the episode.
        """
        return None

    @abc.abstractmethod
    def after(self, cache: dict, env: int) -> None:
        """
        Function that is run after an episode runs.
        """
        return None


def io_loop(queues: dict[str, ManagedQueue], io_lock: Any, dir: str, log: bool = False):

    signal.signal(signal.SIGINT, signal.SIG_IGN)

    while True:

        env, cache = queues['in'].get()
        if env is None and cache is None:
            return

        if not os.path.exists(dir):
            os.makedirs(dir)

        name = f'episode_{cache["episode"]}'
        file = os.path.join(dir, name)

        try:
            # Create image and write to file
            fps = 60

            with io_lock:
                if cache.get('values') is not None:
                    animation = pp.gif_diff_values(cache['values'], tag=f'{file}_values')
                    if log:
                        video = {'name': name + '_values',
                                 'video': animation,
                                 'step': cache['step'],
                                 'format': 'gif',
                                 'fps': fps}
                        queues['out'].put(('log video', video))
                    del animation

                if cache.get('render') is not None:
                    render = cache['render'].numpy().astype(np.uint8)
                    imageio.mimwrite(file + '.mp4', render, fps=fps)
                    imageio.mimwrite(file + '.gif', render, fps=fps)
                    if log:
                        video = {'name': name,
                                 'video': render.transpose(0, 3, 1, 2),
                                 'step': cache['step'],
                                 'format': 'gif',
                                 'fps': fps}
                        queues['out'].put(('log video', video))
                    del render

                with Metadata(dir) as meta:
                    if not 'runs' in meta.data:
                        meta.data['runs'] = {}
                    entry = {'status': cache['status'],
                             'step': cache['step'],
                             'score': cache['score'].tolist(),
                             'reward': cache['reward'].tolist(),
                             'returns': cache['returns'].tolist()}
                    meta.data['runs'][name] = entry

        except:
            queues['out'].put(('exception', {'traceback': traceback.format_exc(), 'module': 'Agent IO'}))

        del env
        del cache


def wandb_loop(queues: dict[str, ManagedQueue]):
    while True:
        cmd, msg = queues['out'].get()
        if cmd == None and msg == None:
            return
        elif cmd == 'log video':
            video = wandb.Video(msg['video'], caption=msg['name'], fps=msg['fps'], format=msg['format'])
            wandb.log({msg['name']: video}, step=msg['step'])
        elif cmd == 'exception':
            section('Exception', module=msg.get('module'), color='red')
            console.log(msg.get('traceback'))


class Agent (OptionsModule):

    environment: str

    data: dict[str, torch.Tensor]

    x_size: int
    a_size: int

    discount: float = 0.99

    max_episodes: Optional[int] = 15
    max_len: Optional[int] = 1000
    min_len: Optional[int] = 16

    parallel: int = 1
    exploration: bool = False
    noise_max: float = 0.4
    noise_min: float = 0.0
    noise_steps: int = 100000

    rollout: int = 500
    log: bool = False

    alive: dict = {}
    dead: dict = {}
    cache: dict = {}

    pool: Optional[Pool] = None

    def _build(self):
        section('Building', module='Agent', color='yellow')
        return super()._build()

    def build(self):

        from ..mp import get_current_manager
        self.manager = get_current_manager()
        self.io_lock = self.manager.Lock()

        self.envs = {}

        self.parallel = max(1, self.parallel)
        self.alive, self.dead = {}, {}

        self.metrics = Dot()
        self.steps = Dot(total=0)
        for env in range(self.parallel):
            self.steps[env] = Dot(step=0, episode=0)

        from ..trainer import CurrentTrainer
        if CurrentTrainer is not None:
            self.dir = os.path.join(CurrentTrainer.dir, 'agent')
            if not os.path.exists(self.dir):
                os.makedirs(self.dir)

        if self.exploration: self.metrics.noise = 0.0
        self.frame = torch.empty(0)
        self.push = self.stop = None

        self.data = self.container()

        # Initialize Environments
        self.envs = {i: Wrapper(self.environment, id=i) for i in range(self.parallel)}
        for env in self.envs:
            self.reset(env)

    def container(self):
        data = {}
        data['X'] = torch.zeros((self.parallel, self.x_size))
        data['A'] = torch.zeros((self.parallel, self.a_size))
        data['R'] = torch.zeros(self.parallel)
        data['T'] = torch.zeros(self.parallel)
        data['render'] = torch.zeros((self.parallel, 256, 256, 3))
        for val in data.values():
            val.share_memory_()
        return data

    def reset(self, env: int = 0):
        self.steps[env].step = 0
        self.data['X'][env] = self.envs[env].reset()
        self.data['A'][env] = self.envs[env].sample_action()
        self.data['R'][env] = 0
        self.data['T'][env] = 0
        self.data['render'][env] = self.envs[env].render()
        return {key: val[env] for key, val in self.data.items()}

    def explore(self):
        """
        Determine whether to choose a random action
        """
        if self.exploration:
            # Compute noise upper bound
            noise = self.noise_max - self.steps.total / self.noise_steps
            noise = max(self.noise_min, noise)
            self.metrics.noise = noise
            return random.uniform(0, 1) < noise
        else:
            return False

    def step(self,
             env: int = 0,
             action: Optional[torch.Tensor] = None,
             render: bool = True,
             **_):

        data = {}

        # Choose action
        if action is None or self.explore():
            data['A'] = self.envs[env].sample_action()
        else:
            data['A'] = action.detach().cpu()

        # Step environment forward
        data['X'], data['R'], data['T'] = self.envs[env].step(data['A'].squeeze())
        self.steps[env].step += 1

        # Render environment
        if render:
            data['render'] = self.envs[env].render()

        return data

    def get_last(self, data: Any):
        """ 
        if item is a list, return the last item.
        """
        if isinstance(data, list) or isinstance(data, torch.Tensor):
            if len(data) > 0:
                return data[-1]
            else:
                return None
        else:
            return data

    def stat(self, data: dict):
        scores = torch.tensor([e['score'] for e in data.values()])
        mean = scores.mean().item()
        std = scores.std().nan_to_num().item()
        return mean, std

    def _table(self, cache: dict, dead: dict):

        table = Table('Episode', 'Environment', 'Score', 'Reward', 'Returns', 'Steps', 'Status',
                      show_header=True, title='Results')

        def format(data: Any):
            data = self.get_last(data)
            if type(data) == float:
                return f'{data: 3.3f}'
            else:
                return f'{data}'

        def add_row(run: dict):
            if 'step' in run:
                table.add_row(f'{format(run["episode"])}',
                              f'{format(run["environment"])}',
                              f'{format(run["score"])}',
                              f'{format(run["reward"])}',
                              f'{format(run["returns"])}',
                              f'{format(run["step"])}',
                              run['status'],
                              style='green' if run['status'] == 'alive' else 'red')

        def stat(data: dict):
            if data:
                scores = torch.tensor([e['score'][-1] for e in data.values()
                                       if (e['score'] is not None and len(e['score']) > 0)])
                if len(scores) > 0:
                    mean = scores.mean().item()
                    std = scores.std().nan_to_num().item()
                    table.add_section()
                    table.add_row(None, None, f'{mean:3.3f} ± {std:3.3f}')

        def ordered(data):
            return sorted(data.items(), key=lambda x: -(x[1].get('score', [0]) or [0])[-1])

        for _, run in ordered(cache):
            add_row(run)
        # stat(self.cache)
        table.add_section()
        for _, run in ordered(dead):
            add_row(run)
        stat(dead)

        return table

    def __rich__(self):
        return self._table(self.cache, self.dead)

    def log_table(self, data: dict):
        columns = ['Episode', 'Environment', 'Score', 'Reward', 'Returns', 'Steps', 'Status']
        rows = []
        for run in data.values():
            rows.append([run["episode"],
                         run["environment"],
                         run["score"][-1],
                         run["reward"][-1],
                         run["returns"][-1],
                         run["step"],
                         run['status']])
        table = wandb.Table(columns=columns, data=rows)
        wandb.log({'results': table})

    def log_performance(self, data: dict):
        scores = torch.tensor([e['score'][-1] for e in data.values()
                               if (e['score'] is not None and len(e['score']) > 0)])
        if len(scores) > 0:
            mean = scores.mean().item()
            std = scores.std().nan_to_num().item()
            table = wandb.Table(columns=['Environment', 'Mean', 'Std'],
                                data=[[self.environment, mean, std]])
            wandb.log({'performance': table})

    def save(self, env: int):
        # cache = {}
        # for key, val in self.cache[env].items():
        #     if isinstance(val, list) and len(val) != 0:
        #         if isinstance(val[0], torch.Tensor):
        #             cache[key] = torch.stack(val)
        #         else:
        #             cache[key] = torch.tensor(val)
        #     elif isinstance(val, torch.Tensor):
        #         cache[key] = val.clone()
        #     else:
        #         cache[key] = val
        # self.io_queue.queues['in'].put((env, cache))
        pass

    def episode_reassign(self, env: int, **kwargs):
        if self.queue:
            self.alive[env] = self.queue.pop(0)
            self.steps[env].episode = self.alive[env]
            default = {'episode': self.alive[env],
                       'environment': env,
                       'status': 'alive'}
            self.cache[env] = defaultdict(list, default)
            for key, val in self.reset(env, **kwargs).items():
                self.cache[env][key].append(val)
        else:
            del self.alive[env]
            del self.cache[env]
            self.steps[env] = Dot(step=0, episode=0)

    def episode_is_done(self, env: int) -> bool:
        done = False
        if self.data['T'][env]:
            if self.min_len is not None and self.steps[env].step <= self.min_len:
                pass
            else:
                done = True
        if self.max_len is not None and self.steps[env].step >= self.max_len:
            done = True
        return done

    def episode_step(self, env: int, actor: Actor, render: bool = True, **kwargs):
        action = self.data['A'][env]
        data = self.step(env, action, render=render, **kwargs)
        cache = cast(dict, self.cache[env])
        for key, val in data.items():
            self.data[key][env] = val
            cache[key].append(val)

        returns = cache['returns'][-1] if 'returns' in cache else 0.0
        cache['steps'] = self.steps[env].step
        cache['reward'].append(data['R'].item())
        cache['returns'].append(returns + cache['reward'][-1])
        cache['score'].append(self.envs[env].score(cache['returns'][-1]) * 100)

        if self.episode_is_done(env):
            cache['status'] = 'complete'
            dead_keys = ['score', 'reward', 'returns', 'steps', 'status', 'environment', 'episode']

            # Collect completed episode
            self.dead[self.alive[env]] = {key: self.get_last(val) for key, val in cache.items() if key in dead_keys}

            actor.after(cache, env)
            self.save(env)
            self.episode_reassign(env)

    @torch.no_grad()
    def episode(self,
                actor: Actor,
                episodes: Optional[int] = None,
                dir: Optional[str] = None,
                log: bool = False,
                render: bool = True,
                stop: Optional[threading.Event] = None,
                **kwargs):
        """
        Runs an episode on the environment. Optionally uses an Actor.
        """

        # --------------------------------------------------------------------------------
        # Setup
        # --------------------------------------------------------------------------------

        episodes = episodes or self.max_episodes
        dir = dir or self.dir

        for env in self.envs:
            self.reset(env)

        actor.before()

        self.alive, self.dead = {}, {}
        self.data = self.container()
        self.cache = {env: defaultdict(list) for env in self.envs.keys()}
        self.queue = list(range(cast(int, episodes)))

        for env in range(self.parallel):
            self.episode_reassign(env)

        # # Initialize pool only once
        # self.pool = self.pool or Pool(4)

        # --------------------------------------------------------------------------------
        # Loop
        # --------------------------------------------------------------------------------

        # self.io_queue = self.pool.queue
        # self.pool.apply_async(target=io_loop, parameters=[self.io_lock, dir, log])

        self.metrics.time = Timer()
        while self.alive:
            self.metrics.time(reset=True, step=True)

            self.alive = {k: v for k, v in self.alive.items() if v != None}

            action = actor.act(self.data, self.cache, self.alive)
            for act, env in enumerate(self.alive):
                self.data['A'][env] = action[act].cpu()

            for env in list(self.alive):
                self.episode_step(env, actor, render, **kwargs)

            self.steps.total += 1

            for env in list(self.alive):
                steps = self.steps[env].step
                if steps != 0 and (steps in [1, 20, 50] or steps % self.rollout == 0):
                    self.save(env)

            if stop is not None and stop.is_set():
                for env in list(self.alive):
                    self.save(env)
                break

        self.dead['mean'], self.dead['std'] = self.stat(self.dead)
        return self.dead

    def close(self):
        section('Exiting', module='Agent', color='yellow')
        for wrapper in self.envs.values():
            wrapper.close()
        check('Wrappers closed', color='yellow')
        # if self.log:
        #     self.log_table(self.dead)
        #     self.log_performance(self.dead)
        #     check('Final results logged', color='yellow')
        check('Finished', color='yellow')
        console.print()
