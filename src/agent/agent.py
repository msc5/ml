import abc
from collections import defaultdict
import itertools
import os
import random
import signal
import threading
import traceback
from typing import Any, Callable, Optional, cast

import imageio
import numpy as np
import torch
import wandb

from .. import plots as pp
from ..cli import console
from ..dot import Dot
from ..mp import ManagedQueue, Pool
from ..options import OptionsModule
from ..renderables import Table, check, section
from ..shared import OnlineResults
from ..util import Metadata, RedirectStream, Timer
from .wrappers import Wrapper
from ..module import Module


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


def io_loop(queues: dict[str, ManagedQueue], io_lock: Any):

    signal.signal(signal.SIGINT, signal.SIG_IGN)

    while True:

        env, cache = queues['in'].get()
        if env is None and cache is None:
            return

        dir = cache['dir']
        if not os.path.exists(dir):
            os.makedirs(dir)

        name = f'episode_{cache["episode"]}'
        file = os.path.join(dir, name)

        # Create image and write to file
        fps = 60

        with io_lock:

            if cache.get('values') is not None:
                animation = pp.gif_diff_values(cache['values'], tag=f'{file}_values')
                del animation

            if cache.get('render') is not None:
                render = cache['render'].numpy().astype(np.uint8)
                imageio.mimwrite(file + '.mp4', render, fps=fps)
                imageio.mimwrite(file + '.gif', render, fps=fps)
                del render

            with Metadata(dir) as meta:
                if not 'runs' in meta.data:
                    meta.data['runs'] = {}
                entry = {'status': cache['status'],
                         # 'step': cache['step'],
                         'score': cache['score'].tolist(),
                         'reward': cache['reward'].tolist(),
                         'returns': cache['returns'].tolist()}
                meta.data['runs'][name] = entry

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

    log: bool = False
    discount: float = 0.99
    parallel_envs: int = 1
    n_episodes: Optional[int] = 15
    max_len: Optional[int] = 1000
    min_len: Optional[int] = None

    alive: dict = {}
    dead: dict = {}
    cache: dict = {}

    def build(self):

        from ..mp import get_current_manager
        self.manager = get_current_manager()
        self.io_lock = self.manager.Lock()

        self.parallel_envs = max(1, self.parallel_envs)
        self.alive, self.dead = {}, {}

        self.metrics = Dot()
        self.steps = Dot(total=0)
        for env in range(self.parallel_envs):
            self.steps[env] = Dot(step=0, episode=0)

        # Initialize Environments
        self.envs = {}
        with RedirectStream():
            self.envs = {i: Wrapper(self.environment, id=i) for i in range(self.parallel_envs)}
            self.x_size, self.a_size = self.envs[0].spaces()

        from ..trainer import CurrentTrainer
        if CurrentTrainer is not None:
            self.dir = os.path.join(CurrentTrainer.dir, 'agent')
            if not os.path.exists(self.dir):
                os.makedirs(self.dir)

        self._results = self.manager.OnlineResults()

    def reset_env(self, env: int = 0):
        """
        Resets an environment.
        """

        self.steps[env].step = 0
        self.data['X'][env] = self.envs[env].reset()
        self.data['A'][env] = self.envs[env].sample_action()
        self.data['R'][env] = 0
        self.data['T'][env] = 0
        self.data['render'][env] = self.envs[env].render()

        data = {key: val[env] for key, val in self.data.items()}

        return data

    def reset(self, n_episodes: Optional[int] = None):
        """
        Resets agent.
        """

        n_episodes = n_episodes or self.n_episodes

        # Reset step data
        self.data = {}
        self.data['X'] = torch.zeros((self.parallel_envs, self.x_size), dtype=torch.float32)
        self.data['A'] = torch.zeros((self.parallel_envs, self.a_size), dtype=torch.float32)
        self.data['R'] = torch.zeros(self.parallel_envs, dtype=torch.float32)
        self.data['T'] = torch.zeros(self.parallel_envs, dtype=torch.bool)
        self.data['render'] = torch.zeros((self.parallel_envs, 256, 256, 3))

        # Reset all environments
        for env in self.envs:
            self.reset_env(env)

        # Reset variables
        self.alive, self.dead = {}, {}
        self.cache = {env: defaultdict(list) for env in self.envs.keys()}
        self.queue = list(range(cast(int, n_episodes)))

        # Assign episodes to environments
        for env in range(self.parallel_envs):
            self.episode_reassign(env)

    def step(self,
             env: int = 0,
             action: Optional[torch.Tensor] = None,
             render: bool = False,
             **_):

        data = {}

        # Choose action
        if action is None:
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

    def _get_last(self, data: Any):
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

    def _table(self, cache: dict, dead: dict):

        table = Table('Episode', 'Environment', 'Score', 'Reward', 'Returns', 'Steps', 'Status',
                      show_header=True, title='Results')

        def format(data: Any):
            data = self._get_last(data)
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

    def save(self, env: int, dir: str, fps: int = 60, **_):

        cache = {}
        for key, val in self.cache[env].items():
            if isinstance(val, list) and len(val) != 0:
                if isinstance(val[0], torch.Tensor):
                    cache[key] = torch.stack(val)
                else:
                    cache[key] = torch.tensor(val)
            else:
                cache[key] = val

        if not os.path.exists(dir):
            os.makedirs(dir)

        name = f'episode_{cache["episode"]}'
        file = os.path.join(dir, name)

        # Create image and write to file
        fps = 60

        if cache.get('values') is not None:
            pp.gif_diff_values(cache['values'], tag=f'{file}_values')

        if cache.get('render') is not None:
            render = cache['render'].numpy().astype(np.uint8)
            imageio.mimwrite(file + '.mp4', render, fps=fps)
            imageio.mimwrite(file + '.gif', render, fps=fps)

        with Metadata(dir) as meta:
            if not 'runs' in meta.data:
                meta.data['runs'] = {}
            entry = {'status': cache['status'],
                     # 'step': cache['step'],
                     'score': cache['score'].tolist(),
                     'reward': cache['reward'].tolist(),
                     'returns': cache['returns'].tolist()}
            meta.data['runs'][name] = entry

    def episode_reassign(self, env: int, **kwargs):

        if self.queue:
            self.alive[env] = self.queue.pop(0)
            self.steps[env].episode = self.alive[env]
            default = {'episode': self.alive[env],
                       'environment': env,
                       'status': 'alive',
                       'steps': 0}
            self.cache[env] = defaultdict(list, default)
            for key, val in self.reset_env(env, **kwargs).items():
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

    def episode_step(self,
                     env: int,
                     dir: Optional[str] = None,
                     save: bool = False,
                     collect_results: bool = True,
                     **kwargs):

        action = self.data['A'][env]
        data = self.step(env, action, **kwargs)
        cache = cast(dict, self.cache[env])
        for key, val in data.items():
            self.data[key][env] = val
            cache[key].append(val)

        returns = cache['returns'][-1] if 'returns' in cache else 0.0
        cache['steps'] = self.steps[env].step
        cache['reward'].append(data['R'].item())
        cache['returns'].append(returns + cache['reward'][-1])
        cache['score'].append(self.envs[env].score(cache['returns'][-1]) * 100)

        # Sync to shared struct
        keys = ['score', 'reward', 'returns', 'steps', 'status', 'environment', 'episode']
        if collect_results:
            vals = {key: self._get_last(val) for key, val in cache.items() if key in keys}
            self._results.set_current(env, vals)

        if self.episode_is_done(env):
            cache['status'] = 'complete'

            # Collect completed episode
            self.dead[self.alive[env]] = cache
            if collect_results:
                self._results.set_complete(env)

            if dir is not None:
                self.save(env, dir, **kwargs)
            self.episode_reassign(env)

    @torch.no_grad()
    def run_episodes(self,
                     actor: Callable,
                     n_episodes: Optional[int] = None,
                     dir: Optional[str] = None,
                     results: Optional[OnlineResults] = None,
                     collect_results: bool = True,
                     render: bool = True,
                     save: bool = True,
                     stop: Optional[threading.Event] = None,
                     **kwargs):
        """
        Runs an episode on the environment. Optionally uses an Actor.
        """

        self._results = results or self._results
        self.reset(n_episodes=n_episodes)

        if collect_results:
            self._results.reset_current()

        # --------------------------------------------------------------------------------
        # Loop
        # --------------------------------------------------------------------------------

        self.metrics.time = Timer()
        while self.alive:
            self.metrics.time(reset=True, step=True)

            # Keep environments that are still alive
            self.alive = {k: v for k, v in self.alive.items() if v != None}

            # Generate actions for environments that are still alive
            states = self.data['X'][list(self.alive)]
            actions = actor(states)
            for act, env in enumerate(self.alive):
                self.data['A'][env] = actions[act].cpu()

            # Step environments
            for env in list(self.alive):
                self.episode_step(env, dir=(dir or self.dir),
                                  render=render, save=save, **kwargs)
            self.steps.total += 1

            if stop is not None and stop.is_set():
                if save:
                    for env in list(self.alive):
                        self.save(env, dir=(dir or self.dir), **kwargs)
                break

        if collect_results:
            self._results.reset_current()
            self._results.reset_history()

        return self.dead

    @torch.no_grad()
    def run_steps(self,
                  actor: Callable,
                  n_steps: int,
                  stop: Optional[threading.Event] = None,
                  **kwargs):
        """
        Runs an episode on the environment. Optionally uses an Actor.
        """

        self._dir = dir or self.dir
        self.reset(n_episodes=n_steps)

        # --------------------------------------------------------------------------------
        # Loop
        # --------------------------------------------------------------------------------

        self.metrics.time = Timer()
        for _ in range(n_steps):
            self.metrics.time(reset=True, step=True)

            # Keep environments that are still alive
            self.alive = {k: v for k, v in self.alive.items() if v != None}

            # Generate actions for environments that are still alive
            states = self.data['X'][list(self.alive)]
            actions = actor(states)
            for act, env in enumerate(self.alive):
                self.data['A'][env] = actions[act].cpu()

            # Step environments
            for env in list(self.alive):
                self.episode_step(env, **kwargs)
            self.steps.total += 1

            if stop is not None and stop.is_set():
                break

        # Collect all episodes
        for env in self.alive:
            self.dead[self.alive[env]] = self.cache[env]

        return self.dead

    def close(self):
        section('Exiting', module='Agent', color='yellow')

        for wrapper in self.envs.values():
            wrapper.close()
        check('Wrappers closed', color='yellow')

        check('Finished', color='yellow')
        console.print()
