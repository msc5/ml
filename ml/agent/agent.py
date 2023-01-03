import abc
from collections import defaultdict
from dataclasses import dataclass
import os
import random
from typing import Any, Optional, cast
import imageio

import torch.multiprocessing as mp
import numpy as np
import torch
import threading

from src.ml.util import Metadata, Timer

from ..cli import console
from ..data import OfflineDataset
from ..options import Dot, OptionsModule
from .wrappers import Wrapper
from .. import plots as pp
from ..renderables import Alive, Progress, Table, Manager
from ..io import generate_name


@dataclass
class State:
    obs: torch.Tensor
    reward: float
    done: bool


@dataclass
class Transition:
    a: State
    b: State
    action: torch.Tensor


class Actor (OptionsModule):

    _initial: Optional[torch.Tensor]
    _actions: Optional[torch.Tensor]

    def __init__(self,
                 actions: Optional[torch.Tensor] = None,
                 initial: Optional[torch.Tensor] = None) -> None:
        self._actions = actions
        self._initial = initial
        self.step = 0

    @abc.abstractmethod
    def __call__(self, *args, **kwargs):
        return self

    @abc.abstractmethod
    def initial(self) -> torch.Tensor:
        if self._initial is not None:
            return self._initial
        else:
            return None

    @abc.abstractmethod
    def pre(self) -> None:
        if self._actions is not None:
            self.step = 0
        return None

    @abc.abstractmethod
    def act(self, data: dict, cache: dict, envs: dict[int, int]) -> torch.Tensor:
        if self._actions is not None:
            action = self._actions[self.step % len(self._actions)]
            self.step += 1
            return action
        else:
            return None

    @abc.abstractmethod
    def post(self, cache: dict, env: int) -> None:
        return None


def io_loop(in_queue: mp.Queue, out_queue: mp.Queue, io_lock: Any, processes: Any, dir: str):

    # pid = str(uuid.uuid4().hex)
    # pid = str(random.randint(0, 100))
    pid = generate_name()
    # processes[pid] = Alive(state=True)
    processes['io'] = processes['io']({pid: Alive(state=True)})

    while True:

        env, cache = in_queue.get()
        out_queue.put(in_queue.size())
        if env is None and cache is None:
            # processes[pid] = Alive(state=False)
            processes['io'][pid].stopped()
            return

        path, name = os.path.join(dir, 'gifs'), f'episode_{cache["episode"]}'
        file = os.path.join(path, name)

        try:

            # Create image and write to file
            if cache['render'] != []:
                if cache['values'] != []:
                    pp.gif_diff_values(cache['values'], tag=f'{file}_values')
                render = cache['render'].numpy().astype(np.uint8)
                imageio.mimwrite(file + '.mp4', render, fps=30)
                imageio.mimwrite(file + '.gif', render, fps=30)
                del render

            # Write metadata
            with io_lock:
                with Metadata(path, default={'runs': {}}) as meta:
                    entry = {'status': cache['status'],
                             'step': cache['step'],
                             'score': cache['score'].tolist(),
                             'reward': cache['reward'].tolist(),
                             'returns': cache['returns'].tolist()}
                    # 'action': cache['A'].tolist()}
                    meta.data['runs'][name] = entry

        except KeyboardInterrupt:
            return

        except Exception:
            console.print_exception()

        del env
        del cache

        # if io_stop.is_set() and in_queue.empty():
        #     processes[pid] = False
        #     return


class Agent (OptionsModule):

    environment: str
    buffer: OfflineDataset

    data: dict[str, torch.Tensor]

    x_size: int
    a_size: int

    discount: float = 0.99

    max_episodes: Optional[int] = 15
    max_len: Optional[int] = 400
    min_len: Optional[int] = 16

    parallel: int = 1
    exploration: bool = False
    noise_max: float = 0.4
    noise_min: float = 0.0
    noise_steps: int = 100000

    rollout: int = 100

    alive: dict = {}
    dead: dict = {}
    cache: dict = {}

    def build(self):
        self.manager = Manager()
        self.manager.start()

        self.parallel = max(1, self.parallel)
        self.envs = {i: Wrapper(self.manager, self.environment, id=i) for i in range(self.parallel)}
        self.alive, self.dead = {}, {}

        self.processes = self.manager.Dot(envs={}, io={})
        # self.processes['envs'] = self.manager.Dot({wrapper.name: wrapper.status for wrapper in self.envs.values()})
        # self.processes['io'] = self.manager.Dot()
        for wrapper in self.envs.values():
            self.processes['envs'] = self.processes['envs']({wrapper.name: wrapper.status})

        self.metrics = Dot()
        self.steps = Dot(total=0)
        for env in range(self.parallel):
            self.steps[env] = Dot(step=0, episode=0)

        from ..trainer import CurrentTrainer
        if CurrentTrainer is not None:
            self.dir = os.path.join(CurrentTrainer.dir, 'agent')
            path = os.path.join(self.dir, 'gifs')
            if not os.path.exists(path): os.makedirs(path)

        if self.exploration: self.metrics.noise = 0.0
        self.frame = torch.empty(0)
        self.push = self.stop = None

        self.data = self.container()

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

    def reset(self, env: int = 0, **_):
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

    def load(self, name: str):
        with Metadata(self.dir) as meta:
            self.steps = Dot(meta.data['steps'])
            self.metrics = Dot(meta.data['metrics'])
        self.data = torch.load(os.path.join(self.dir, name + '.pt'))

    def __rich__(self):

        table = Table('Episode', 'Environment', 'Score', 'Reward', 'Returns', 'Steps', 'Status',
                      show_header=True, title='Results')

        def add_row(key: int, run: dict):
            if 'step' in run:
                # actions = [f'{action:3.3f}' for action in run['A'][-1]]
                # actions = '(' + (', ').join(actions) + ')'
                table.add_row(f'{run["episode"]}',
                              f'{run["environment"]}',
                              f'{run["score"][-1]:3.3f}',
                              f'{run["reward"][-1]:3.3f}',
                              f'{run["returns"][-1]:3.3f}',
                              # f'{actions}',
                              f'{run["step"]}',
                              run['status'], style='green' if run['status'] == 'alive' else 'red')

        for key, run in sorted(self.cache.items(), key=lambda x: -x[1].get('score', [0])[-1]):
            add_row(key, run)
        table.add_section()
        for key, run in sorted(self.dead.items(), key=lambda x: -x[1].get('score', [0])[-1]):
            add_row(key, run)

        # Compute Stats
        if self.dead:
            scores = torch.tensor([e['score'][-1] for e in self.dead.values()
                                   if (e['score'] is not None
                                       and len(e['score']) > 0
                                       and e['status'] == 'complete')])
            mean = scores.mean().item()
            std = scores.std().nan_to_num().item()
            table.add_section()
            table.add_row(None, f'{mean:3.3f} +- {std:3.3f}')

        return table

    @torch.no_grad()
    def episode(self,
                actor: Actor,
                max_len: Optional[int] = None,
                min_len: Optional[int] = None,
                episodes: Optional[int] = None,
                render: bool = True,
                **kwargs):
        """
        Runs an episode on the environment. Optionally uses an Actor, which
        implements or extends the Actor class.
        """

        max_len = max_len or self.max_len
        min_len = min_len or self.min_len
        episodes = episodes or self.max_episodes

        for env in self.envs:
            self.reset(env, **kwargs)

        actor.pre()

        self.alive, self.dead = {}, {}
        self.data = self.container()
        self.cache = {env: defaultdict(list) for env in self.envs.keys()}
        self.queue = list(range(cast(int, episodes)))

        def reassign(env: int):
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
                # self.metrics.envs[env].status.stopped()

        def done(env: int) -> bool:
            done = False
            if self.data['T'][env]:
                if min_len is not None and self.steps[env].step <= min_len:
                    pass
                else:
                    done = True
            if max_len is not None and self.steps[env].step >= max_len:
                done = True
            return done

        def step(env: int):
            action = self.data['A'][env]
            # action = torch.full_like(action, self.alive[env])
            data = self.step(env, action, render=render, **kwargs)
            cache = cast(dict, self.cache[env])
            for key, val in data.items():
                self.data[key][env] = val
                cache[key].append(val)

            returns = cache['returns'][-1] if 'returns' in cache else 0.0
            cache['step'] = self.steps[env].step
            cache['reward'].append(data['R'].item())
            cache['returns'].append(returns + cache['reward'][-1])
            cache['score'].append(self.envs[env].score(cache['returns'][-1]) * 100)

            if done(env):
                cache['status'] = 'complete'
                dead_keys = ['score', 'reward', 'returns', 'step', 'status', 'environment', 'episode']
                self.dead[self.alive[env]] = {key: val for key, val in cache.items() if key in dead_keys}
                actor.post(cache, env)
                save(env)
                reassign(env)

        def save(env: int):
            cache = {}
            for key, val in self.cache[env].items():
                if isinstance(val, list) and len(val) != 0:
                    if isinstance(val[0], torch.Tensor):
                        cache[key] = torch.stack(val)
                    else:
                        cache[key] = torch.tensor(val)
                elif isinstance(val, torch.Tensor):
                    cache[key] = val.clone()
                else:
                    cache[key] = val
            in_queue.put((env, cache))
            self.metrics.in_queue.update('queue', in_queue.size())

        def progress(queue: mp.Queue):
            while True:
                size = queue.get()
                if size is None:
                    return
                self.metrics.in_queue.update('queue', size)

        for env in range(self.parallel):
            reassign(env)

        io_lock = self.manager.Lock()
        maxsize = 4 * self.parallel
        in_queue = self.manager.Queue(maxsize=maxsize)
        out_queue = self.manager.Queue(maxsize=maxsize)
        self.metrics.in_queue = Progress(tasks=['queue'], total=maxsize)

        # Start queue thread
        progress_loop = threading.Thread(target=progress, args=[out_queue], daemon=True)
        progress_loop.start()

        # Start IO processes
        args = [in_queue, out_queue, io_lock, self.processes, self.dir]
        processes = [mp.Process(args=args, target=io_loop) for _ in range(4)]
        for process in processes: process.start()

        self.metrics.time = Timer()
        while self.alive:
            self.metrics.time(reset=True, step=True)

            self.alive = {k: v for k, v in self.alive.items() if v != None}

            action = actor.act(self.data, self.cache, self.alive)
            for act, env in enumerate(self.alive):
                self.data['A'][env] = action[act].cpu()

            for env in list(self.alive):
                step(env)

            self.steps.total += 1

            try:
                for env in list(self.alive):
                    steps = self.steps[env].step
                    if steps != 0 and (steps in [1, 20, 50] or steps % 100 == 0):
                        save(env)
            except Exception:
                console.print_exception()

        for process in enumerate(processes):
            in_queue.put((None, None))
        for process in processes:
            process.join()
        # self.processes['agent'] = self.processes

    def close(self):
        for wrapper in self.envs.values():
            wrapper.close()
            self.processes['envs'][wrapper.name] = wrapper.status
        # self.processes['envs'] = Dot({wrapper.name: wrapper.status for wrapper in self.envs.values()})
