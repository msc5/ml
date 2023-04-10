import os
import threading
from typing import Callable, Optional, cast

import imageio
import numpy as np
import torch

from ..cli import console
from ..options import OptionsModule
from ..renderables import check, section
from ..util import Metadata, RedirectStream
from .wrappers import Wrapper
from ..data import OnlineDataset


class Agent (OptionsModule):
    """
    Helper class for interacting with Gym environments.
    """

    environment: str

    data: dict[str, torch.Tensor]
    buffer: OnlineDataset

    log: bool = False
    discount: float = 0.99
    parallel_envs: int = 1
    n_episodes: Optional[int] = 15
    max_len: Optional[int] = 1000
    min_len: Optional[int] = None

    x_size: int
    a_size: int
    frame_shape: list[int] = [256, 256, 3]

    def build(self):

        from ..mp import get_current_manager
        self.manager = get_current_manager()
        self.results = self.manager.OnlineResults()

        # Number of environments to run in parallel
        self.parallel_envs = max(1, self.parallel_envs)

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
            self.results = CurrentTrainer._online_results

        self.p_episode = 0

    def reset_env(self, env: int = 0):
        """
        Resets an environment.
        """

        # Reset live environment
        self.states[env] = self.envs[env].reset()
        self.frames[env] = []
        self.scores[env] = {'returns': 0.0, 'score': 0.0}
        self.steps[env] = 0

        return self.states[env]

    def reset(self):
        """
        Resets agent.
        """

        # Reset step data
        self.states = torch.zeros((self.parallel_envs, self.x_size), dtype=torch.float32)
        self.frames = {env: [] for env in self.envs}
        self.scores = {env: {'returns': 0.0, 'score': 0.0} for env in self.envs}
        self.steps = {env: 0 for env in self.envs}

        # Reset all environments
        for env in self.envs:
            self.reset_env(env)

        if self.results is not None:
            self.results.reset_current()

    def save(self, env: int, complete: bool = True, fps: int = 60, **_):

        name = f'episode_{self.p_episode}'
        file = os.path.join(self.dir, name)

        # Create image and write to file
        fps = 60

        render = np.stack(self.frames[env]).astype(np.uint8)
        imageio.mimwrite(file + '.mp4', render, fps=fps)
        imageio.mimwrite(file + '.gif', render, fps=fps)

        with Metadata(self.dir) as meta:
            meta.data[self.p_episode] = {**self.scores[env], 'steps': self.steps[env], 'complete': complete}

        self.p_episode += 1

    def step(self,
             env: int,
             state: torch.Tensor,
             action: torch.Tensor,
             render: bool = False,
             buffer: bool = True,
             save: bool = False,
             results: bool = False,
             **kwargs):
        """
        Steps an environment forward using action and collects data.
        """

        data = {'X': state}

        # Render environment
        if render:
            frame = self.envs[env].render() if render else torch.zeros(self.frame_shape)
            self.frames[env].append(frame)

        # Choose action
        if action is None:
            data['A'] = self.envs[env].sample_action()
        else:
            data['A'] = action.detach().cpu()

        # Step environment forward
        data['N'], data['R'], data['T'] = self.envs[env].step(data['A'].squeeze())
        self.steps[env] += 1

        # Track scores
        self.scores[env]['returns'] += data['R'].item()
        self.scores[env]['score'] = cast(float, self.envs[0].score(self.scores[env]['returns']))
        if self.results is not None:
            self.results.set_current(env, {'steps': self.steps[env], **self.scores[env]})

        # Push step data to buffer
        if buffer:
            self.buffer.push(data)

        # Episode completed
        if data['T']:
            if save:
                self.save(env, **kwargs)
            if results and self.results is not None:
                self.results.set_complete(env)
            self.reset_env(env)

        return data['T'].item()

    @torch.no_grad()
    def run_steps(self,
                  actor: Callable,
                  n_steps: int,
                  stop: Optional[threading.Event] = None,
                  **kwargs):
        """
        Runs "n_steps" in the environment. Optionally uses an Actor.
        """

        self.reset()

        for _ in range(n_steps):

            # Generate actions
            actions = actor(self.states)

            # Step environments
            for env in self.envs:
                self.step(env, self.states[env], actions[env], **kwargs)

            if stop is not None and stop.is_set():
                break

    @torch.no_grad()
    def run_episodes(self,
                     actor: Callable,
                     n_episodes: int,
                     render: bool = True,
                     buffer: bool = False,
                     save: bool = True,
                     stop: Optional[threading.Event] = None,
                     results: bool = True,
                     **kwargs):
        """
        Runs "n_episodes" episodes in the environment. Optionally uses an Actor.
        """

        self.reset()

        alive = list(range(min(self.parallel_envs, n_episodes)))
        p_episodes = self.parallel_envs

        while alive:

            # Generate actions for environments that are still alive
            actions = actor(self.states[alive])

            # Step environments
            terminate = []
            for i in range(len(alive)):

                # Select states and actions
                env = alive[i]
                action = actions[i]
                state = self.states[alive][i]

                done = self.step(env, state, action,
                                 render=render, buffer=buffer,
                                 save=save, results=results,
                                 **kwargs)

                if done:
                    if p_episodes < n_episodes:
                        terminate += [False]
                        p_episodes += 1
                    else:
                        terminate += [True]
                else:
                    terminate += [False]

            # Discard completed episodes
            alive = [alive[i] for i in range(len(alive)) if not terminate[i]]

            if stop is not None and stop.is_set():
                if save:
                    for env in alive:
                        self.save(env, complete=False, **kwargs)
                break

        if self.results is not None:
            self.results.reset_history()
            self.results.reset_current()

    def close(self):
        section('Exiting', module='Agent', color='yellow')

        for wrapper in self.envs.values():
            wrapper.close()
        check('Wrappers closed', color='yellow')

        check('Finished', color='yellow')
        console.print()
