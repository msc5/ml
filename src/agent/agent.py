import os
import threading
from typing import Callable, Optional, cast

import imageio
import numpy as np
import torch
from torchvision.transforms.functional import crop, resize

from ..cli import console
from ..data import OnlineDataset
from ..options import OptionsModule
from ..renderables import check, section
from ..util import Metadata, RedirectStream
from .wrappers import Wrapper


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
    use_video: bool = False

    x_size: int
    a_size: int
    frame_shape: list[int]

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

    def video_transform(self, frame: torch.Tensor):
        frame_size = self.frame_shape[-1]
        frame = frame.permute(2, 0, 1)
        frame = crop(frame, top=64, left=64, width=128, height=192)
        frame = resize(frame, (frame_size, frame_size), antialias=True)  # type: ignore
        return frame

    def reset_env(self, env: int = 0):
        """
        Resets an environment.
        """

        # Reset live environment
        self.states[env] = self.envs[env].reset()
        self.frames[env] = []
        self.scores[env] = {'returns': 0.0, 'score': 0.0}
        self.steps[env] = 0
        self.episode[env] = self.p_episode
        self.p_episode += 1

        if self.use_video:
            self.v_states[env] = self.video_transform(self.envs[env].render())
        else:
            self.v_states[env] = torch.zeros(1)

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
        self.episode = torch.zeros((self.parallel_envs, 1), dtype=torch.int64)

        if self.use_video:
            self.v_states = torch.zeros((self.parallel_envs, *self.frame_shape), dtype=torch.float32)
        else:
            self.v_states = torch.zeros((self.parallel_envs, 1))

        # Reset all environments
        for env in self.envs:
            self.reset_env(env)

        if self.results is not None:
            self.results.reset_current()

    def save(self, env: int, complete: bool = True, fps: int = 60, **_):

        episode = int(self.episode[env])
        name = f'episode_{episode}'
        file = os.path.join(self.dir, name)

        # Create image and write to file
        render = np.stack(self.frames[env]).astype(np.uint8)
        imageio.mimwrite(file + '.gif', render, fps=fps)  # type: ignore

        with Metadata(self.dir) as meta:
            meta.data[episode] = {**self.scores[env], 'steps': self.steps[env], 'complete': complete}

    def step(self,
             env: int,
             action: Optional[torch.Tensor] = None,
             render: bool = False,
             buffer: bool = True,
             eval: bool = False,
             **kwargs):
        """
        Steps an environment forward using action and collects data.
        """

        data = {'X': self.states[env].clone()}

        # Render environment
        frame = None
        if render:
            frame = self.envs[env].render()
            self.frames[env].append(frame)
        if self.use_video:
            frame = frame if frame is not None else self.envs[env].render()
            frame = self.video_transform(frame)
            self.v_states[env] = data['F'] = frame

        # Choose action
        if action is None:
            data['A'] = self.envs[env].sample_action()
        else:
            data['A'] = action.detach().cpu()

        # Step environment forward
        data['N'], data['R'], data['T'] = self.envs[env].step(data['A'].squeeze())
        self.states[env] = data['N']
        self.steps[env] += 1

        # Track scores
        self.scores[env]['returns'] += data['R'].item()
        self.scores[env]['score'] = cast(float, self.envs[env].score(self.scores[env]['returns']))

        # Push step data to buffer
        if buffer:
            data['I'] = self.episode[env]
            self.buffer.push(data)

        # Push step data to renderable
        if self.results is not None:
            current = {'steps': self.steps[env], 'episode': int(self.episode[env]), **self.scores[env]}
            self.results.set_current(env, current)

        # Episode completed
        if data['T']:

            if eval:

                # # Ignore terminal steps for evaluation episodes
                # if self.steps[env] != 1000:
                self.results.set_complete(env, int(self.episode[env]))
                self.save(env, **kwargs)

            self.reset_env(env)

        return data['T'].item()

    @torch.no_grad()
    def run_steps(self,
                  n_steps: int,
                  n_envs: Optional[int] = None,
                  actor: Optional[Callable] = None,
                  stop: Optional[threading.Event] = None,
                  **kwargs):
        """
        Runs "n_steps" in the environment. Optionally uses an Actor.
        """

        if self.p_episode == 0:
            self.reset()

        # Random actions unless actor provided
        n_envs = n_envs or self.parallel_envs
        i_envs = list(range(n_envs))
        actions = [None] * n_envs

        for _ in range(n_steps):

            # Generate actions
            if actor is not None:
                if self.use_video:
                    actions = actor(self.v_states[i_envs])
                else:
                    actions = actor(self.states[i_envs])

            # Step environments
            for env in i_envs:
                self.step(env, actions[env], **kwargs)

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
                     eval: bool = True,
                     **kwargs):
        """
        Runs "n_episodes" episodes in the environment. Optionally uses an Actor.
        """

        self.reset()

        alive = list(range(min(self.parallel_envs, n_episodes)))
        p_episodes = self.parallel_envs

        while alive:

            # Generate actions for environments that are still alive
            if self.use_video:
                actions = actor(self.v_states[alive])
            else:
                actions = actor(self.states[alive])

            # Step environments
            terminate = []
            for i in range(len(alive)):

                # Select states and actions
                env = alive[i]
                action = actions[i]

                done = self.step(env, action, render=render, buffer=buffer, eval=eval, **kwargs)

                # 1. done and (p_episodes < n_episodes)
                #   -> p_episodes += 1
                #   -> False
                # 2. done and not (p_episodes < n_episodes)
                #   -> True
                # 3. not done:
                #   -> False

                # stop_env

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

        self.reset()

    def close(self):
        section('Exiting', module='Agent', color='yellow')

        for wrapper in self.envs.values():
            wrapper.close()
        check('Wrappers closed', color='yellow')

        check('Finished', color='yellow')
        console.print()
