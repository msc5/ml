import copy
import signal
from typing import Optional, Union

from gym import logger
import gym
from gym.spaces import Box, Discrete
import mujoco_py as mjc
import numpy as np
import torch
import os

from ..cli import console
from ..mp import ManagedQueue, Process
from ..options import OptionsModule
from ..renderables import Alive
from ..util import quiet

logger.set_level(40)

State = tuple[torch.Tensor, torch.Tensor, torch.Tensor]

os.environ['QT_QPA_PLATFORM'] = 'offscreen'


def env_worker(queues: dict[str, ManagedQueue], environment: str):

    signal.signal(signal.SIGINT, signal.SIG_IGN)

    if 'breakout' in environment:
        wrapper = AtariWrapper(environment)
    else:
        wrapper = GymWrapper(environment)

    while True:

        function, input = queues['in'].get()
        if function is None and input is None:
            wrapper.close()
            return

        try:
            if function == 'reset':
                output = wrapper.reset()
            elif function == 'step':
                output = wrapper.step(input.clone())
            elif function == 'render':
                output = wrapper.render()
            elif function == 'score':
                output = wrapper.score(input)
            elif function == 'sample_action':
                output = wrapper.sample_action()
            else:
                raise Exception('Invalid function')

            queues['out'].put(output)
            del function
            del input
            del output

        except:
            wrapper.close()
            return


class Wrapper:

    _alive: bool = False

    def __init__(self, environment: str, id: int = 0):

        self.name = f'Env {id}'
        self.status = Alive(state=True)

        # self.process = Process(target=self.worker, name=self.name, args=[environment])
        self.process = Process(target=env_worker, args=[environment])
        self.in_queue, self.out_queue = self.process.queues['in'], self.process.queues['out']
        self.process.start()

        self._alive = True

    def close(self):
        self.process.close()

    def reset(self):
        self.in_queue.put(('reset', None))
        output = self.out_queue.get()
        result = output.clone()
        del output
        return result

    def step(self, action: Union[int, torch.Tensor]) -> State:
        self.in_queue.put(('step', action))
        output = self.out_queue.get()
        result = State(o.clone() for o in output)
        del output
        return result

    def render(self, height: int = 256, width: int = 256):
        with quiet():
            self.in_queue.put(('render', (height, width)))
            output = self.out_queue.get()
            result = output.clone()
            del output
            return result

    def sample_action(self) -> torch.Tensor:
        self.in_queue.put(('sample_action', None))
        output = self.out_queue.get()
        result = output.clone()
        del output
        return result

    def score(self, returns) -> torch.Tensor:
        self.in_queue.put(('score', returns))
        output = self.out_queue.get()
        result = copy.deepcopy(output)
        del output
        return result


class GymWrapper (gym.Env):

    env: str
    sim: mjc.MjSim

    def __init__(self, environment: str):
        super().__init__()

        with quiet():
            import d4rl as _

        self.env = environment

        self._env = gym.make(environment)
        self.sim = self._env.sim  # type: ignore

        self.action_space = self._env.action_space

        if isinstance(self.action_space, Discrete):
            self.a = self.action_space.n
        elif isinstance(self.action_space, Box):
            (self.a, ) = self.action_space.shape
        else:
            raise NotImplementedError()

    def state(self) -> torch.Tensor:
        # if 'walker' in self.env:
        #     data = self.sim.data
        #     qpos, qvel = data.qpos, data.qvel
        #     state = np.concatenate([qpos, qvel])
        #     state = torch.from_numpy(state.copy())
        #     self._state = state
        return self._state

    def step(self, action: Union[int, torch.Tensor]) -> State:
        if isinstance(action, torch.Tensor):
            # if isinstance(self.action_space, Discrete):
            #     action = int(action.argmax(-1).item())
            # else:
            action = action.cpu().squeeze().numpy()
        state, reward, done, _ = self._env.step(action)
        self._state = torch.from_numpy(state)
        self.state()
        reward = torch.tensor(reward, dtype=torch.float32)
        done = torch.tensor(done, dtype=torch.bool)
        return self._state, reward, done

    def sample_action(self) -> torch.Tensor:
        action = self.action_space.sample()
        if isinstance(self.action_space, Discrete):
            action = torch.zeros((2, self.a))
            action[0, action] = 1
        else:
            action = torch.from_numpy(action.copy())
        return action

    def reset(self) -> torch.Tensor:
        state = self._env.reset()
        state = torch.from_numpy(state)
        self._state = state
        return self.state()

    def render(self, height: int = 256, width: int = 256):
        frame = self.sim.render(height, width, camera_name='track', mode='offscreen')
        frame = np.flip(frame, axis=0)
        frame = torch.from_numpy(frame.copy())
        frame = frame.to(torch.uint8)
        return frame

    def close(self):
        self._env.close()

    def score(self, returns) -> float:
        return self._env.env.get_normalized_score(returns)  # type: ignore


class AtariWrapper (OptionsModule, gym.Env):

    def __init__(self, environment: str):
        super().__init__()

        with quiet():
            import d4rl_atari as _

        self._env = gym.make(environment)

        assert isinstance(self._env.action_space, Discrete)
        assert isinstance(self._env.observation_space, Box)
        self.a = self._env.action_space.n
        self.x = self._env.observation_space.shape

    def state(self) -> torch.Tensor:
        return self._state

    def step(self, action: Union[int, torch.Tensor]) -> State:
        if isinstance(action, torch.Tensor):
            action = int(action.argmax(-1).item())
        self._state, self._reward, self._done, self._info = self._env.step(action)
        self._state = torch.from_numpy(self._state)
        self._reward = torch.tensor(self._reward, dtype=torch.float32)
        self._done = torch.tensor(self._done, dtype=torch.bool)
        return self._state, self._reward, self._done

    def sample_action(self) -> torch.Tensor:
        action = self._env.action_space.sample()
        return action

    def reset(self, initial: Optional[torch.Tensor] = None) -> torch.Tensor:
        if initial is not None: pass
        state = self._env.reset()
        state = torch.from_numpy(self._state)
        self._state = state
        return self._state

    def close(self):
        self._env.close()

    def score(self, _) -> float:
        return 0.0
