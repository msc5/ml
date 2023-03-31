from ..util import RedirectStream

import gym
from gym import logger
import torch

logger.set_level(100)


def make_d4rl_dataset(name) -> dict:

    # with quiet():

    import d4rl as _
    import d4rl_atari as _

    env = gym.make(name)
    dataset = env.get_dataset()  # type: ignore

    keys = ['observations',
            'next_observations',
            'actions',
            'rewards',
            'terminals',
            'timeouts',
            'infos/qpos',
            'infos/qvel']
    data = {key: torch.from_numpy(dataset[key]) for key in keys if key in dataset}

    return data
