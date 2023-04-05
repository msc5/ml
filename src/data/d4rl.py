import gym
from gym import logger
import torch

from ..util import RedirectStream

logger.set_level(100)


def make_d4rl_dataset(name) -> dict:

    with RedirectStream():
        import d4rl as _
        # import d4rl_atari as _

        env = gym.make(name)
        dataset = env.get_dataset()  # type: ignore

    # Map original keys to shorter keys
    keys = {
        'observations': 'X',
        'next_observations': 'N',
        'actions': 'A',
        'rewards': 'R',
        'terminals': 'TM',
        'timeouts': 'TO',
        'infos/qpos': 'QP',
        'infos/qvel': 'QV',
    }

    data = {data_key: torch.from_numpy(dataset[raw_key])
            for raw_key, data_key in keys.items() if raw_key in dataset}

    for key, val in data.items():
        if isinstance(val, torch.Tensor) and len(val.shape) == 1:
            data[key] = val.unsqueeze(-1)

    return data
