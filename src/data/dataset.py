import os
import random
from typing import Iterable, Literal, Optional

from rich.progress import track
import torch

from ..dot import Dot
from ..cli import console
from ..data.d4rl import make_d4rl_dataset
from ..options import OptionsModule
from ..renderables import section, check


class OfflineDataset (OptionsModule):

    debug: bool = False
    device: Literal['cuda', 'cpu'] = 'cuda'
    reload: bool = False

    stats: dict

    environment: Optional[str] = None
    discount: float = 0.99
    terminal_penalty: float = -10.0
    max_length: int = 1000

    capacity: Optional[int] = None

    def _build(self):
        if self.environment is not None:
            section('Building', module='Dataset', color='green')
        return super()._build()

    def build(self):

        self._episodes: list[torch.Tensor] = []
        self._lengths = []
        self.data = {key: torch.empty(0) for key in ['X', 'A', 'R', 'T', 'V']}
        self.stats = {key: torch.empty(0) for key in ['X', 'A', 'R', 'T', 'V']}

        if self.environment is not None:
            self.load(self.environment)

    def save(self, data: dict, env: str):
        """
        Saves compressed version of D4RL dataset to 'datasets/'
        """
        path = os.path.join('datasets', env)
        file = os.path.join(path, 'data.pt')
        if not os.path.exists(path):
            os.makedirs(path)
        torch.save(data, file, pickle_protocol=5)
        check(f'Saved Dataset to {file}', color='green')

    def load(self, env: str):
        """
        Attempts to load compressed dataset from 'datasets/'. If no dataset
        exist, generates it.
        """

        save = False

        # -------------------- Load Data -------------------- #

        # Load raw data
        path = os.path.join('datasets', env, 'data.pt')
        if self.reload or not os.path.exists(path):
            raw = make_d4rl_dataset(env)
            check(f'Creating Dataset', color='green')
        else:
            raw = torch.load(path)
            check(f'Loading Dataset', color='green')

        X = raw['observations']
        N = raw['next_observations']
        A = raw['actions']
        if len(A.shape) == 1:
            A = A[:, None]
        R = raw['rewards'][:, None]

        # Episode delimiters
        D = raw['terminals'][:, None]
        if 'timeouts' in raw:
            D += raw['timeouts'][:, None]

        assert len(X) == len(A) == len(R) == len(D)

        # -------------------- Make Indices -------------------- #

        if 'episodes' in raw and not self.reload:
            self._episodes = raw['episodes']
        else:
            self._episodes = raw['episodes'] = self.split(D)
            save = True

        self._lengths = [len(e) for e in self._episodes]

        # -------------------- Compute Values -------------------- #

        if 'values' in raw and self.discount in raw['values'] and not self.reload:
            V = raw['values'][self.discount]
        else:
            V = self.compute_values(R, self.discount)
            raw['values'] = {self.discount: V}
            save = True

        # -------------------- Collect Data -------------------- #

        self.data = {'X': X, 'N': N, 'A': A, 'R': R, 'T': D, 'V': V}
        if 'infos/qpos' in raw:
            self.data['QP'] = raw['infos/qpos']
        if 'infos/qvel' in raw:
            self.data['QV'] = raw['infos/qvel']

        # -------------------- Compute Stats -------------------- #

        if 'stats' in raw and not self.reload:
            self.stats = raw['stats']
        else:
            for key, val in self._track(self.data.items(), description='Stats'):
                if isinstance(val, torch.Tensor):
                    self.stats[key] = self.compute_stats(val)
            raw['stats'] = self.stats
            save = True

        if save: self.save(raw, env)

    def split(self, terminals: torch.Tensor):
        """
        Generates indices of raw dataset.
        Inputs:
            terminals:  [ size, 1 ]
        Outputs:
            episodes:   list[list[int]]
        """

        # Nonzero indices of terminals
        terminals[-1] = True
        ends = terminals.squeeze().nonzero()

        # Compute lengths of each episode
        ends[1:] = ends[1:] - ends[:-1]
        ends[0] = ends[0] + 1

        # Generate list of indices split by each episode
        episodes = list(torch.arange(len(terminals)).split(tuple(ends)))

        return episodes

    def _track(self, iterable: Iterable, description: str = ''):
        if self.debug:
            return iterable
        else:
            return track(iterable, transient=True,
                         description=f' [blue][reset]   Computing {description}...')

    def __len__(self):
        return len(self._episodes)

    def compute_values(self, R: torch.Tensor, discount: Optional[float] = None):
        """
        Computes state value function for entire dataset using rewards R.
        Inputs:
            R:     [ size, 1 ]
        Outputs:
            values:     [ size, 1 ]
        """

        discount = discount or self.discount

        L, N = max(self._lengths), len(self._lengths)
        x = R.split(self._lengths)
        rewards = torch.zeros(len(x), L)
        for i, episode in enumerate(x):
            rewards[i, 0:len(episode)] = episode.squeeze()

        discounts = discount ** (torch.arange(0, L))
        values = torch.zeros_like(rewards)
        for t in self._track(range(L), description='Values'):
            V = (rewards[:, t + 1:] * discounts[:-(t + 1)]).sum(dim=1)
            values[:, t] = V
        values[:, -1] = values[:, -2]

        values = torch.cat([values[i, :self._lengths[i]] for i in range(N)])
        values = values[:, None]

        assert len(values) == sum(self._lengths)

        return values

    def compute_stats(self, x: torch.Tensor):
        """
        Compiles various information about the input data x.
        Inputs:
            data:      [ *, size ]
        Outputs:
            high, low: [ size ]
            mean, var: [ size ]
            shape:     ( *, size )
            type:      torch.dtype
        """

        # Collect shape
        shape = x.shape
        type = x.dtype
        x = x.flatten(0, -2)  # [ *, size ] -> [ batch, size ]

        # Compute Ranges
        (low, _), (high, _) = x.min(0), x.max(0)

        # Compute Moments
        if x.is_floating_point():
            mean, var = x.mean(0), x.var(0)
        else:
            mean, var = None, None

        return {'high': high, 'low': low,
                'mean': mean, 'var': var,
                'shape': shape, 'type': type}

    def normalize(self, x: torch.Tensor, key: Optional[str] = None):
        """
        Normalizes each dimension of input tensor x to [-1, 1] using min and
        max values.
        Inputs / Outputs:
            x:  [ *, size ]
        """
        if key is not None and key in self.stats:
            high, low = self.stats[key]['high'], self.stats[key]['low']
            normalized = (x.flatten(0, -2) - low.to(x.device)) / (high.to(x.device) - low.to(x.device))
            normalized = 2 * normalized - 1
            normalized = normalized.reshape(x.shape).to(torch.float32)
            return normalized
        else:
            return x

    def unnormalize(self, x: torch.Tensor, key: Optional[str] = None):
        """
        Unnormalizes each dimension of input tensor x from [-1, 1] using min
        and max values.
        Inputs / Outputs:
            x:  [ *, size ]
        """
        if key is not None and key in self.stats:
            high, low = self.stats[key]['high'], self.stats[key]['low']
            unnormalized = (x.flatten(0, -2) + 1) / 2.
            unnormalized = unnormalized * (high.to(x.device) - low.to(x.device)) + low.to(x.device)
            unnormalized = unnormalized.reshape(x.shape).to(torch.float32)
            return unnormalized
        else:
            return x

    def sample_sequence(self, batch_size: int = 1, seq_len: int = 1) -> Dot:
        """
        Sample 'batch_size' continuous sequences of data of length given by 'seq_len'.
        """

        def slice_sequence(episode: torch.Tensor):
            valid_max = min(self.max_length, len(episode)) - seq_len
            start = random.randint(0, valid_max)
            episode = episode[start:(start + seq_len)]
            sequence = {key: data[episode] for key, data in self.data.items()}
            sequence['timesteps'] = torch.arange(start, start + seq_len)[:, None]
            return sequence

        batch_size = min(batch_size, len(self))

        # Select batch_size from episodes that are at least seq_len long
        episodes = [episode for episode in self._episodes if len(episode) >= seq_len]
        episodes = random.sample(episodes, batch_size)

        # Randomly cut episodes down to seq_len and select data
        batches = [slice_sequence(episode) for episode in episodes]
        batches = {key: torch.stack([batch[key] for batch in batches]) for key in [*self.data.keys(), 'timesteps']}
        batches = {key: data.to(torch.float32) if data.is_floating_point() else data for key, data in batches.items()}

        return Dot(batches)

    def sample_step(self, batch_size: int = 1) -> Dot:
        """
        Sample 'batch_size' single-timestep data.
        """

        index = torch.randint(0, len(self.data['X']), size=(batch_size,))
        sample = {key: self.data[key][index] for key in ['X', 'A', 'N', 'R', 'T', 'V']}

        return Dot(sample)
