import math
import os
import random
from typing import Any, Callable, Iterable, Literal, Optional, Union, cast

from rich import box
from rich.console import group
from rich.progress import track
from rich.table import Table
import torch

from ..options import Dot

from ... import ml
from ..cli import console
from ..data.d4rl import make_d4rl_dataset
from ..options import OptionsModule
from ..renderables import section, check


class OfflineDataset (OptionsModule):

    debug: bool = False
    device: Literal['cuda', 'cpu'] = 'cuda'
    reload: bool = False

    stats: dict

    env: Optional[str] = None
    discount: float = 0.99
    terminal_penalty: float = -10.0
    max_length: int = 1000

    capacity: Optional[int] = None

    def _build(self):
        if self.env is not None:
            section('Building', module='Dataset', color='green')
        return super()._build()

    def build(self):

        self.metrics = Dot({'i': 0, 'n': 0})
        self._episodes: list[torch.Tensor] = []
        self._lengths = []
        self.data = {key: torch.empty(0) for key in ['X', 'A', 'R', 'T', 'V']}
        self.stats = {key: torch.empty(0) for key in ['X', 'A', 'R', 'T', 'V']}

        if self.env is not None:
            self.load(self.env)

    def save(self, data: dict, env: str):
        path = os.path.join('datasets', env)
        file = os.path.join(path, 'data.pt')
        if not os.path.exists(path):
            os.makedirs(path)
        torch.save(data, file, pickle_protocol=5)
        check(f'Saved Dataset to {file}', color='green')

    def load(self, env: str):

        save = False

        # Load raw data
        path = os.path.join('datasets', env, 'data.pt')
        if self.reload or not os.path.exists(path):
            raw = make_d4rl_dataset(env)
            check('Created Dataset', color='green')
        else:
            raw = torch.load(path)
            check(f'Loaded Dataset from {path}', color='green')

        X = raw['observations']
        N = raw['next_observations']
        A = raw['actions']
        if len(A.shape) == 1:
            A = A[:, None]
        R = raw['rewards'][:, None]
        D = raw['terminals'][:, None]
        # if 'timeouts' in raw:
        #     D += raw['timeouts'][:, None]
        # D[-1] = True
        # R[D] = self.terminal_penalty

        assert len(X) == len(A) == len(R) == len(D)

        if 'episodes' in raw and not self.reload:
            self._episodes = raw['episodes']
            check('Loaded Dataset Splits', color='green')
        else:
            raw['episodes'] = self.split(D)
            save = True
            check('Computed Dataset Splits', color='green')

        self.metrics = Dot({'i': len(X), 'n': len(self._episodes)})

        self._lengths = [len(e) for e in self._episodes]
        self.x_size = X.shape[-1]
        self.a_size = A.shape[-1]

        if 'values' in raw and self.discount in raw['values'] and not self.reload:
            V = raw['values'][self.discount]
            check('Loaded Values', color='green')
        else:
            V = self.values(R, self.discount)
            raw['values'] = {self.discount: V}
            save = True
            check('Computed Values', color='green')

        self.data = {'X': X, 'N': N, 'A': A, 'R': R, 'T': D, 'V': V}
        if 'infos/qpos' in raw:
            self.data['QP'] = raw['infos/qpos']
        if 'infos/qvel' in raw:
            self.data['QV'] = raw['infos/qvel']

        if 'stats' in raw and not self.reload:
            self.stats = raw['stats']
            check('Loaded Stats', color='green')
        else:
            for key, val in self.track(self.data.items(), description='Stats'):
                if isinstance(val, torch.Tensor):
                    self.stats[key] = self.stat(val)
            raw['stats'] = self.stats
            save = True
            check('Computed Stats', color='green')

        if save: self.save(raw, env)
        check('Finished', color='green')

    def split(self, terminals: torch.Tensor):

        # Nonzero indices of terminals
        terminals[-1] = True
        ends = terminals.squeeze().nonzero()

        # Compute lengths of each episode
        ends[1:] = ends[1:] - ends[:-1]
        ends[0] = ends[0] + 1

        # Generate list of indices split by each episode
        self._episodes = list(torch.arange(len(terminals)).split(tuple(ends)))

        return self._episodes

    def track(self, iterable: Iterable, description: str = ''):
        if self.debug:
            return iterable
        else:
            return track(iterable, transient=True,
                         description=f' [blue][reset]   Computing {description}...')

    def __len__(self):
        return len(self._episodes)

    def values(self, R: torch.Tensor, discount: float):
        """
        Computes state value function for entire dataset using rewards R 
        and terminal states D.
        Inputs:
            R: [ batch, 1 ]
            D: [ batch, 1 ]
        Outputs:
            V: [ batch, 1 ]
        """

        L, N = max(self._lengths), len(self._lengths)
        x = R.split(self._lengths)
        # breakpoint()
        # rewards = torch.stack([F.pad(r.squeeze(), (0, L - len(r))) for r in x])
        rewards = torch.zeros(len(x), L)
        for i, episode in enumerate(x):
            rewards[i, 0:len(episode)] = episode.squeeze()

        discounts = discount ** (torch.arange(0, L))
        values = torch.zeros_like(rewards)
        for t in self.track(range(L), description='Values'):
            V = (rewards[:, t + 1:] * discounts[:-(t + 1)]).sum(dim=1)
            # V = (1 / (L - t)) * (rewards[:, t:]).sum(dim=1)
            # gamma = discounts[:-(t + 1)]
            # V = (1 / (gamma.sum() + 1e-30)) * (gamma * rewards[:, t + 1:]).sum(dim=1)
            values[:, t] = V
        values[:, -1] = values[:, -2]

        values = torch.cat([values[i, :self._lengths[i]] for i in range(N)])
        values = values[:, None]

        assert len(values) == sum(self._lengths)

        return values

    def stat(self,
             x: torch.Tensor,
             discretization: Literal['quantile', 'uniform'] = 'quantile',
             vocab_size: int = 100):
        """
        Compiles various information about the input data x.
        Inputs:
            data:      [ *, size ]
        Outputs:
            threshold: [ size, vocab_size ]
            high, low: [ size ]
            mean, var: [ size ]
            ranges:    ( min, mean, max )
            shape:     ( *, size )
            size:      int
            type:      torch.dtype
        """

        # Collect shape
        shape = x.shape
        size = shape[-1]
        type = x.dtype
        x = x.flatten(0, -2)  # [ *, size ] -> [ batch, size ]

        # Compute Ranges
        (low, _), (high, _) = x.min(0), x.max(0)
        ranges = ml.ranges(x)

        # Compute Moments
        if x.is_floating_point():
            mean, var = x.mean(0), x.var(0)
        else:
            mean, var = None, None

        # Compute Thresholds
        if x.is_floating_point():
            if discretization == 'quantile':
                n = math.ceil(len(x) / vocab_size)
                sorted = x.sort(0).values
                threshold = (sorted[::n]).T
            elif discretization == 'uniform':
                stops = torch.linspace(0, 1, vocab_size, device=self.device)[None]
                threshold = torch.lerp(low, high, stops)
            else:
                raise Exception()
            threshold = torch.cat([threshold, high[:, None]], dim=-1)
        else:
            threshold = None

        return {'threshold': threshold,
                'high': high, 'low': low,
                'mean': mean, 'var': var,
                'shape': shape, 'size': size,
                'type': type, 'ranges': ranges}

    def discretize(self,
                   x: torch.Tensor,
                   threshold: Union[str, torch.Tensor],
                   **_):
        """
        Discretizes each dimension of input tensor x into vocab_size buckets.
        Inputs:
            x:         [ *, size ]
            threshold: [ size, vocab_size ]
        Outputs:
            x:         [ *, size ]
        """
        if threshold is None or not x.is_floating_point():
            return x
        if isinstance(threshold, str):
            threshold = cast(torch.Tensor, self.stats[threshold]['threshold'])
        size = threshold.shape[0]
        threshold = threshold.to(x.device)
        flat = x.view(-1, size)[:, :, None]
        discrete = (flat > threshold[None]).sum(-1) - 1
        discrete = discrete.reshape(x.shape)
        discrete = discrete.clamp(0, threshold.shape[-1] - 1)
        return discrete

    def undiscretize(self,
                     x: torch.Tensor,
                     threshold: Union[str, torch.Tensor],
                     vocab_size: int = 100,
                     **_):
        """
        Returns each dimension of discretized input tensor x to the midpoint
        of original thresholds.
        Inputs:
            x:         [ *, size ]
            threshold: [ size, vocab_size ]
        Outputs:
            x:         [ *, size ]
        """
        if threshold is None or x.is_floating_point():
            return x
        if isinstance(threshold, str):
            threshold = cast(torch.Tensor, self.stats[threshold]['threshold'])
        size = threshold.shape[0]
        threshold = threshold.to(x.device).transpose(0, 1)
        flat = x.to(torch.long).clamp(0, vocab_size - 1).view(-1, size)
        high, low = threshold.gather(0, flat), threshold.gather(0, flat + 1)
        undiscretized = (high + low) / 2
        undiscretized = undiscretized.view(x.shape)
        return undiscretized

    def test_discretize(self, target: torch.Tensor, threshold: Union[str, torch.Tensor], **_):
        if threshold is None or not target.is_floating_point():
            return {'discrete': target, 'error': 0}
        else:
            discrete = self.discretize(target, threshold)
            undiscrete = self.undiscretize(discrete, threshold)
            error = ((undiscrete - target).abs()).mean().item()
            return {'error': error,
                    'threshold': threshold,
                    'target': target,
                    'discrete': discrete,
                    'undiscrete': undiscrete}

    def one_hot(self, x: torch.Tensor, vocab_size: int = 100, **_):
        """
        Turns x into a one hot encoding of the discrete value.
        Inputs:
            x:     [ *, size ]
        Outputs:
            x:     [ *, size, vocab_size ]
        """
        assert isinstance(vocab_size, int)
        one_hot = torch.zeros(*x.shape, vocab_size, device=x.device, dtype=torch.bool)
        one_hot = one_hot.scatter(-1, x.clamp(0, vocab_size - 1).unsqueeze(-1), 1)
        return one_hot

    def normalize(self, x: torch.Tensor, key: Optional[str] = None):
        """
        Normalizes each dimension of input tensor x to [-1, 1] using min and max values.
        Inputs / Outputs:
            x:         [ *, size ]
        """
        if key is not None:
            high, low = self.stats[key]['high'], self.stats[key]['low']
            normalized = (x.flatten(0, -2) - low.to(x.device)) / (high.to(x.device) - low.to(x.device))
            normalized = 2 * normalized - 1
            normalized = normalized.reshape(x.shape).to(torch.float32)
            return normalized
        else:
            return x

    def unnormalize(self, x: torch.Tensor, key: Optional[str] = None):
        """
        Unnormalizes each dimension of input tensor x to [-1, 1] using min and max values.
        Inputs / Outputs:
            x:         [ *, size ]
        """
        if key is not None:
            high, low = self.stats[key]['high'], self.stats[key]['low']
            unnormalized = (x.flatten(0, -2) + 1) / 2.
            unnormalized = unnormalized * (high.to(x.device) - low.to(x.device)) + low.to(x.device)
            unnormalized = unnormalized.reshape(x.shape).to(torch.float32)
            return unnormalized
        else:
            return x

    def test_normalize(self, target: torch.Tensor, key: Optional[str] = None):
        normalized = self.normalize(target, key)
        unnormalized = self.unnormalize(normalized, key)
        error = ((unnormalized - target).abs()).mean()
        return {'error': error,
                'target': target,
                'normalized': normalized,
                'unnormalized': unnormalized}

    def standardize(self, x: torch.Tensor, mean: torch.Tensor, var: torch.Tensor, **_):
        """
        Standardizes each dimension of input tensor x using mean and variance.
        Inputs:
            x:         [ *, size ]
            mean, var: [ size ]
        Outputs:
            x:         [ *, size ]
        """
        if mean is not None and var is not None:
            standardized = (x.flatten(0, -2) - mean.to(x.device)[None]) / var.to(x.device)[None]
            standardized = standardized.reshape(x.shape)
            standardized = standardized.to(torch.float32)
            return standardized
        else:
            return x

    def unstandardize(self, x: torch.Tensor, mean: torch.Tensor, var: torch.Tensor, **_):
        """
        Unstandardizes each dimension of input tensor x using mean and variance.
        Inputs:
            x:         [ *, size ]
            mean, var: [ size ]
        Outputs:
            x:         [ *, size ]
        """
        if mean is not None and var is not None:
            unstandardized = (x.flatten(0, -2) * var[None].to(x.device)) + mean.to(x.device)[None]
            unstandardized = unstandardized.reshape(x.shape)
            return unstandardized
        else:
            return x

    def map(self, batch: dict[str, torch.Tensor], function: Callable):
        return {key: function(data, key) if (key != 'T' and key in self.data.keys()) else data
                for key, data in batch.items()}

    # def split(self, x: torch.Tensor, keys: list[str]):
    #     """
    #     Splits concatenated tensor into dataset keys.
    #     Inputs:
    #         x:     [ *, {a, b, c, ...} ]
    #         keys:  [ a, b, c, ...]
    #     Outputs:
    #         batch: { a: [ *, size ], ...}
    #     """
    #     dim = -1 if len(x.shape) > 1 else 0
    #     sizes = [self.data[key].shape[-1] for key in keys]
    #     splits = x.split(sizes, dim=dim)
    #     split = {key: split for key, split in zip(keys, splits)}
    #     return split

    def cat(self, x: dict[str, torch.Tensor], keys: list[str]):
        return torch.cat([x[key] for key in keys], dim=-1)

    def sequence(self, episode: torch.Tensor, seq_len: int, start: Optional[int] = None):
        if start is None:
            start = random.randint(0, min(self.max_length, len(episode)) - seq_len)
        episode = episode[start:(start + seq_len)]
        sequence = {key: data[episode] for key, data in self.data.items()}
        sequence['timesteps'] = torch.arange(start, start + seq_len)[:, None]
        sequence['samples'] = episode
        return sequence

    def sample(self,
               seq_len: int,
               episodes: Union[int, list[int]] = 1,
               starts: Optional[list[int]] = None):

        if episodes == []: episodes = 1
        if isinstance(episodes, int):
            episodes = min(episodes, len(self))
            candidates = [i for i in range(len(self)) if len(self._episodes[i]) >= seq_len]
            _episodes = random.sample(candidates, episodes)
            _episodes = [self._episodes[e] for e in _episodes]
        else:
            _episodes = [self._episodes[s] for s in episodes]

        if starts is not None and starts != []:
            batches = [self.sequence(episode, seq_len, start) for episode, start in zip(_episodes, starts)]
        else:
            batches = [self.sequence(episode, seq_len) for episode in _episodes]

        batches = {key: torch.stack([batch[key] for batch in batches])
                   for key in [*self.data.keys(), 'timesteps', 'samples']}
        batches = {key: data.to(torch.float32)
                   if data.is_floating_point() else data
                   for key, data in batches.items()}
        # batches['samples'] = torch.cat(_episodes)

        return batches

    def sample_step(self, batch_size: int):
        ind = torch.randint(0, len(self.data['X']), size=(batch_size,))
        sample = {key: self.data[key][ind] for key in ['X', 'A', 'N', 'R', 'T', 'V']}
        return sample

    def start(self):
        episode = []
        self.metrics.n += 1

        def push(data: dict[str, Any]):
            for key, val in data.items():
                if key in self.data:
                    if not isinstance(val, torch.Tensor): val = torch.tensor(val)
                    if len(self.data[key]) == 0:
                        self.data[key] = val[None]
                    else:
                        self.data[key] = torch.cat([self.data[key], val[None]])
            episode.append(self.metrics.i)
            self.metrics.i += 1

        def stop():
            self._episodes.append(torch.tensor(episode))

        return push, stop

    @group()
    def table(self):

        title = f'Dataset'
        if self.env is not None: title += f' [reset]([yellow]{self.env}[reset])'
        table = Table(title=title, show_header=False, expand=True, box=box.ROUNDED, style='blue')
        table.add_row('Episodes', str(len(self._episodes)))
        table.add_row('Lengths', str(ml.ranges(torch.tensor(self._lengths))))

        data = Table('Key', 'Size', 'Type', 'Min', 'Mean', 'Max',
                     expand=True, box=box.ROUNDED, style='blue')
        for key, stat in self.stats.items():
            low, mean, high = stat['ranges']
            data.add_row(key, str(stat['size']), str(stat['type']),
                         str(low), str(mean), str(high))
        table.add_row('Data', data)

        yield table

    def __rich__(self):
        return self.table()
