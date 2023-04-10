from contextlib import nullcontext
import os
import random
from typing import Iterable, Literal, Optional
import gym

from rich.progress import track
import torch
import numpy as np
from torchvision.transforms import Resize
from torchvision.transforms.functional import crop

from ..dot import Dot
from ..cli import console
from ..data.d4rl import make_d4rl_dataset
from ..options import OptionsModule
from ..util import Metadata, RedirectStream


class OfflineDataset (OptionsModule):

    debug: bool = False
    profile: bool = False
    device: Literal['cuda', 'cpu'] = 'cuda'
    reload: bool = False

    environment: str = ''
    discount: float = 0.99
    terminal_penalty: float = -100.0
    max_length: int = 1000
    frame_size: int = 64
    use_video: bool = False

    capacity: Optional[int] = None

    # Required
    X: torch.Tensor
    N: torch.Tensor
    A: torch.Tensor
    R: torch.Tensor
    T: torch.Tensor
    E: torch.Tensor

    # Optional
    QP: torch.Tensor
    QV: torch.Tensor

    # Generated
    F: torch.Tensor
    indices: list[torch.Tensor]
    stats: dict[str, dict[str, torch.Tensor]]
    meta: dict

    def _track(self, iterable: Iterable, description: str = ''):
        if self.debug:
            return iterable
        else:
            return track(iterable, transient=True,
                         description=f' [blue][reset]  {description}...')

    def __len__(self):
        return len(self.indices)

    def build(self):

        self.keys = ['X', 'N', 'A', 'R', 'T', 'E', 'QP', 'QV']

        self.dir = os.path.join('datasets', self.environment)
        if not os.path.exists(self.dir):
            os.makedirs(self.dir)

        if self.environment != '':
            self.load()

    def items(self):
        for key in self.keys:
            yield (key, getattr(self, key, None))

    def save(self):
        """
        Saves compressed version of D4RL dataset to 'datasets/'
        """

        Metadata.save(self.dir, self.meta)
        for key, val in self.items():
            file = os.path.join(self.dir, f'{key}.pt')
            torch.save(val, file, pickle_protocol=5)

        for key in ('stats', 'indices', 'F'):
            file = os.path.join(self.dir, f'{key}.pt')
            if (val := getattr(self, key, None)) is not None:
                torch.save(val, file, pickle_protocol=5)

    def load(self):
        """
        Attempts to load compressed dataset from 'datasets/'. If no dataset
        exist, generates it.
        """

        # # Reload
        # reloaded = False
        # if self.reload or not os.path.exists(self.dir):

        # Load from D4RL
        self.meta = {}
        d4rl = make_d4rl_dataset(self.environment)
        for key, val in d4rl.items():
            setattr(self, key, val)

        # reloaded = True

        # # Load from saved
        # else:
        #     self.meta = Metadata.load(self.dir)
        #     for key, _ in self.items():
        #         file = os.path.join(self.dir, f'{key}.pt')
        #         setattr(self, key, torch.load(file))

        self.load_splits()
        self.load_values()
        self.load_video()
        self.load_stats()

        # if reloaded:
        #     self.save()

    def load_splits(self):
        """
        Generates indices of raw dataset.
        Inputs:
            terminals:  [ size, 1 ]
        Outputs:
            episodes:   list[list[int]]
        """

        path = os.path.join(self.dir, 'indices.pt')

        if os.path.exists(path):
            self.indices = torch.load(path)

        else:

            # Episode delimiters
            terminals = self.T[:, None]
            terminals += self.E[:, None]

            # Set termination penalties
            penalty_timesteps = self.T & ~self.E
            self.R[:, None][penalty_timesteps] += self.terminal_penalty

            # Nonzero indices of terminals
            terminals[-1] = True
            ends = terminals.squeeze().nonzero()

            # Compute lengths of each episode
            ends[1:] = ends[1:] - ends[:-1]
            ends[0] = ends[0] + 1

            # Generate list of indices split by each episode
            episodes = list(torch.arange(len(terminals)).split(tuple(ends)))

            self.indices = episodes

        self.meta['lengths'] = [len(e) for e in self.indices]

    def load_values(self):
        """
        Computes state value function for entire dataset using rewards R.
        Inputs:
            R:     [ size, 1 ]
        Outputs:
            values:     [ size, 1 ]
        """

        path = os.path.join(self.dir, 'V.pt')

        # Load values
        if os.path.exists(path):
            self.V = torch.load(path)

        # Compute Values
        else:

            L, N = max(self.meta['lengths']), len(self.meta['lengths'])
            x = self.R.split(self.meta['lengths'])

            # Pad with zeros
            rewards = torch.zeros(len(x), L)
            for i, episode in enumerate(x):
                rewards[i, 0:len(episode)] = episode.squeeze()

            # Compute values
            discounts = self.discount ** (torch.arange(0, L))
            values = torch.zeros_like(rewards)
            for t in self._track(range(L), description='Values'):
                V = (rewards[:, t:] * discounts[:L - t]).sum(dim=1)
                values[:, t] = V

            # Re-flatten values
            values = torch.cat([values[i, :self.meta['lengths'][i]] for i in range(N)])
            values = values[:, None]
            assert len(values) == sum(self.meta['lengths'])

            self.V = values

        self.meta['discount'] = self.discount

    def load_stats(self):
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

        path = os.path.join(self.dir, 'stats.pt')

        if os.path.exists(path):
            self.stats = torch.load(path)

        else:

            stats = {}
            for key, val in self._track(self.items(), description='Computing Stats'):
                if isinstance(val, torch.Tensor):

                    # Collect shape
                    shape = val.shape
                    dtype = val.dtype
                    x = val.flatten(0, -2)  # [ *, size ] -> [ batch, size ]

                    # Compute Ranges
                    (low, _), (high, _) = x.min(0), x.max(0)

                    # Compute Moments
                    if x.is_floating_point():
                        mean, std = x.mean(0), x.std(0)
                    else:
                        mean, std = None, None

                    stats[key] = {'high': high, 'low': low,
                                  'mean': mean, 'std': std,
                                  'shape': shape, 'type': dtype}

            self.stats = stats

    def load_video(self):

        path = os.path.join(self.dir, 'F.pt')

        if not self.use_video:
            return

        elif os.path.exists(path):
            self.F = torch.load(path)

        else:

            with RedirectStream():
                env = gym.make(self.environment)

            n = len(self.QP)
            frames = None
            resize = Resize((self.frame_size, self.frame_size), antialias=True)

            for i in self._track(range(n), description='Rendering Frames'):

                qp, qv = self.QP[i], self.QV[i]
                env.set_state(qp, qv)

                height, width = 256, 256
                with RedirectStream():
                    frame = env.sim.render(height, width, camera_name='track', mode='offscreen')
                frame = np.flip(frame, axis=0)
                frame = torch.from_numpy(frame.copy())
                frame = frame.to(torch.uint8)
                frame = frame.permute(2, 0, 1)

                frame = crop(frame, top=64, left=64, width=128, height=192)
                frame = resize(frame)

                if frames is None:
                    frames = torch.zeros((n, *frame.shape), dtype=torch.uint8)

                frames[i] = frame

            assert frames is not None
            self.F = frames

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
            unnormalized = (x.clamp(-1.0, 1.0).flatten(0, -2) + 1) / 2.
            unnormalized = unnormalized * (high.to(x.device) - low.to(x.device)) + low.to(x.device)
            unnormalized = unnormalized.reshape(x.shape).to(torch.float32)
            return unnormalized
        else:
            return x

    def standardize(self, x: torch.Tensor, key: Optional[str], **_):
        """
        Standardizes each dimension of input tensor x to 0 mean and 1 std.
        Inputs / Outputs:
            x:  [ *, size ]
        """
        if key is not None and key in self.stats:
            mean, std = self.stats[key]['mean'].to(x.device), self.stats[key]['std'].to(x.device)
            standardized = (x.flatten(0, -2) - mean[None]) / std[None]
            standardized = standardized.reshape(x.shape)
            standardized = standardized.to(torch.float32)
            return standardized
        else:
            return x

    def unstandardize(self, x: torch.Tensor, key: Optional[str], **_):
        """
        Unstandardizes each dimension of input tensor x from 0 mean and 1 std.
        Inputs / Outputs:
            x:  [ *, size ]
        """
        if key is not None and key in self.stats:
            mean, std = self.stats[key]['mean'].to(x.device), self.stats[key]['std'].to(x.device)
            unstandardized = (x.flatten(0, -2) * std[None].to(x.device)) + mean.to(x.device)[None]
            unstandardized = unstandardized.reshape(x.shape)
            return unstandardized
        else:
            return x

    def sample_sequence(self, batch_size: int = 1, seq_len: int = 1) -> Dot:
        """
        Sample 'batch_size' continuous sequences of data of length given by 'seq_len'.
        """

        tensors = dict(filter(lambda x: isinstance(x[1], torch.Tensor), self.items()))

        def slice_sequence(episode: torch.Tensor):
            valid_max = min(self.max_length, len(episode)) - seq_len
            start = random.randint(0, valid_max - 1)
            episode = episode[start:(start + seq_len)]
            sequence = {key: data[episode] for key, data in tensors.items()}
            sequence['timesteps'] = torch.arange(start, start + seq_len)[:, None]
            return sequence

        batch_size = min(batch_size, len(self))

        # Select batch_size from episodes that are at least seq_len long
        episodes = [episode for episode in self.indices if len(episode) > seq_len]
        episodes = random.sample(episodes, batch_size)

        # Randomly cut episodes down to seq_len and select data
        batches = [slice_sequence(episode) for episode in episodes]
        batches = {key: torch.stack([batch[key] for batch in batches]) for key in [*tensors.keys(), 'timesteps']}
        batches = {key: data.to(torch.float32) if data.is_floating_point() else data for key, data in batches.items()}

        return Dot(batches)

    def sample_step(self, batch_size: int = 1) -> Dot:
        """
        Sample 'batch_size' single-timestep data.
        """

        index = torch.randint(0, len(self.X), size=(batch_size,))
        sample = {key: getattr(self, key)[index] for key in ['X', 'A', 'N', 'R', 'T', 'V']}

        return Dot(sample)
