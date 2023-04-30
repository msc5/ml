from typing import Optional
import torch

from ..options import OptionsModule
from ..dot import Dot


class OnlineDataset (OptionsModule):
    """
    A.K.A. "Replay Buffer"
    """

    x_size: int
    a_size: int
    frame_shape: list[int]

    use_video: bool = False

    buffer_size: int = int(1e6)

    # Data
    I: torch.Tensor
    X: torch.Tensor
    N: torch.Tensor
    A: torch.Tensor
    R: torch.Tensor
    T: torch.Tensor
    F: Optional[torch.Tensor] = None

    def build(self):

        self.I = - torch.ones((self.buffer_size, 1), dtype=torch.int64)
        self.X = torch.zeros((self.buffer_size, self.x_size), dtype=torch.float32)
        self.N = torch.zeros((self.buffer_size, self.x_size), dtype=torch.float32)
        self.A = torch.zeros((self.buffer_size, self.a_size), dtype=torch.float32)
        self.R = torch.zeros((self.buffer_size, 1), dtype=torch.float32)
        self.T = torch.zeros((self.buffer_size, 1), dtype=torch.bool)

        if self.use_video:
            self.F = torch.zeros((self.buffer_size, *self.frame_shape), dtype=torch.float32)

        self.p = 0
        self.n = 0

        self.metrics = Dot({'steps': self.n, 'terminals': 0})

    def items(self):
        yield ('I', self.I)
        yield ('X', self.X)
        yield ('N', self.N)
        yield ('A', self.A)
        yield ('R', self.R)
        yield ('T', self.T)

        if self.use_video and self.F is not None:
            yield ('F', self.F)

    def __get_item__(self, i: int):
        return {key: val[i] for key, val in self.items()}

    def __len__(self):
        return self.n

    def push(self, data: dict) -> None:
        """
        Add step to online RL dataset.
        """

        for key, val in self.items():
            if key in data:
                val[self.p % self.buffer_size] = data[key]

        self.p += 1
        self.n = max(self.n, self.p)

        self.metrics.steps = self.n
        self.metrics.terminals = self.T.sum().item()

    def sample_step(self, batch_size: int = 1) -> Dot:
        """
        Sample 'batch_size' single-timestep batches.
        """

        if len(self) < batch_size:
            raise Exception('Buffer too small to sample this batch size')

        # Sample random mini-batches
        indices = torch.randint(0, len(self) - 1, size=(batch_size, ))

        # # Reject terminal timesteps
        # terminals = (self.T[indices].squeeze()
        #              | (indices == len(self.X) - 1)
        #              | (self.I[indices] != self.I[indices + 1]).squeeze())
        # indices[terminals] -= 1

        sample = {key: field[indices] for key, field in self.items()}

        # Compute next Frame
        if self.use_video and self.F is not None:
            sample['FN'] = self.F[indices + 1]
            if not self.I[indices].equal(self.I[indices + 1]):
                raise Exception('Episodes are not contiguous!')

        return Dot(sample)
