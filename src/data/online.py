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

    buffer_size: int = int(1e6)

    def build(self):

        self.X = torch.zeros((self.buffer_size, self.x_size), dtype=torch.float32)
        self.N = torch.zeros((self.buffer_size, self.x_size), dtype=torch.float32)
        self.A = torch.zeros((self.buffer_size, self.a_size), dtype=torch.float32)
        self.R = torch.zeros((self.buffer_size, 1), dtype=torch.float32)
        self.T = torch.zeros((self.buffer_size, 1), dtype=torch.bool)

        self.p = 0
        self.n = 0

        self.metrics = Dot({'steps': self.n, 'terminals': 0})

    def items(self):
        yield ('X', self.X)
        yield ('N', self.N)
        yield ('A', self.A)
        yield ('R', self.R)
        yield ('T', self.T)

    def __get_item__(self, i: int):
        return {key: val[i] for key, val in self.items()}

    def __len__(self):
        return self.n

    def push(self, data: dict) -> None:
        """
        Add step to online RL dataset.
        """

        for key, val in self.items():
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

        indices = torch.randint(0, len(self) - 1, size=(batch_size, ))
        sample = {key: field[indices] for key, field in self.items()}

        return Dot(sample)
