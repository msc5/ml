from typing import DefaultDict

import torch
from torch.utils.data import IterableDataset
from torch.utils.data import DataLoader

import src.ml as ml


class ReplayBuffer (IterableDataset, ml.Module):
    """ 
    Implements a continuous Replay Buffer. Stores transitions from each agent
    step into a tensor and randomly loads them in sequences.
    """

    capacity: int = 5000

    discount: float = 0.99

    def build(self):
        self.buffer = DefaultDict(list)
        self.lengths = DefaultDict(int)
        self.data = {'X': None,
                     'A': None,
                     'R': torch.empty((self.capacity, 1)),
                     'V': torch.empty((self.capacity, 1)),
                     'D': torch.empty((self.capacity, 1), dtype=torch.bool)}
        self.i, self.full, self.empty = 0, False, True

    def __len__(self):
        if self.full:
            return self.capacity
        else:
            return self.i

    def loader(self, batch_size: int = 1, **kwargs):
        return DataLoader(self, batch_size=batch_size, drop_last=True, **kwargs)

    def push(self,
             observation: torch.Tensor,
             action: torch.Tensor,
             reward: float,
             done: bool):

        if self.data['X'] is None or self.data['A'] is None:
            self.data['X'] = torch.empty((self.capacity, *observation.shape))
            self.data['A'] = torch.empty((self.capacity, *action.shape))

        with torch.no_grad():
            self.data['X'][self.i] = observation.cpu()
            self.data['A'][self.i] = action.cpu()
            self.data['R'][self.i] = reward
            self.data['D'][self.i] = done
            if self.i > 0:
                self.data['V'][self.i] = reward + self.discount * self.data['V'][self.i - 1]
            else:
                self.data['V'][self.i] = reward

        self.i = (self.i + 1) % self.capacity
        self.full = self.full or self.i == 0

    def sample(self, batch_size: int = 1, seq_len: int = 10):

        if self.data['X'] is None or self.data['A'] is None:
            raise Exception('Buffer Empty')

        #  TODO: sample only from a contiguous episode
        indices = (torch.arange(0, seq_len) % self.i)[None].repeat(batch_size, 1)
        shift = torch.randint(0, len(self) - seq_len, (batch_size, ))[:, None]
        indices += shift

        with torch.no_grad():
            sample = {key: data[indices] for key, data in self.data.items()}

        return sample
