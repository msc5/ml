import torch

from .dataset import OfflineDataset
from ..dot import Dot


class OnlineDataset (OfflineDataset):

    buffer: dict = {}

    buffer_size: int = 10000

    def __len__(self):
        if 'indices' in self.buffer:
            return len(self.buffer['indices'])
        else:
            return 0

    def push(self, episode: dict):
        """
        Add episode to online RL dataset.
        Inputs:
            episode: {
                env: {
                    'score': float, normalized score,
                    'reward': float, normalized rewards,
                    'X': list(tensor[float32]), environment states,
                    'A': list[tensor[float32]], environment actions
                }
            }
        """

        for _, result in episode.items():

            # Create tensors from episode
            X = torch.stack(result['X']).cpu()
            A = torch.stack(result['A']).cpu()
            R = torch.tensor(result['reward'] + [self.terminal_penalty])[:, None].cpu()
            TM = torch.tensor([False] * len(result['reward']) + [True])[:, None].cpu()
            TO = torch.tensor([False] * len(TM))[:, None].cpu()
            if len(R) >= self.max_length:
                TO[self.max_length - 1] = True

            length = len(X)
            indices = list(range(len(self), len(self) + length))

            # Add new episode to buffer
            if self.buffer == {}:
                self.buffer: dict = {
                        'X': X,
                        'A': A,
                        'R': R,
                        'TM': TM,
                        'TO': TO,
                        'lengths': [length],
                        'indices': [indices]
                        }
            else:
                self.buffer['X'] = torch.cat([X, self.buffer['X'][:self.buffer_size]], dim=0)
                self.buffer['A'] = torch.cat([A, self.buffer['A'][:self.buffer_size]], dim=0)
                self.buffer['R'] = torch.cat([R, self.buffer['R'][:self.buffer_size]], dim=0)
                self.buffer['TM'] = torch.cat([TM, self.buffer['TM'][:self.buffer_size]], dim=0)
                self.buffer['TO'] = torch.cat([TO, self.buffer['TO'][:self.buffer_size]], dim=0)
                self.buffer['lengths'] = [length] + self.buffer['lengths'][:self.buffer_size]
                self.buffer['indices'] = [indices] + self.buffer['indices'][:self.buffer_size]

    def sample_step(self, batch_size: int = 1) -> Dot:
        """
        Sample 'batch_size' single-timestep data.
        """

        if len(self) < batch_size:
            raise Exception('Size too small to sample this batch size')

        indices = torch.randint(0, len(self.buffer['X']) - 1, size=(batch_size,))
        sample = {key: self.buffer[key][indices] for key in ['X', 'A', 'R', 'TM']}
        sample['N'] = self.buffer['X'][indices + 1]

        return Dot(sample)
        
