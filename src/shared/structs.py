import torch

from ..cli import console


class OnlineResults:

    current: dict = {}
    history: list[dict] = [{}]

    def set_current(self, env: int, current: dict):
        self.current[env] = current

    def get_current(self):
        return self.current

    def reset_current(self):
        self.current = {}

    def set_complete(self, env: int):
        self.history[-1][env] = self.current.pop(env)

    def get_history(self):
        return [run for run in self.history if run.get('complete', False)]

    def reset_history(self):

        # Compute stats for completed episodes
        scores = torch.tensor([e['score'] for e in self.history[-1].values()])
        mean = scores.mean().item()
        std = scores.std().nan_to_num().item()
        self.history[-1]['mean'], self.history[-1]['std'] = mean, std
        self.history[-1]['complete'] = True

        self.history += [{}]
