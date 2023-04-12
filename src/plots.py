import os
from typing import Optional, cast

from einops import rearrange
import matplotlib.pyplot as plt
import torch
import wandb

from mpl_toolkits.axes_grid1 import ImageGrid

from .cli import console
from .options import OptionsModule
from .trainer import Trainer
from .util import RedirectStream

plt.switch_backend('Agg')


class Plots (OptionsModule):

    def build(self):

        from .trainer import CurrentTrainer
        if CurrentTrainer is not None:
            self.log = CurrentTrainer.log
            self.progress = CurrentTrainer.progress
            self.dir = os.path.join(CurrentTrainer.dir, 'plots')
            if not os.path.exists(self.dir): os.makedirs(self.dir)
        else:
            raise Exception('No Current Trainer')

    def save(self, fig, name: str, step: Optional[int] = None, **_):
        step = step if step is not None else self.progress.get('session')
        if self.log:
            try:
                wandb.log({name: fig}, step=step)
            except:
                pass
        name += f'_{step}.png'
        file = os.path.join(self.dir, name)
        fig.savefig(file, bbox_inches='tight', dpi=300)
        plt.close(fig)

    def _img(self, ax: plt.Axes, img: torch.Tensor, title: str = '', **kwargs):
        ax.imshow(img, cmap='viridis', **kwargs)
        ax.invert_yaxis()
        ax.set_title(title)
        ax.set_xlabel('Time Step (t)')
        ax.set_ylabel('Dimension')
        ax.grid(which='minor')
        return ax

    def _img_prep(self, img: torch.Tensor):
        img = img.cpu()

        if len(img.shape) == 3:
            img = rearrange(img, 'c h w -> h w c')
            img = img.flip(dims=(0, ))
            img = img.clamp(0.0, 1.0)

        elif len(img.shape) == 2:
            img = rearrange(img, 'h w -> w h')

        return img

    def _grid(self, n_rows: int, n_cols: int, **kwargs):

        with RedirectStream():
            fig = plt.figure(figsize=(20, 18))
            defaults = {'axes_pad': 0.2}
            defaults.update(kwargs)
            grid = ImageGrid(fig, 111, nrows_ncols=(n_rows, n_cols), **defaults)

        return fig, grid

    @torch.no_grad()
    def img(self, sample: torch.Tensor, name: str = 'img', **kwargs):
        """
        Plots a single image.
        """

        with RedirectStream():
            fig = plt.figure(figsize=(18, 18))
            axis = fig.subplots()
            axis = cast(plt.Axes, axis)

        self._img(axis, self._img_prep(sample))

        return self.save(fig=fig, name=name, **kwargs)
