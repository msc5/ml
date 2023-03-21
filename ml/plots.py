from typing import cast
import torch
import matplotlib.pyplot as plt
import matplotlib.animation as anim
import os

from .cli import console
from .util import quiet


def figure(*args, **kwargs) -> tuple[plt.Figure, plt.Axes]:
    with quiet():
        fig, axes = plt.subplots(*args, figsize=(10, 10), **kwargs)
    return fig, cast(plt.Axes, axes)


def gif_diff_values(values: torch.Tensor, tag: str = 'values'):

    fig, ax = figure()
    ax.set_ylim([0, 800])
    ax.grid()
    timesteps = torch.arange(values.shape[-1]).flip(0)
    (ln, ) = ax.plot(timesteps, timesteps, 'r')
    ln = cast(plt.Line2D, ln)

    def update(frame: int):
        ln.set_data(timesteps, values[frame, :])
        ax.set_title(f'Step {frame}')
        return (ln, )

    ani = anim.FuncAnimation(fig, update, frames=torch.arange(len(values)), blit=True, interval=5)

    writer = anim.FFMpegWriter(fps=30)
    dir = os.path.dirname(tag)
    if not os.path.exists(dir):
        os.makedirs(dir)
    path = f'{tag}.mp4'
    ani.save(path, writer=writer)
    plt.close(fig)

    return path


def scores_plot(runs: dict, tag: str = 'scores'):

    fig, ax = figure()
    ax.grid()
    ax.set_title('Episode Score')

    for run, data in runs.items():
        ax.plot(data['score'], label=run)
    ax.legend()
    ax.set_xlabel('Environment Timestep')
    ax.set_ylabel('Normalized Score')

    dir = os.path.dirname(tag)
    if dir and not os.path.exists(dir):
        os.makedirs(dir)
    plt.savefig(f'{tag}.png')
    plt.close(fig)


def rewards_plot(runs: dict, tag: str = 'rewards'):

    fig, ax = figure()
    ax.grid()
    ax.set_title('Episode Reward')

    for run, data in runs.items():
        ax.plot(data['reward'], label=run)
    ax.legend()
    ax.set_xlabel('Environment Timestep')
    ax.set_ylabel('Reward')

    dir = os.path.dirname(tag)
    if dir and not os.path.exists(dir):
        os.makedirs(dir)
    plt.savefig(f'{tag}.png')
    plt.close(fig)


class Figure:

    def __init__(self, path: str) -> None:
        self.path = path

    # def __enter__(self):
