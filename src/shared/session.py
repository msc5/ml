"""
This module stores variables that are relevant for a single training session.
"""

import git
import os
import wandb

from ..dot import Dot
from ..io import generate_name
from ..trainer import Trainer
from ..util import Metadata
from ..renderables import check
from ..mp import Manager
from ..cli import console

trainer: Trainer
info: Dot

manager: Manager


def start(trainer: Trainer):
    """
    Starts a training session from an initialized trainer object.
    """

    # Initialize session info
    global info
    info = Dot()

    # Initialize manager for multiprocessing
    global manager
    manager = Manager()
    manager.start()
    info.exit_event = manager.Event()

    # Get run version number if "./metadata.json" exists, otherwise create
    # file and set version number to 0.
    with Metadata('.') as meta:
        meta.data['version'] = info.version = meta.data.get('version', 0) + 1

    # Get github info
    info.github = g = Dot()
    g.repo = git.Repo(os.getcwd())  # type: ignore
    g.master = g.repo.head.reference
    g.branch = str(g.master.name)
    g.commit = str(g.master.commit.message).replace('\n', ' ').strip()

    # Get slurm job name
    name = os.environ.get('SLURM_JOB_NAME')
    id = os.environ.get('SLURM_JOB_ID')
    if name is not None and id is not None:
        info.slurm_id = f'{name}-{id}'

    # Initialize run name and directory
    info.name = trainer.group if trainer.group != "misc" else generate_name()
    info.dir = os.path.join(trainer.results_dir, trainer.opts.sys.module, info.name)
    check(f'Created Directory [cyan]{info.dir}[reset]', color='green')

    # Initialize wandb
    info.wandb = None
    if trainer.log:
        console.print()
        info.wandb = wandb.init(project=trainer.opts.sys.module, name=info.name,
                                group=trainer.wandb_group,
                                tags=[*trainer.tags, trainer.group],
                                config={k: v.value for k, v in trainer._gather_params()},
                                id=trainer.wandb_id if trainer.wandb_resume else None)
        console.print()
        if trainer.wandb_resume:
            check(f'Resumed wandb run [cyan]{trainer.wandb_id}[reset]', color='green')
        else:
            check('Initialized Wandb', color='green')
    check('Not Using Wandb', color='green')

    return info
