"""
This module stores variables that are relevant for a single training session.
"""

from collections import defaultdict
import random
import git
import os
import wandb
import socket
import time

from ..dot import Dot
from ..io import generate_name
from ..trainer import Trainer
from ..util import Metadata
from ..renderables import check
from ..mp import Manager, Thread
from ..cli import console
from ..database import mysql, mongo

# -------------------- Global Variables -------------------- #

trainer: Trainer
manager: Manager

main_thread: Thread = Thread(main=True)
modes: defaultdict = defaultdict(dict)

threads: dict
info: Dot

# -------------------- Functions -------------------- #


def start(trainer: Trainer):
    """
    Starts a training session from an initialized trainer object.
    """

    # Initialize session info
    global info
    info = Dot()
    info.start_time = time.time()

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
    info.slurm_id = f'{name}-{id}' if name is not None and id is not None else ''

    # Check for database instances
    def is_port_in_use(port: int) -> bool:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            return s.connect_ex(('localhost', port)) == 0
    info.influxdb = is_port_in_use(8086)
    info.mysql = is_port_in_use(3307)
    info.mongodb = is_port_in_use(27017)

    # Initialize run name and directory
    # info.id = random.getrandbits(32)
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
                                config={k: v.value for k, v in trainer._gather_params()})
        check('Initialized Wandb', color='green')
    check('Not Using Wandb', color='green')

    # Push "info" to metadata database (mongo)
    mongo.initialize()
    info.id = mongo.log_run(info)

    return info


def thread(mode: str = ''):

    def wrapped(function):

        # Add to modes map
        global modes
        modes[mode][function.__name__] = function

    return wrapped


def start_threads():

    global threads
    global modes
    global trainer

    threads = {}
    for name, function in modes[trainer.mode].items():
        threads[name] = Thread(target=function, args=[trainer], daemon=False)

    for thread in threads.values():
        thread.start()
