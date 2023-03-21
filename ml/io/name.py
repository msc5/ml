import os
import random

from rich.progress import track

from ..cli import console

MAX_WORD_LENGTH = 8

_used: set[str] = set()
_loaded: dict[str, list[str]] = {}


def load(name: str):
    global _loaded
    if not _loaded.get(name):
        dirname = os.path.dirname(os.path.realpath(__file__))
        lines = open(os.path.join(dirname, 'words', name + '.txt')).readlines()
        filtered = [line.replace('\n', '') for line in lines if len(line) < MAX_WORD_LENGTH]
        _loaded[name] = filtered
    return _loaded[name]


def readline(name: str):
    lines = load(name)
    choice = random.choice(lines)
    return choice


def name() -> str:
    name = ''
    global _used
    while name == '' or name in _used:
        adjective = readline('adjectives')
        noun = readline('nouns')
        name = f'{adjective}-{noun}'
    _used.add(name)
    return name


if __name__ == "__main__":

    names = []
    for _ in track(range(100000), description='Checking for collisions'):
        names.append(name())

    console.log(len(names) == len(set(names)))
