import os
import random


def readline(filename):
    path = os.path.dirname(os.path.realpath(__file__))
    return random.choice(
        open(os.path.join(path, filename)).readlines()).replace('\n', '')


def name() -> str:
    adjective = readline(os.path.join('words', 'adjectives.txt'))
    noun = readline(os.path.join('words', 'nouns.txt'))
    return f'{adjective}-{noun}'


if __name__ == "__main__":

    for i in range(50):
        print(name())
