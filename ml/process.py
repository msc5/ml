from typing import Callable
import torch.multiprocessing as mp

from src.ml.cli import console


def inline(function: Callable):

    def wrapper(*args, **kwargs):
        manager = mp.Manager()
        queue = manager.Queue()

        def line(*args, **kwargs):
            result = function(*args, **kwargs)
            queue.put(result)

        process = mp.Process(target=line, args=args, kwargs=kwargs)
        process.start()
        process.join()

        result = queue.get()

        return result

    return wrapper


if __name__ == "__main__":

    @inline
    def func(x):
        return x**2

    data = func(25)
    console.log(data)
