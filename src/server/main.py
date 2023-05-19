"""
    Main server script.

    1. Starts Servers
        a. InfluxDB (backend database)
        b. Web Server (frontend)
    2. Closes Servers on exit

"""

import time
import asyncio

from rich.live import Live

from ..mp import Thread
from .server import ServerThread


def start():

    main_thread = Thread(main=True)

    threads = [ServerThread(name='influxdb', command='influxd'),
               ServerThread(name='website', command='npm run --prefix site/ dev')]

    with Live(main_thread):

        try:

            for thread in threads:
                thread.start()

            while True:
                time.sleep(1)

        except KeyboardInterrupt:

            for thread in threads:
                thread.join()


if __name__ == "__main__":

    start()
