"""
    Main server script.

    1. Initializes Servers
        1. InfluxDB
        2. Web Server
"""

import time

from rich.live import Live

from ..mp import Thread
from .server import ServerThread

main_thread = Thread(main=True)

threads = [ServerThread('influxdb', 'influxd')]

with Live(main_thread):

    for thread in threads:
        thread.start()

    time.sleep(30)
