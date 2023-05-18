
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
