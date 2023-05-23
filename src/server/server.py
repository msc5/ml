import os
import io
import atexit
import signal
import subprocess
import time
from typing import Optional

from ..mp import Thread

# Create log directory
log_dir = os.path.join(os.getcwd(), 'logs')
if not os.path.exists(log_dir):
    os.makedirs(log_dir)


class ServerThread (Thread):

    name: str
    command: str

    log: io.TextIOWrapper
    log_path: str

    process: Optional[subprocess.Popen] = None

    def __init__(self, name: str = 'server', command: str = '', *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.name = name
        self.command = command

        self.log_path = os.path.join(log_dir, self.name + '.log')

    def run(self):
        """
        Start server in new thread.
        """

        with open(self.log_path, 'w') as self.log:

            self.log.write(f'[ {self.name} ] server starting\n')

            path = os.path.dirname(os.path.realpath(__file__))
            os.chdir(path)

            self.process = subprocess.Popen('exec ' + self.command, shell=True,
                                            stdout=self.log, stderr=self.log)

            self.log.write(f'[ {self.name} ] server started\n')

        # Register a callback function to be called when the thread exits
        atexit.register(self.cleanup)

        while self.process.poll() is None:
            time.sleep(1)

    def cleanup(self):
        """
        Close server (before closing thread).
        """

        with open(self.log_path, 'a') as self.log:

            self.log.write(f'[ {self.name} ] server closing\n')

            # Terminate the Redis server process
            if self.process and self.process.poll() is None:
                self.process.terminate()
                self.process.wait()

            self.log.write(f'[ {self.name} ] server closed\n')
