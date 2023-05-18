import os
import io
import atexit
import subprocess

from ..mp import Thread

# Create log directory
log_dir = 'logs'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)


class ServerThread (Thread):

    name: str
    command: str

    log: io.TextIOWrapper
    log_path: str

    def __init__(self, name: str = 'server', command: str = '', *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.name = name
        self.command = command
        self.log_path = os.path.join(log_dir, self.name + '.log')

    def run(self):

        # Start the Redis server as a subprocess
        with open(self.log_path, 'w') as self.log:

            self.process = subprocess.Popen(self.command, shell=True,
                                            stdout=self.log, stderr=self.log)

            self.log.write('Redis Server Started')

        # Register a callback function to be called when the thread exits
        atexit.register(self.cleanup)

    def cleanup(self):

        # Terminate the Redis server process
        if self.process and self.process.poll() is None:
            self.process.terminate()
            self.process.wait()
