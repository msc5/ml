import os
import atexit
import subprocess
import time
import asyncio
import redis
import websockets
import io

from ..mp import Thread
from ..renderables import check

# Create log directory
log_dir = 'logs'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)


class RedisServerThread (Thread):

    log: io.TextIOWrapper
    log_path: str =  os.path.join(log_dir, 'redis.log')

    def run(self):

        # Set the command to start the Redis server
        redis_server_command = 'redis-server'

        # Start the Redis server as a subprocess
        with open(self.log_path, 'w') as self.log:

            self.redis_process = subprocess.Popen(redis_server_command, shell=True,
                                                  stdout=self.log, stderr=self.log)

            self.log.write('Redis Server Started')
            check('Redis Server Started')

        # Register a callback function to be called when the thread exits
        atexit.register(self.cleanup)

    def cleanup(self):

        # Terminate the Redis server process
        if self.redis_process and self.redis_process.poll() is None:
            self.redis_process.terminate()
            self.redis_process.wait()


class WebSocketServerThread (Thread):

    log: io.TextIOWrapper
    log_path: str =  os.path.join(log_dir, 'websocket.log')

    def run(self):

        # Establish a connection to the Redis server
        redis_client = redis.Redis(host='localhost', port=6379)

        with open(self.log_path, 'w') as self.log:

            async def websocket_echo(websocket, path):
                """
                WebSocket server handler.
                Echoes incoming messages from redis server.
                """

                # Handle incoming WebSocket connections
                async for message in websocket:

                    self.log.write(f'Message:       {message}\n')
                    self.log.flush()

                    # Retrieve data from the Redis server (comes in bytes)
                    data_from_redis = redis_client.get(message)

                    if data_from_redis is not None:

                        response = data_from_redis.decode()

                        self.log.write(f'Response:      {response}\n')
                        self.log.flush()

                        # Echo the data back to the client
                        await websocket.send(response)

            async def run_server():
                """
                Run the event loop
                """

                # Start the WebSocket server
                async with websockets.serve(websocket_echo, 'localhost', 8765):

                    self.log.write('WebSocket server started')
                    self.log.flush()
                    check('WebSocket Server Started')

                    await asyncio.Future()

            asyncio.run(run_server())


# Create an instance of the RedisServerThread
redis_thread = RedisServerThread()
redis_thread.start()

# Create an instance of the WebSocketServerThread
websocket_server_thread = WebSocketServerThread()
websocket_server_thread.start()

# # Create an instance of the WebServerThread
# webserver_thread = WebServerThread()
# webserver_thread.start()

time.sleep(2)
