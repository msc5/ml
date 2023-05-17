
import asyncio
import websockets
import redis

# Schema
# python -> redis <-- websocket <-- web app

# Establish a connection to the Redis server
redis_client = redis.Redis(host='localhost', port=6379)


async def websocket_echo(websocket, path):
    """
    WebSocket server handler.
    Echoes incoming messages from redis server.
    """

    # Handle incoming WebSocket connections
    async for message in websocket:

        print(websocket, path)
        print(message)

        # Retrieve data from the Redis server (comes in bytes)
        data_from_redis = redis_client.get(message)

        if data_from_redis is not None:

            # Echo the data back to the client
            await websocket.send(data_from_redis.decode())


async def run_server():
    """
    Run the event loop
    """

    # Start the WebSocket server
    async with websockets.serve(websocket_echo, 'localhost', 8765):

        print("WebSocket server started")
        await asyncio.Future()

# Run the server
asyncio.run(run_server())
