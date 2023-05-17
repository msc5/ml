
import time
import asyncio
import websockets


async def send_test_request():

    async with websockets.connect('ws://localhost:8765') as websocket:

        start = time.perf_counter()

        # Send a test message to the WebSocket server
        message = 'matthew'
        await websocket.send(message)

        # Wait for the response from the server
        response = await websocket.recv()

        # Print the response received from the server
        print(f'Response from server: {response}')

        stop = time.perf_counter()
        print(stop - start)

# Run the test request
asyncio.run(send_test_request())
