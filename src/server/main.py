#
# import asyncio
#
# import redis
#
# from websockets.server import serve
#
# from ..mp import Thread
#
# # # ----------------------------------------
# # # Websocket
# # # ----------------------------------------
# #
# #
# # async def echo(websocket):
# #     async for message in websocket:
# #         await websocket.send(message)
# #
# #
# # async def main():
# #     async with serve(echo, "localhost", 8765):
# #         await asyncio.Future()  # run forever
# # # ----------------------------------------
#
# # ----------------------------------------
# # Redis
# # ----------------------------------------
# _host = 'localhost'
# _port = 6379
#
#
# def server(r: redis.Redis):
#
#     while True:
#
#         from ..trainer import CurrentTrainer
#         if CurrentTrainer is not None:
#             r.mset(CurrentTrainer.metrics._dict())
# # ----------------------------------------
#
#
# def start():
#
#     # r = redis.Redis(_host, _port)
#     # thread = Thread(target=server, args=[r], daemon=True)
#     # thread.start()
#
#     asyncio.run(main())
