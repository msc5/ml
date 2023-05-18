

# import influxdb_client
# import os
# import time
#
# from influxdb_client import InfluxDBClient, Point, WritePrecision
# from influxdb_client.client.write_api import SYNCHRONOUS
#
# token = os.environ.get("INFLUXDB_TOKEN")
# org = "ml"
# url = "http://localhost:8086"
# write_client = influxdb_client.InfluxDBClient(url=url, token=token, org=org)
#
#
# bucket="initial_bucket"
#
# write_api = write_client.write_api(write_options=SYNCHRONOUS)
#
# import torch
#
# for value in torch.linspace(0, 1, steps=100):
#   point = (
#     Point("measurement2")
#     .field("field1", value.item())
#   )
#   write_api.write(bucket=bucket, org="ml", record=point)
#   # time.sleep(1) # separate points by 1 second
