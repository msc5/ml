
import os

from influxdb_client import InfluxDBClient, Point
from influxdb_client.client.write_api import SYNCHRONOUS

token = os.environ.get("INFLUXDB_TOKEN")
org = "ml"
url = "http://localhost:8086"
bucket = "initial_bucket"

write_client = InfluxDBClient(url=url, token=token, org=org)
write_api = write_client.write_api(write_options=SYNCHRONOUS)

import torch

for value in torch.linspace(0, 1, steps=10):
    point = (
        Point("measurement2")
        .field("field1", value.item())
    )
    write_api.write(bucket=bucket, org="ml", record=point)
    # time.sleep(1) # separate points by 1 second

write_api.close()
write_client.close()
