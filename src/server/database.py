
import os
from typing import Optional, Union

from influxdb_client import InfluxDBClient, Point, WriteApi
from influxdb_client.client.write_api import SYNCHRONOUS
import torch

api: Optional[WriteApi] = None

token = os.environ.get("INFLUXDB_TOKEN")
org = "ml"
url = "http://localhost:8086"
bucket = "initial_bucket"


def initialize():

    client = InfluxDBClient(url=url, token=token, org=org)

    global api
    api = client.write_api(write_options=SYNCHRONOUS)

    return api


def log(key: str, step: int, value: Union[float, torch.Tensor]):

    if isinstance(value, torch.Tensor):
        value = value.item()

    point = (
        Point(key)
        .field("value", value)
        .field("step", step)
    )

    global api
    if api is None:
        api = initialize()
    api.write(bucket=bucket, org="ml", record=point)

def close():

    global api
    if api is not None:
        api.close()
