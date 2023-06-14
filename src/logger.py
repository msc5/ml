"""
Logs data to Influxdb.
"""

from typing import Optional, Union

from influxdb_client import InfluxDBClient, Point, WriteApi
from influxdb_client.client.write_api import SYNCHRONOUS
import torch

FLUSH_INTERVAL: int = 100

api: Optional[WriteApi] = None
points: list[Point] = []

# Config

token = "token"
org = "ml"
url = "http://localhost:8086"
bucket = "metrics"


def initialize():
    """
    Initialize Influxdb client.
    """

    client = InfluxDBClient(url=url, token=token, org=org)

    global api
    api = client.write_api(write_options=SYNCHRONOUS)

    global session
    from .shared import session

    return api


def log(key: str, value: Union[float, torch.Tensor], tags: dict = {}, fields: dict = {}):
    """
    Log a data point to Influxdb.
    """

    global points

    if isinstance(value, torch.Tensor):
        value = value.item()

    if session.info.influxdb:

        # Construct point
        point = (
            Point(key)
            .tag("run_version", session.info.version)
            .tag("run_name", session.info.name)
            .field("value", value)
            .field("step", session.trainer.progress.get('session'))
        )

        # Add additional tags
        for key, value in tags.items():
            point = point.tag(key, value)

        # Add additional fields
        for key, value in fields.items():
            point = point.field(key, value)

        points.append(point)

        # Flush if list is full
        if len(points) >= FLUSH_INTERVAL:

            global api
            if api is None:
                api = initialize()

            api.write(bucket=bucket, org="ml", record=point)

            points = []


def close():
    """
    Close Influxdb client.
    """

    global api
    if api is not None:
        api.close()

    del api
