
from typing import Optional, Union

from influxdb_client import InfluxDBClient, Point, WriteApi
from influxdb_client.client.write_api import SYNCHRONOUS
import torch

api: Optional[WriteApi] = None

token = "token"
org = "ml"
url = "http://localhost:8086"
bucket = "metrics"


def initialize():

    client = InfluxDBClient(url=url, token=token, org=org)

    global api
    api = client.write_api(write_options=SYNCHRONOUS)

    global session
    from ..shared import session

    return api


def log(key: str, value: Union[float, torch.Tensor], tags: dict = {}, fields: dict = {}):

    if isinstance(value, torch.Tensor):
        value = value.item()

    if session.info.influxdb:

        # Construct point
        point = (
            Point(key)
            .tag("type", "metric")
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

        global api
        if api is None:
            api = initialize()

        api.write(bucket=bucket, org="ml", record=point)


def close():

    global api
    if api is not None:
        api.close()

    del api
