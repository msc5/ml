"""
Logs data to Influxdb.
"""

from typing import Optional

from influxdb_client import InfluxDBClient, Point, WriteApi, WriteOptions
import torch

from ..dot import Dot

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
    api = client.write_api(write_options=WriteOptions(flush_interval=FLUSH_INTERVAL))

    global session
    from ..shared import session

    return api


def log_metrics(metrics: Dot):

    if session.info.influxdb:

        # Construct point
        point = (
            Point('metrics')
            .tag("run_id", session.info.id)
            .tag("run_name", session.info.name)
            .tag("run_version", session.info.version)
            .tag("run_start_time", session.info.start_time)
            .field("step", session.trainer.progress.get('session'))
        )

        # Add fields
        for key, value in metrics:
            if isinstance(value, float) or isinstance(value, int):
                point = point.field(key, value)
            elif isinstance(value, torch.Tensor):
                point = point.field(key, value.item())

        global api
        if api is None:
            api = initialize()

        api.write(bucket=bucket, org="ml", record=point)


def close():
    """
    Close Influxdb client.
    """

    global api
    if api is not None:
        api.close()

    del api
