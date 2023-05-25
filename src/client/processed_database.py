
from typing import Optional, Union

from influxdb_client import InfluxDBClient, Point, WriteApi
from influxdb_client.client.write_api import SYNCHRONOUS
import torch

from ..mp import ManagedQueue, Process

# api: Optional[WriteApi] = None
process: Optional[Process] = None

token = "token"
org = "ml"
url = "http://localhost:8086"
bucket = "metrics"


def log_loop(queues: dict[str, ManagedQueue], api: WriteApi):

    while True:

        key, value = queues['in'].get()

        if key == None and value == None:
            api.close()
            return

        else:

            (tags, fields) = value

            if isinstance(value, torch.Tensor):
                value = value.item()

            # Construct point
            point = (
                Point(key)
                .tag("type", "metric")
                # .field("value", value)
                # .tag("run_version", session.info.version)
                # .tag("run_name", session.info.name)
                # .field("step", session.trainer.progress.get('session'))
            )

            for key, val in tags.items():
                point = point.tag(key, val)

            for key, val in fields.items():
                point = point.field(key, val)

            api.write(bucket=bucket, org="ml", record=point)


def initialize():

    client = InfluxDBClient(url=url, token=token, org=org)

    global api
    api = client.write_api(write_options=SYNCHRONOUS)

    global session
    from ..shared import session

    global process
    process = Process(target=log_loop, args=[api])

    global queue
    queue = process.queues['in']

    process.start()


def log(key: str, value: Union[float, torch.Tensor]):

    global process
    global queue
    global session

    tags = {'run_version': session.info.version, 'run_name': session.info.name}
    fields = {'value': value, 'step': session.trainer.progress.get('session')}

    # if queue is not None and process is not None and process.process.is_alive():
    #     queue.put((key, (tags, fields)))


def close():

    global process
    if process is not None:
        process.close()
