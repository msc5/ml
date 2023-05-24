FROM influxdb:latest

RUN influx setup \
    --username=username \
    --password=password \
    --bucket metrics \
    --org ml \
    --force
