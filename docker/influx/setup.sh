#!/bin/bash

# Start influxdb server
influxd run --reporting-disabled &
echo "[ influxdb ] Server Started"

# Wait for influxdb server to be initialized
while [ ! $(influx ping 2> /dev/null) ]
do
	sleep 0.1
done
echo "[ influxdb ] Initialization Complete"

influx setup \
	--bucket metrics \
	--org ml \
	--username=username \
	--password=password \
	--token=token \
	--host=http://influxdb:8086 \
	--force \
	2> /dev/null
echo "[ influxdb ] Setup Complete"

sleep infinity
