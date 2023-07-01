import express from 'express'
import cors from 'cors'

import { db as mysqldb } from './mysql.js'
import { db as influxdb, iterateRows } from './influx.js'

const app = express()
const port = 3300

app.use(cors())

app.get('/metrics/meta', (req, res) => {
	console.log('Querying metrics metadata...')
	const query = 'select * from metadata'
	mysqldb.query(
		query,
		function (err, results) {
			if (err) console.error(err)
			if (results) console.log(results)
			res.send(results)
		},
	)
	console.log(`Query Complete`)
})

app.get('/runs/meta', (req, res) => {
	console.log('Querying runs metadata...')
	const query = 'select * from runs'
	mysqldb.query(
		query,
		function (err, results) {
			if (err) console.error(err)
			if (results) console.log(results)
			res.send(results)
		},
	)
	console.log(`Query Complete`)
})

app.get('/run-:id/metrics', (req, res) => {
	console.log(`Querying Run (id: ${req.params.id}) Metrics...`)
	const query = `
	from(bucket: "metrics")
		|> range(start: -1000y)
		|> filter(fn: (r) => r["_measurement"] == "metrics")
		|> filter(fn: (r) => r["run_id"] == "${req.params.id}")
		|> pivot(rowKey: ["_time"], columnKey: ["_field"], valueColumn: "_value")`
	iterateRows(query).then((value) => {
		console.log('Rows: ', value.length)
		res.send(value)
		console.log(`Query Complete`)
	})
})

app.listen(port, () => {
	console.log(`server listening on port ${port}`)
})
