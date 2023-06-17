import express from 'express'

import { db as mysqldb } from './mysql.js'
import { db as influxdb, iterateRows } from './influx.js'

const app = express()
const port = 3300

app.get('/runs/meta', (req, res) => {
	console.log('Querying Runs...')
	mysqldb.query(
		'select * from Runs',
		function (err, results) {
			if (err) console.error(err)
			if (results) console.log(results)
			res.send(results)
		},
	)
	console.log(`Query Complete`)
})

app.get('/run-:run/metrics', (req, res) => {
	console.log(`Querying Run (${req.params.run}) Metrics...`)
	const fluxQuery = `from(bucket:"metrics") |> range(start: -100y)`
	iterateRows(fluxQuery).then((value) => {
		console.log('Rows: ', value.length)
		res.send(value)
		console.log(`Query Complete`)
	})
})

app.listen(port, () => {
	console.log(`server listening on port ${port}`)
})
