import express from 'express'
import cors from 'cors'

import { db as influx } from './influx.js'
import { db as mongo } from './mongo.js'

const app = express()
const port = 3300

app.use(cors())

app.get('/metrics/meta', async (req, res) => {
	console.log('Querying metrics metadata...')
	const documents = await mongo.collection('metrics').find({}).toArray()
	res.send(documents)
	console.log(`Found ${documents.length} metrics`)
	console.log(`Query Complete`)
})

app.get('/runs/meta', async (req, res) => {
	console.log('Querying runs metadata...')
	const documents = await mongo.collection('runs').find({}).toArray()
	res.send(documents)
	console.log(`Found ${documents.length} runs`)
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
	// iterateRows(query).then((value) => {
	// 	console.log('Rows: ', value.length)
	// 	res.send(value)
	// 	console.log(`Query Success`)
	// })
	influx.rows(query)
		// .pipe(({ values, tableMeta }) => tableMeta.toObject(values))
		.subscribe({
			next(o) {
				console.log(o)
				// console.log(`${o._time} ${o._measurement} in '${o.location}' (${o.example}): ${o._field}=${o._value}`)
			},
			error(e) {
				console.error(e)
				console.log('\nFinished ERROR')
			},
			// complete() {
			// 	console.log('\nFinished SUCCESS')
			// },
		})
})

app.listen(port, () => {
	console.log(`server listening on port ${port}`)
})
