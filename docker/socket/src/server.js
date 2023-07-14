import express from 'express'
import cors from 'cors'

import { iterateRows } from './influx.js'
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
	iterateRows(query).then((value) => {
		console.log('Rows: ', value.length)
		res.send(value)
		console.log(`Query Success`)
	})
})

app.get('/plots', async (req, res) => {
	console.log('Querying plots metadata...')
	const documents = await mongo.collection('plots').find({}).toArray()
	res.send(documents)
	console.log(`Query Complete`)
})

app.post('/plots-add', (req, res) => {
	console.log(`Adding plot ${req.body}`)
	const plots = mongo.collection('plots')
	const id = plots.insertOne(req.body).insertedId
	res.send(id)
})

app.listen(port, () => {
	console.log(`server listening on port ${port}`)
})
