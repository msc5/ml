import { InfluxDB } from '@influxdata/influxdb-client'

const org = 'ml'
const token = 'token'
// const url = 'http://influxdb:8086'
const url = 'http://localhost:8086'

const db = new InfluxDB({ url, token }).getQueryApi(org)

// Execute query and receive table metadata and table row values using async iterator.
async function iterateRows(query) {
	const rows = []
	for await (const { values, tableMeta } of db.iterateRows(query)) {
		rows.push(tableMeta.toObject(values))
	}
	return rows
}

export { db, iterateRows }
