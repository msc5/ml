import mysql from 'mysql2'

// Define db creds
const db = mysql.createConnection({
	host: 'mysql',
	database: 'meta',
	port: '3306',
	user: 'root',
	password: 'password',
})

// Log any errors connected to the db
db.connect(function (err) {
	if (err) console.log(err)
})

export { db }
