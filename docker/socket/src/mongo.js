import { MongoClient } from 'mongodb'

// const uri = 'mongodb://username:password@mongo:27017/'
const uri = 'mongodb://username:password@localhost'

const client = new MongoClient(uri)
const db = client.db('ml')

export { db }
