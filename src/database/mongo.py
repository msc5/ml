"""
Logs run data and metadata to MongoDB.
"""

from typing import Optional

from pymongo import MongoClient
from pymongo.database import Database

from ..dot import Dot

seen_metrics = set()

client: Optional[MongoClient] = None
db: Optional[Database] = None


def initialize():

    global client
    client = MongoClient("mongodb://localhost:27017/",
                         username="username",
                         password="password")

    global db
    db = client['ml']

    global session
    from ..shared import session


def log_run(info: Dot):

    global db
    if db is None:
        raise Exception("MongoDB connector not initialized")

    meta = {
        "name": info.name,
        "version": 0,
        "start_time": info.start_time
    }

    # Store run metadata
    inserted = db['runs'].insert_one(meta)

    # Return id
    return inserted.inserted_id


def log_metrics(metrics: Dot):

    global db
    if db is None:
        raise Exception("MongoDB connector not initialized")

    # Iterate through given metrics
    for name, _ in metrics:

        # Skip if seen
        if not name in seen_metrics:
            seen_metrics.add(name)

            # Try to find existing metric in database
            document = db['metrics'].find_one({'name': name})

            # If it does not exist, create
            if not document:
                db['metrics'].insert_one({"name": name, "runs": [session.info.id]})

            # If it does exist, append this run
            else:
                db['metrics'].update_one({"name": name}, {"$push": {'runs': session.info.id}})


if __name__ == "__main__":
    initialize()
