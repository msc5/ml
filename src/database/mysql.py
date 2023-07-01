"""
Logs run metadata to MySQL.
"""

from typing import Optional
import sqlalchemy as db

from ..dot import Dot
from ..cli import console

# Specify database configuration
config = {
    'host': 'localhost',
    'port': 3307,
    'user': 'root',
    'password': 'password',
    'database': 'meta'
}

engine: Optional[db.Engine] = None
connection: Optional[db.Connection] = None


def initialize():

    from ..shared import session
    if session.info.mysql:

        db_user = config.get('user')
        db_pwd = config.get('password')
        db_host = config.get('host')
        db_port = config.get('port')
        db_name = config.get('database')

        # Connection string
        connection_str = f'mysql+pymysql://{db_user}:{db_pwd}@{db_host}:{db_port}/{db_name}'

        # Connect to database
        global engine
        engine = db.create_engine(connection_str, echo=False)

        global connection
        connection = engine.connect()

        meta = db.MetaData()
        db.MetaData.reflect(meta, bind=engine)

        return engine, connection


def log_run(info: Dot):

    from ..shared import session
    if session.info.mysql:

        global engine
        global connection
        if connection is None or engine is None:
            engine, connection = initialize()

        meta = db.MetaData()
        db.MetaData.reflect(meta, bind=engine)

        table = meta.tables["runs"]

        statement = (db.insert(table)
                     .values(id=info.id,
                             name=info.name,
                             version=0,
                             start_time=info.start_time))

        connection.execute(statement)
        connection.commit()


def log_metric(metric: Dot):

    from ..shared import session
    if session.info.mysql:

        global engine
        global connection
        if connection is None or engine is None:
            engine, connection = initialize()

        meta = db.MetaData()
        db.MetaData.reflect(meta, bind=engine)

        breakpoint()

        table = meta.tables["metrics"]

        statement = (db.insert(table)
                     .values(name=metric.name))

        connection.execute(statement)
        connection.commit()


if __name__ == "__main__":
    initialize()
