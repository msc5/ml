
from flask import Flask

from ..mp import Thread

app = Flask(__name__)


@app.route('/metrics')
def get_metrics():

    from ..trainer import CurrentTrainer
    if CurrentTrainer is not None:
        return CurrentTrainer.metrics._dict()
    else:
        return {}


def start():

    def server():
        app.run(host='localhost', port=8081, debug=False)

    thread = Thread(target=server, daemon=True)
    thread.start()
