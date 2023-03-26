import argparse
import importlib
import socket
import subprocess
import sys

import textual.app as app
from textual.widget import Widget
import textual.widgets as widgets

from ..cli import console
from ..options import Options
from ..trainer import Trainer


class Dashboard (Widget):

    is_scrollable: bool = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Trainer initialization
        parser = argparse.ArgumentParser(prog='VAIL')
        parser.add_argument('model', type=str, default=None)
        parser.add_argument('--update', action='store_true')

        args = parser.parse_args([sys.argv[1]])

        if args.update:
            try:
                subprocess.call(['git', 'pull'])
                console.log('[blue]Pulled recent changes')
            except:
                console.log('[red]Could not pull recent changes')

        module = args.model
        Module = importlib.import_module(f'src.models.{module}')

        args = Module.Trainer.parse()
        args.sys = Options(host=socket.gethostname(),
                           module=module)

        trainer: Trainer = Module.Trainer(args)
        # trainer.train()

        self._trainer = trainer
        # self._trainer.start()

    def render(self) -> app.RenderResult:

        return self._trainer.dashboard()
        # return self._trainer.renderables.title()
        # return self._trainer.renderables.status()


class App (app.App):

    BINDINGS = [('q', 'quit', 'Quit')]
    CSS_PATH = 'app.css'

    is_scrollable: bool = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._dashboard = Dashboard()

    def compose(self) -> app.ComposeResult:

        yield self._dashboard

        # yield widgets.Header()
        # yield widgets.Footer()

        self._dashboard._trainer.train()


if __name__ == "__main__":

    app = App()
    app.run()
