from .src.options import Dot
from .src.cli import console

import argparse

parser = argparse.ArgumentParser(prog='ml')

subparsers = parser.add_subparsers(title='command')

runs_parser = subparsers.add_parser('runs', description='Perform actions on collected runs')
runs_parser.add_argument('command', type=str, choices=['runs'], default=None)

args = Dot(parser.parse_args().__dict__)
console.log(args)
