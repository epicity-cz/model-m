#!/usr/bin/env python3
import os
import sys

import click

sys.path.append(os.path.join(os.path.dirname(__file__), '../src/gen_graph'))

from classes.region import Region
from config.base_config import BaseConfig


@click.command()
@click.option('-s', '--set', 'param_set', type=(str, str), multiple=True, help="set config value",
              metavar='<KEY> <VALUE>')
@click.argument('inifile', type=click.File('r'), metavar='<filename.ini>')
def main(param_set, inifile):
    """Generates graph files."""
    os.chdir(os.path.join(os.path.dirname(__file__), '..'))
    cfg = BaseConfig()
    cfg.read_ini(inifile)
    for (key, value) in param_set:
        cfg.set_key(key, value)
    Region.generate(cfg)


if __name__ == "__main__":
    main()
