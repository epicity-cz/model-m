import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))

import click
from utils.config_utils import ConfigFile

from model_m.model_m import load_graph, save_graph


@click.command()
@click.argument('filename', default="example.ini")
def main(filename):
    """ Load the graph and pickle. """

    cf = ConfigFile()
    cf.load(filename)

    filename = cf.section_as_dict("GRAPH").get("file", None)

    if filename is not None:
        graph = load_graph(cf)
    else:
        print("Please, specify path and name for the graph to save in you INI file (e.g. 'file=./graph.pickle').")
        print("Graph not loaded.")


if __name__ == "__main__":
    main()
