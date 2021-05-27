import timeit
import time
import click
import random
import pickle
from config_utils import ConfigFile
from model_m import load_graph


@click.command()
@click.argument('filename', default="example.ini")
@click.argument('outputname', default="graph.pickle")
@click.option('--precalc_matrix/--no_matrix', default=False)
def main(filename, outputname, precalc_matrix):
    """ Load the graph and pickle. """

    cf = ConfigFile()
    cf.load(filename)

    graph = load_graph(cf)

    # for e in set(graph.G.edges()):
    #     print(e)
    #     edges = graph.get_layers_for_edge(*e)
    #     print(edges)
    # exit()

    if precalc_matrix:
        graph.final_adjacency_matrix()

    with open(outputname, 'wb') as f:
        pickle.dump(graph, f, protocol=4)


if __name__ == "__main__":
    main()
