import click
from utils.config_utils import ConfigFile

from model_m.model_m import load_graph, save_graph


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

    save_graph(outputname, graph)


if __name__ == "__main__":
    main()
