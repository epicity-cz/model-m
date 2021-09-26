import numpy as np
from scipy.sparse import csr_matrix, lil_matrix


def compute_mean_degree(graph, nodes):
    """ computes mean number of expected contacts over the set of `nodes`
    in the given `graph`. `graph` is a LightGraph 
    """

    # first create matrix of all probs of contacts
    # (can be optimised, I do not care about time now - so for all nodes)
    graph_matrix = lil_matrix((graph.num_nodes, graph.num_nodes), dtype=float)
    
    for n1 in graph.nodes:
        for n2 in graph.nodes:
            index = graph.A[n1, n2]
            if index == 0: # no edge
                continue
            edges_repo = graph.edges_repo[index]
            probs = graph.get_edges_probs(np.array(edges_repo))
            probs = 1 - probs
            graph_matrix[n1, n2] = 1 - probs.prod()
    graph_matrix = csr_matrix(graph_matrix)


    def node_degree(node):
        return graph_matrix[node].sum()

    degrees = [node_degree(node) for node in nodes]
    return sum(degrees)/len(degrees)

