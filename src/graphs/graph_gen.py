# these graphs are not used and the file is not maintained at the moment
# random graphs should be used in the future

# -*- coding: utf-8 -*-
import networkx as nx
import numpy as np
import scipy.stats as stats
from scipy.sparse import csr_matrix
import pandas as pd

from utils.sparse_utils import multiply_zeros_as_ones


class GraphGenerator:
    layer_names = ['F', 'D', 'P', 'E', 'H', 'K',
                   'C', 'S', 'O', 'L', 'R', 'T', 'X', 'Z']
    layer_probs = [1.0] * len(layer_names)
    layer_types = list(range(len(layer_names)))

    def __init__(self, random_seed=None):
        self.G = nx.MultiGraph()
        self.Graphs = {}
        self.G.graph['layer_names'] = self.layer_names
        self.G.graph['layer_probs'] = self.layer_probs

        if random_seed:
            np.random.seed(random_seed)

        self.A = None
        self.A_valid = False
        self.A_invalids = set()
        self.quarantined_edges = {}

    @property
    def nodes(self):
        return self.G.nodes

    def number_of_nodes(self):
        return self.G.number_of_nodes()

    def as_multigraph(self):
        return self.G

    def as_one_graph(self):
        return nx.Graph(self.G)

    def as_aggregated_graph(self):
        ag = nx.Graph()
        ag.add_nodes_from(self.G)

        lp = []
        selected_edges = []

        for i, l in enumerate(self.layer_names):
            lp[i] = self.G.graph['layer_probs'][i]
            selected_edges[i] = [(u, v, e) for u, v, e in self.G.edges(
                data=True) if e['type'] == l]

        return ag

    def final_adjacency_matrix(self):
        """ recalculates final adjacency matrix and returns it as csr_matrix """

        if self.A_valid:
            return self.A

        if self.A is None or not self.A_invalids:
            N = self.number_of_nodes()
            prob_no_contact = csr_matrix((N, N))  # empty values = 1.0

            for name, prob in self.get_layers_info().items():
                a = nx.adj_matrix(self.get_graph_for_layer(name))
                if len(a.data) == 0:  # no edges, prob of no-contact = 1
                    continue
                a = a.multiply(prob)  # contact on layer
                not_a = a  # empty values = 1.0
                not_a.data = 1.0 - not_a.data  # prob of no-contace
                prob_no_contact = multiply_zeros_as_ones(
                    prob_no_contact, not_a)
                del a

            # prob_of_contact = 1.0 - prob_no_contact
            prob_of_contact = prob_no_contact
            prob_of_contact.data = 1.0 - prob_no_contact.data
            self.A = prob_of_contact
            self.A_valid = True
            self.A_invalids = set()
            return self.A

        else:
            # recalculate only invalids
            for u, v in list(self.A_invalids):
                edges_to_aggregate = self.G[u][v]
                sublayers = dict()
                for e in edges_to_aggregate.values():
                    sublayers[e["type"]] = sublayers.get(
                        e["type"], 0) + e["weight"]

                no_contact_prob = 1.0
                for layer_type, weight in sublayers.items():
                    w = 1.0 - weight * self.G.graph["layer_probs"][layer_type]
                    no_contact_prob *= w
                self.A[u, v] = 1.0 - no_contact_prob
                self.A[v, u] = 1.0 - no_contact_prob
            self.A_valid = True
            self.A_invalids = set()
            return self.A

    def get_graph_for_layer(self, layer_name):
        ret_g = nx.Graph()
        ret_g.graph['layer_name'] = layer_name
        layer_index = self.layer_names.index(layer_name)
        ret_g.graph['layer_prob'] = self.G.graph['layer_probs'][layer_index]

        ret_g.add_nodes_from(self.G)
        selected_edges = [(u, v, e)
                          for u, v, e in self.G.edges(data=True)
                          if e['type'] == self.layer_names.index(layer_name)]
        ret_g.add_edges_from(selected_edges)

        return ret_g

    def as_dict_of_graphs(self):
        self.Graphs = {}

        for i, l in enumerate(self.layer_names):
            FG = nx.Graph()
            FG.graph['layer_name'] = l
            FG.graph['layer_prob'] = self.G.graph['layer_probs'][i]
            FG.add_nodes_from(self.G)
            selected_edges = [(u, v, e) for u, v, e in self.G.edges(
                data=True) if e['type'] == l]
            FG.add_edges_from(selected_edges)
            self.Graphs[l] = FG
        return self.Graphs

    def get_attr_list(self, attr):
        attr_list = []
        for (p, d) in self.G.nodes(data=True):
            attr_list.append(d[attr])
        return attr_list

    def get_edges_for_node(self, node_id):
        """ changes edges' weights """

        if not self.G.has_node(node_id):
            print(f"Warning: modify_layer_for_node called for nonexisting node_id {node_id}")
            return

        # for key in what_by_what:
        #     edges_to_modify = [
        #         (u, v, k) for u, v, k, d in self.G.edges(node, data=True, keys=True)
        #         if d['label'] == key
        #     ]
        #     print(edges_to_modify)
        #     for e in edges_to_modify:
        #         self.G.edges[e]['weight'] *= what_by_what[key]

        # for e in self.G.edges(node, data=True, keys=True):
        #     print(*e)

#        for e in self.G.edges([node_id], data=True, keys=True):
#            print(*e)
        return self.G.edges([node_id], data=True, keys=True)

    def get_layers_for_edge(self, u, v):
        edges = self.G[u][v]
        return [d["type"] for d in edges.values()]

    def modify_layers_for_nodes(self, node_id_list, what_by_what, is_quarrantined=None):
        """ changes edges' weights """

        if not what_by_what:
            return

        # if not self.G.has_node(node_id):
        #     print(f"Warning: modify_layer_for_node called for nonexisting node_id {node_id}")
        #     return

        changed = set()
        # keep the original list (it is modified in the cycle)
        for u, v, k, d in self.G.edges(set(node_id_list), data=True, keys=True):
            if is_quarrantined is not None and (is_quarrantined[u] > 0 or is_quarrantined[v] > 0):
                # edge is already quarrantined
                continue
            layer_label = d["type"]
            if layer_label in what_by_what:
                assert (u, v, k, d["type"], d["subtype"]) not in self.quarantined_edges, \
                    f"edge {(u, v, k, d['type'], d['subtype'])} is already quaratined, {is_quarrantined[u]}{is_quarrantined[v]}"
                s, e = (u, v) if u < v else (v, u)
                self.quarantined_edges[(
                    s, e, k, d["type"], d["subtype"])] = d['weight']
                self.G.edges[(u, v, k)
                             ]['weight'] = min(self.G.edges[(u, v, k)]['weight'] * what_by_what[layer_label], 1.0)
                changed.add((u, v))

        if changed:
            self.A_invalids.update(changed)
            self.A_valid = False

    def recover_edges_for_nodes(self, release, normal_life, is_quarrantined):

        changed = set()
        for u, v, k, d in self.G.edges(release, data=True, keys=True):
            if is_quarrantined[u] or is_quarrantined[v]:
                # still one of nodes is in quarrantine
                continue
            e_type = d["type"]
            e_subtype = d["subtype"]
            # new_weight = None
            # for edata in normal_life.G.get_edge_data(u, v).values():
            #     if edata["type"] == e_type and edata["subtype"] == e_subtype:
            #         new_weight = edata["weight"]
            #         break
            # if new_weight is None:
            #     raise ValueError("Edge not found in normal life.")
            s, e = (u, v) if u < v else (v, u)
            self.G.edges[(u, v, k)]['weight'] = self.quarantined_edges[(
                s, e, k, e_type, e_subtype)]
            del self.quarantined_edges[(
                s, e, k, e_type, e_subtype)]
            changed.add((s, e))
        if changed:
            self.A_invalids.update(changed)
            self.A_valid = False

    def get_layers_info(self):
        """ returns dictionary of layer names and probabilities """
        return dict(zip(self.G.graph["layer_names"], self.G.graph["layer_probs"]))

    def print_multi(self):
        dot_G = nx.nx_pydot.to_pydot(self.G)
        print(dot_G)

    def draw_multi(self, filename='empty_graph.png'):
        A = nx.nx_agraph.to_agraph(self.G)
        A.layout('dot')
        A.draw(filename)

    def close_layers(self, list_of_layers, coefs=None):
        for idx, name in enumerate(list_of_layers):
            i = self.G.graph["layer_names"].index(name)
            self.G.graph["layer_probs"][i] = 0 if not coefs else coefs[idx]
            print(f"Closing {name} (new value {self.G.graph['layer_probs'][i]})")
        self.A_valid = False

    def write_pickle(self, path):
        nx.write_gpickle(self.G, path)

    def read_picke(self, path):
        self.G = nx.read_gpickle(path)
        self.layer_names = self.G.graph['layer_names']
        self.layer_probs = self.G.graph['layer_probs']


def custom_exponential_graph(base_graph=None, scale=100, min_num_edges=0, m=9, n=None):
    """ Generate a random preferential attachment power law graph as a starting point.
    By the way this graph is constructed, it is expected to have 1 connected component.
    Every node is added along with m=8 edges, so the min degree is m=8.
    """
    if(base_graph):
        graph = base_graph.copy()
    else:
        assert(n is not None), "Argument n (number of nodes) must be provided when no base graph is given."
        graph = nx.barabasi_albert_graph(n=n, m=m)

# To get a graph with power-law-esque properties but without the fixed minimum degree,
# We modify the graph by probabilistically dropping some edges from each node.
    for node in graph:
        neighbors = list(graph[node].keys())
        quarantineEdgeNum = int(max(min(np.random.exponential(
            scale=scale, size=1), len(neighbors)), min_num_edges))
#        print(quarantineEdgeNum)
        quarantineKeepNeighbors = np.random.choice(
            neighbors, size=quarantineEdgeNum, replace=False)
        for neighbor in neighbors:
            if(neighbor not in quarantineKeepNeighbors):
                graph.remove_edge(node, neighbor)
    return graph


class RandomSingleGraphGenerator(GraphGenerator):

    def __init__(self, num_nodes=10000, **kwargs):
        super().__init__(**kwargs)
        self.num_nodes = num_nodes

        # generate random connections
        baseGraph = nx.barabasi_albert_graph(n=num_nodes, m=9)
        FG = custom_exponential_graph(baseGraph, scale=100)

#       list_of_zeroes = [ n for n, d in FG.degree() if d == 0 ]
#       if list_of_zeroes != []:
#           print('OMG: ', list_of_zeroes)

        # generate random weights from trunc norm
        lower, upper = 0, 1
        mu, sigma = 0.7, 0.3
        iii = 0
        for (u, v, d) in FG.edges(data=True):
            iii += 1
            FG[u][v]['weight'] = stats.truncnorm.rvs(
                (lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma)
        print(iii, 'Edges')
        self.G = FG


class RandomGraphGenerator(GraphGenerator):
    """ generating random graph with mean degree 13 and num_nodes nodes
    both weights and layer weights are initialized randomly from trunc. norm (0.7, 0.3)
    """

    def __init__(self, num_nodes=10000, **kwargs):
        super().__init__(**kwargs)
        self.num_nodes = num_nodes

        i = 0
        for l in self.layer_names:
            # generate random connections
            baseGraph = nx.barabasi_albert_graph(n=num_nodes, m=9)
            FG = custom_exponential_graph(baseGraph, scale=100)
            # generate random weights from trunc norm
            lower, upper = 0, 1
            mu, sigma = 0.7, 0.3
            for (u, v, d) in FG.edges(data=True):
                FG[u][v]['weight'] = stats.truncnorm.rvs(
                    (lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma)
                FG[u][v]['type'] = l
            # other params of the graph
            FG.graph['layer_name'] = l
            self.G.graph['layer_probs'][i] = stats.truncnorm.rvs(
                (lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma)
            FG.graph['layer_prob'] = self.G.graph['layer_probs'][i]
            self.Graphs[l] = FG
            i = i + 1
#            dot = nx.nx_pydot.to_pydot(FG)
#            print(dot)
        for l in self.layer_names:
            self.G.add_edges_from(self.Graphs[l].edges(data=True))

    def as_dict_of_graphs(self):
        return self.Graphs


class PickleGraphGenerator(GraphGenerator):
    def __init__(self, path_to_pickle='graph.pickle', **kwardgs):
        self.read_pickle(path_to_pickle)


class CSVGraphGenerator(GraphGenerator):

    layer_names = []
    layer_probs = []

    def __init__(self, path_to_nodes='nodes.csv', path_to_edges='edges.csv', path_to_layers='etypes.csv', **kwargs):
        super().__init__(**kwargs)

        csv_hacking = {'na_values': 'undef', 'skipinitialspace': True}
        nodes = pd.read_csv(path_to_nodes, **csv_hacking)
        edges = pd.read_csv(path_to_edges, **csv_hacking)
        layers = pd.read_csv(path_to_layers, **csv_hacking)

        indexNames = edges[edges['vertex1'] == edges['vertex2']].index
        if len(indexNames):
            print(f"Warning: dropping self edges!!!! {indexNames}")
            #            print(edges[edges['vertex1'] == edges['vertex2']])
            edges.drop(indexNames, inplace=True)

        #        print(layers)
        # fill the layers
#        layer_names = tuple(zip(layers.loc('id'), layers.loc('id2')))
        layers_to_add = layers.to_dict('list')
        self.layer_names = layers_to_add['name']
#        print(layers_names)
        self.layer_probs = layers_to_add['weight']
        self.layer_ids = layers_to_add['id']

        self.G.graph['layer_names'] = self.layer_names
        self.G.graph['layer_probs'] = self.layer_probs

        # fill the nodes
        nodes_to_add = nodes.to_dict('records')
        idx_s = list(range(0, len(nodes_to_add)))
        self.G.add_nodes_from(zip(idx_s, nodes_to_add))

        # fill the edges
        # get rid of duplicate edges
        if __debug__:
                    # fill the edges
            edges["e"] = edges.apply(
                lambda row: (
                    (int(row.vertex1), int(row.vertex2))
                    if int(row.vertex1) < int(row.vertex2)
                    else (int(row.vertex2), int(row.vertex1))
                ),
                axis=1
            )
            edges.drop_duplicates(inplace=True, subset=[
                                  "type", "subtype", "e", "weight"])
            edges.drop(columns=["e"], inplace=True)

        edges_to_add = edges.to_dict('list')
        froms = edges_to_add['vertex1']
        tos = edges_to_add['vertex2']
        datas = [{
            k: v
            for k, v in d.items()
            if k != 'vertex1' and k != 'vertex2'
        } for d in edges.to_dict('records')]
        self.G.add_edges_from(zip(froms, tos, datas))

    def __str__(self):
        return "\n".join([str(e) for e in self.G.edges(data=True)])
