import networkx as nx
import numpy as np
import time

from config_utils import ConfigFile
from csv_graph import CSVGraph
from graph_gen import CSVGraphGenerator, custom_exponential_graph
from graph_gen import RandomGraphGenerator, RandomSingleGraphGenerator, GraphGenerator
from light_graph import LightGraph
from model_zoo import model_zoo
from policy import bound_policy
from scipy.sparse import csr_matrix
from sparse_utils import multiply_zeros_as_ones

from typing import Dict, Tuple


def load_model_from_config(cf: ConfigFile,
                           preloaded_graph: Tuple = None,
                           hyperparams: Dict = None,
                           model_random_seed: int = 42,
                           use_policy: str = None):

    # load model hyperparameters; default params from config file are overwritten by hyperparams dict (if provided)
    model_params = cf.section_as_dict("MODEL")
    if hyperparams is not None:
        model_params = {**model_params, **hyperparams}

    # load graph as described in config file; optionally use a different graph if provided in `preloaded graph`
    graph, A = load_graph(cf) if preloaded_graph is None else preloaded_graph

    # create model
    class_name = cf.section_as_dict("TASK").get(
        "model", "ExtendedNetworkModel")
    Model = model_zoo[class_name]
    model = Model(G=graph if A is None else A,
                  **model_params,
                  random_seed=model_random_seed)

    # apply policy on model
    if use_policy:  # TODO: cfg versus --policy option
        load_policy(cf, graph, use_policy)

    ndays = cf.section_as_dict("TASK").get("duration_in_days", 60)
    print_interval = cf.section_as_dict("TASK").get("print_interval", 1)
    verbose = cf.section_as_dict("TASK").get("verbose", "Yes") == "Yes"

    return model, {'T': ndays, 'print_interval': print_interval, 'verbose': verbose}


def load_graph(cf: ConfigFile):
    num_nodes = cf.section_as_dict("TASK").get("num_nodes", None)

    graph_name = cf.section_as_dict("GRAPH")["name"]
    nodes = cf.section_as_dict("GRAPH").get("nodes", "nodes.csv")
    edges = cf.section_as_dict("GRAPH").get("edges", "edges.csv")
    layers = cf.section_as_dict("GRAPH").get("layers", "etypes.csv")

    start = time.time()
    graph = create_graph(graph_name, nodes=nodes, edges=edges,
                         layers=layers, num_nodes=num_nodes)

    A = matrix(graph, cf)
    end = time.time()
    print("Graph loading: ", end - start, "seconds")
    # print(graph)

    return graph, A


def load_policy(cf: ConfigFile, graph, policy: str):
    policy_cfg = cf.section_as_dict("POLICY")
    if policy not in policy_cfg["name"]:
        raise ValueError("Unknown policy name.")

    if policy_cfg and "filename" in policy_cfg:
        policy = getattr(__import__(
            policy_cfg["filename"]), policy)
        policy = bound_policy(policy, graph)
        model.set_periodic_update(policy)
    else:
        print("Warning: NO POLICY IN CFG")
        print(policy_cfg)


def create_graph(name, nodes="nodes.csv", edges="edges.csv", layers="etypes.csv", num_nodes=None):

    if name == "romeo_and_juliet":
        if not verona_available:
            raise NotImplementedError(
                "Verona not available. Contact Roman Neruda for source files.")
        else:
            return Verona(random_seed=7)

    if name == "csv":
        return CSVGraphGenerator(path_to_nodes=nodes, path_to_edges=edges, path_to_layers=layers)

    if name == "csv_petra":
        return CSVGraph(path_to_nodes=nodes, path_to_edges=edges, path_to_layers=layers)

    if name == "csv_light":
        return LightGraph(path_to_nodes=nodes, path_to_edges=edges, path_to_layers=layers)

    if name == "seirsplus_example":
        base_graph = nx.barabasi_albert_graph(n=num_nodes, m=9, seed=7)
        np.random.seed(42)
        return custom_exponential_graph(base_graph, scale=100)

    if name == "random":
        return RandomGraphGenerator()

    raise ValueError(f"Graph {name} not available.")


def matrix(graph, cf):

    if cf:
        scenario = cf.section_as_dict("SCENARIO")
    else:
        scenario = False

    if isinstance(graph, CSVGraph):
        if scenario:
            raise NotImplementedError(
                "CSVGraph does not support closing layers yet.")
        return graph.G

    if isinstance(graph, LightGraph):
        if scenario:
            raise NotImplementedError(
                "LightGraph does not support closing layers yet.")
        return graph.A

    if isinstance(graph, RandomSingleGraphGenerator):
        if scenario:
            raise NotImplementedError(
                "RandomGraphGenerator does not support closing layers.")
        return graph.G

    if isinstance(graph, GraphGenerator):
        if scenario:
            list_of_closed_layers = scenario["closed"]
            graph.close_layers(list_of_closed_layers)
        return magic_formula(
            graph
        )
    else:
        return None


def magic_formula(graph):

    # rozvrtany ... nutno opravit

    N = graph.number_of_nodes()

    #    print(graph.get_layers_info())

    prob_no_contact = csr_matrix((N, N))  # empty values = 1.0

    for name, prob in graph.get_layers_info().items():
        a = nx.adj_matrix(graph.get_graph_for_layer(name))
        if len(a.data) == 0:
            continue
        a = a.multiply(prob)  # contact on layer
        not_a = a  # empty values = 1.0
        not_a.data = 1.0 - not_a.data
        prob_no_contact = multiply_zeros_as_ones(prob_no_contact, not_a)
        del a

    # probability of contact (whatever layer)
    prob_of_contact = prob_no_contact
    prob_of_contact.data = 1.0 - prob_no_contact.data
    return prob_of_contact
