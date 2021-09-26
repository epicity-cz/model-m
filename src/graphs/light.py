import logging
from operator import iconcat
from functools import reduce
from itertools import chain
import numexpr as ne
import numpy as np
import pandas as pd
from copy import copy
from scipy.sparse import csr_matrix, lil_matrix
import json

# import os
# os.environ["NUMEXPR_MAX_THREADS"] = "272"  # to fix numexpr complains


def concat_lists(l):
    """
    Returns concatenation of lists in the iterable l
    :param l: iterable of lists
    :return: concatenation of lists in l
    """
    return list(chain.from_iterable(l))


class LightGraph:

    """
    Graph for agent based network model. 
    Just a collection of numpy arrays (since we found networkx too slow for our purposes). 

    Typical usage is to load it from CSV file once and than pickle it for future simulations, 
    since building it from CSV is pretty slow. 

    Nodes has ids but internaly and in the model are indexed by their possition in original file (index).
    """

    # __slots__ = ['e_types', 'e_subtypes', 'e_probs', 'e_intensities', 'e_source', 'e_dest', 'e_valid', 'edges_repo',
    # 'edges_directions', '__dict__'] # not really helpful here, beneficial only for lots of small objects
    def __init__(self, random_seed=None):
        if random_seed:
            np.random.seed(random_seed)
        self.random_seed = random_seed
        self.edge_repo = None
        self.A = None
        self.is_quarantined = None

    def read_csv(self,
                 path_to_nodes='p.csv',
                 path_to_external='e.csv',
                 path_to_layers='etypes.csv',
                 path_to_edges='edges.csv',
                 path_to_quarantine=None,
                 path_to_layer_groups=None):

        csv_hacking = {'na_values': 'undef', 'skipinitialspace': True}
        base_nodes = pd.read_csv(
            path_to_nodes, **csv_hacking).drop_duplicates().reset_index()
        edges = pd.read_csv(path_to_edges, **csv_hacking)
        layers = pd.read_csv(path_to_layers, **csv_hacking)
        external_nodes = pd.read_csv(path_to_external, **csv_hacking)

        nodes = pd.concat([base_nodes, external_nodes],
                          ignore_index=True).drop(columns=["index"])

        if path_to_quarantine is None:
            self.QUARANTINE_COEFS = None
        else:
            df = pd.read_csv(path_to_quarantine, header=None, index_col=0)
            self.QUARANTINE_COEFS = {
                i: df.loc[i][1]
                for i in df.index
            }

        if path_to_layer_groups is not None:
            with open(path_to_layer_groups, "r") as f:
                self.LAYER_GROUPS = json.load(f)
        else:
            self.LAYER_GROUPS = None

        # layer names, ids and weights go to graph
        layers_to_add = layers.to_dict('list')

        self.layer_ids = layers_to_add['id']
        self.layer_name = layers_to_add['name']
        self.layer_weights = np.array(layers_to_add['weight'], dtype=float)

        # nodes
        # select categorical columns
        cat_columns = nodes.select_dtypes(['object']).columns
        nodes[cat_columns] = nodes[cat_columns].apply(
            lambda x: x.astype('category'))

        # save codes for backward conversion
        self.cat_table = {
            col: list(nodes[col].cat.categories)
            for col in cat_columns
        }

        # covert categorical to numbers
        nodes[cat_columns] = nodes[cat_columns].apply(
            lambda x: x.cat.codes)
        # pprint(nodes)

        # just test of conversion back
        # print(cat_columns)
        # for col in list(cat_columns):
        #     nodes[[col]] = nodes[[col]].apply(
        #         lambda x: pd.Categorical.from_codes(
        #             x, categories=cat_table[col])
        #     )
        # pprint(nodes)

        for col in nodes.columns:
            setattr(self, "nodes_"+col, np.array(nodes[col]))

        self.nodes = np.array(nodes.index)
        self.num_nodes = len(self.nodes)
        self.num_base_nodes = len(base_nodes)

        if self.num_nodes > 65535:
            raise ValueError(
                "Number of nodes too high (we are using unit16, change it to unit32 for higher numbers of nodes.")

        #        self.ignored = set(external_nodes["id"])

        # edges
        # drop self edges
        indexNames = edges[edges['vertex1'] == edges['vertex2']].index
        if len(indexNames):
            logging.warning("Warning: dropping self edges!!!!")
            edges.drop(indexNames, inplace=True)

        # fill edges to a graph
        n_edges = len(edges)
        # edges data"
        self.e_types = np.empty(n_edges, dtype="int16")
        self.e_subtypes = np.empty(n_edges, dtype="int16")
        self.e_probs = np.empty(n_edges, dtype="float32")
        self.e_intensities = np.empty(n_edges, dtype="float32")
        self.e_source = np.empty(n_edges, dtype="uint16")
        self.e_dest = np.empty(n_edges, dtype="uint16")
        self.e_active = np.ones(n_edges, dtype="bool")

        # if value == 2 than is valid, other numbers prob in quarantine
        self.e_valid = 2 * np.ones(n_edges, dtype="float32")
        # edges repo which will eventually be list of sets and not a dict
        self.edges_repo = {
            0: []
        }
        self.edges_directions = {
            0: []
        }
        key = 1
        # working matrix
        tmpA = lil_matrix((self.num_nodes, self.num_nodes), dtype="uint32")

        forward_edge = True
        backward_edge = False

        id_dict = {self.nodes_id[i]: i for i in range(self.num_nodes)}

        # fill data and get indicies
        for i, row in enumerate(edges.itertuples()):
            self.e_types[i] = row.layer
            self.e_subtypes[i] = row.sublayer
            self.e_probs[i] = row.probability
            self.e_intensities[i] = row.intensity

            # if row.vertex1 in self.ignored or row.vertex2 in self.ignored:
            #     continue

            try:
                i_row = id_dict[row.vertex1]
                i_col = id_dict[row.vertex2]
            except IndexError:
                print("Node does not exist")
                print(row.vertex1, row.vertex2)
                print(np.where(self.nodes_id == row.vertex1),
                      np.where(self.nodes_id == row.vertex2))
                exit()

            i_row, i_col = min(i_row, i_col), max(i_row, i_col)

            self.e_source[i] = i_row
            self.e_dest[i] = i_col

            if tmpA[i_row, i_col] == 0:
                # first edge between (row, col)
                self.edges_repo[key], self.edges_directions[key] = [
                    i], forward_edge
                self.edges_repo[key + 1], self.edges_directions[key +
                                                                1] = [i], backward_edge
                tmpA[i_row, i_col] = key
                tmpA[i_col, i_row] = key + 1
                key += 2
            else:
                # add to existing edge list
                print("+", end="")
                key_forward = tmpA[i_row, i_col]
                key_backward = tmpA[i_col, i_row]
                self.edges_repo[key_forward].append(i)
                assert self.edges_directions[key_forward] == forward_edge
                # self.edges_directions[key_forward].append(forward_edge)
                self.edges_repo[key_backward].append(i)
                # self.edges_directions[key_backward].append(backward_edge)
                assert self.edges_directions[key_backward] == backward_edge

            if i % 10000 == 0:
                print("\nEdges loaded", i)

        # create matrix (A[i,j] is an index of edge (i,j) in array of edges)
        print("\nConverting lil_matrix A to csr ...", end="")
        self.A = csr_matrix(tmpA)
        print("level done")
        del tmpA

        print("Converting edges_repo to list ...", end="")
        # data = [None]
        # subedges_counts = [0]
        # for i_key in range(1, key):
        #     value_set = self.edges_repo[i_key]
        #     # if len(value_list) > 1:
        #     #     print(i_key)
        #     data.append(value_set)
        #     subedges_counts.append(len(value_set))
        # self.edges_repo = data
        # the above can be replaced by
        self.edges_repo = np.array(
            list(self.edges_repo.values()), dtype=object)
        subedges_counts = [len(s) for s in self.edges_repo]
        # subedges_counts = [len(s) for s in np.nditer(self.edges_repo, flags=['refs_ok'], op_flags=['readonly'])]
        print("level done")

        print("Converting edges_directions to list ... ", end="")
        data = [None]
        for i_key in range(1, key):
            dir_list = [self.edges_directions[i_key]] * subedges_counts[i_key]
            data.append(dir_list)
        self.edges_directions = np.array(data, dtype=object)
        print("level done")

        print("Control check ... ", end="")
        for i_key in range(1, key):
            assert len(self.edges_repo[i_key]) == len(
                self.edges_directions[i_key])
        print("ok")

        print("Precalculate array of layer weights ... ", end="")
        self.e_layer_weight = self.layer_weights[self.e_types]
        print("ok")
        print("LightGraph is ready to use.")

        logging.info(f"Max intensity {self.e_intensities.max()}")

    @property
    def number_of_nodes(self):
        return self.num_nodes

    def get_nodes(self, layer):
        """ returns numpy array of nodes that posses at least one edge of the given layer"""
        sources = self.e_source[self.e_types == layer]
        dests = self.e_dest[self.e_types == layer]
        return np.union1d(sources, dests)

    def get_edges_nodes(self, edges, edges_dirs):
        """ returns source and dest nodes numbers (not ids)
        two np arrays - source and dest 
        the position corresponds to position in given array with edges 
        """
        sources = self.e_source[edges]
        dests = self.e_dest[edges]
        # sources, dests numpy vectors on nodes
        # edges_dirs - bool vector
        # if True take source if False take dest
        flags = edges_dirs
        # print(edges_dirs)
        source_nodes = sources * flags + dests * (1 - flags)
        dest_nodes = sources * (1 - flags) + dests * flags
        return source_nodes, dest_nodes

        #    def get_edges_subset(self, source_flags, dest_flags):
        #        active_subset = self.A[source_flags == 1, :][:, dest_flags == 1]
        #        edge_lists = [self.edges_repo[key] for key in active_subset.data]
        #        return subset, sum(edge_lists, [])

    def get_edges(self, source_flags, dest_flags, dirs=True):
        """
        Returns all edges between two sets of node.
        The first set is given by source_flags, the second by dest_flags (vectors 0/1).
        returns array of edge numbers + array of directions 
        """
        active_subset = self.A[source_flags == 1, :][:, dest_flags == 1]
        active_edges_indices = active_subset.data
        if len(active_edges_indices) == 0:
            return np.array([]), np.array([])
        edge_lists = self.edges_repo[active_edges_indices]
        result = np.array(concat_lists(edge_lists))
        if dirs:
            dirs_lists = self.edges_directions[active_edges_indices]
            result_dirs = np.array(concat_lists(dirs_lists), dtype=bool)
            return result, result_dirs
        return result

    def get_nodes_edges(self, nodes):
        """
        Get all edges adjacent to the given nodes.
        Returns list.
        """
        if len(nodes) == 0:
            return []
        active_subset = self.A[nodes]
        active_edges_indices = active_subset.data
        if len(active_edges_indices) == 0:
            logging.warning(f"Warning: no edges for nodes  {nodes}")
            return []
        edge_lists = self.edges_repo[active_edges_indices]
        result = concat_lists(edge_lists)
        return result

    def get_nodes_edges_on_layers(self, nodes, layers):
        """
        Same as get_node_edges, but only edges from list of layers are taken. 
        layers ... list of allowed layers 
        returns list
        """

        edges = self.get_nodes_edges(nodes)
        if len(edges) == 0:
            return edges
        edges_layers = self.e_types[edges]
        selected_edges = np.isin(edges_layers, layers)
        # todo: list or convert to numpy array?
        return [x
                for i, x in enumerate(edges)
                if selected_edges[i]
                ]

    def switch_off_edges(self, edges):
        """ Hard switch off of edges. Does not influence quarantine. """
        assert type(edges) == list
        self.e_active[edges] = False

    def switch_on_edges(self, edges):
        """ Opposite of switch off. Does not influence quarantine. """
        assert type(edges) == list
        self.e_active[edges] = True

    def get_all_edges_probs(self):
        """
        Returns array of probabilities of all edges.
        """
        # probs = self.e_probs.copy()
        # invalid = self.e_valid != 2
        # probs[invalid] =  self.e_valid[invalid]
        # weights = self.layer_weights[self.e_types]
        # probs[self.e_active == False] = 0
        # #        return ne.evaluate("probs * weights")
        # return probs * weights
        return ne.evaluate("active * (e_probs * (e_valid == 2) + e_valid * (e_valid != 2)) * weights",
                           local_dict={
                               'active': self.e_active,
                               'e_probs': self.e_probs,
                               'e_valid': self.e_valid,
                               'weights': self.e_layer_weight
                           }
                           )

    def get_edges_probs(self, edges):
        """
        Returns array of corresponding probabilities.
        """
        assert type(edges) == np.ndarray
        layer_types = self.e_types[edges]
        probs = self.e_probs[edges] * (self.e_valid[edges] == 2)
        probs += self.e_valid[edges] * (self.e_valid[edges] != 2)
        weights = self.e_layer_weight[edges]
        return self.e_active[edges] * probs * weights

    def get_edges_intensities(self, edges):
        """
        Returns array of corresponding probabilities.
        """
        assert type(edges) == np.ndarray
        return self.e_intensities[edges]

    # these methods work only with hodonin's layers
    def is_super_edge(self, edges):
        assert type(edges) == np.ndarray
        etypes = self.e_types[edges]
        return etypes >= 33

    def is_family_edge(self, edges):
        assert type(edges) == np.ndarray
        etypes = self.e_types[edges]
        return np.logical_or(etypes == 1, etypes == 2)

    def is_class_edge(self, edges, all=True):
        assert type(edges) == np.ndarray
        etypes = self.e_types[edges]
        if all:
            # all schools
            return np.logical_and(etypes >= 4, etypes <= 11)
        else:
            # excepty high and higher elementary
            return np.logical_and(etypes >= 4, etypes <= 7)

    def is_pub_edge(self, edges):
        assert type(edges) == np.ndarray
        etypes = self.e_types[edges]
        # pubs
        return np.logical_or(etypes == 20,
                             np.logical_or(etypes == 28, etypes == 29))

    def modify_layers_for_nodes(self, node_id_list, what_by_what):
        """ 
        Changes probs of edges adjacent to given nodes.
        :node_id_list  list of given nodes
        :what_by_what  dictionary {layer: multiplication_coefficient}
        """

        if self.is_quarantined is None:
            self.is_quarantined = np.zeros(self.number_of_nodes, dtype=int)

        self.is_quarantined[node_id_list] += 1

        if not what_by_what:
            raise ValueError("what_by_what missing or empty")

        relevant_edges = np.unique(self.get_nodes_edges(node_id_list))

        if len(relevant_edges) == 0:
            return

        # select edges to change (those who are not in quarantine)
        valid = self.e_valid[relevant_edges]
        relevant_edges = relevant_edges[valid == 2]
        edges_types = self.e_types[relevant_edges]

        logging.info(f"DBG edges that goes to quarantinexs {len(relevant_edges)}")
        # print(edges_types)
        for layer_type, coef in what_by_what.items():
            # print(layer_type)
            edges_on_this_layer = relevant_edges[edges_types == layer_type]
            # modify their probabilities
            self.e_valid[edges_on_this_layer] = self.e_probs[edges_on_this_layer] * coef
            # use out in clip to work inplace
            np.clip(self.e_valid[edges_on_this_layer], 0.0,
                    1.0, out=self.e_valid[edges_on_this_layer])

    def recover_edges_for_nodes(self, release):
        """
        Recover original probabilites for edges adjacent to given nodes except edges where one node is still quarantined.
        :release list of given nodes
        """

        self.is_quarantined[release] -= 1
        assert np.all(self.is_quarantined >= 0)
        no_quara = self.is_quarantined[release] == 0
        release = release[no_quara]

        if len(release) == 0:
            return
        relevant_edges = np.unique(self.get_nodes_edges(release))
        if len(relevant_edges) == 0:
            logging.warning("Warning: recovering nodes with no edges")
            return
        # from source and dest nodes select those who are not in quarantine
        source_nodes = self.e_source[relevant_edges]
        dest_nodes = self.e_dest[relevant_edges]

        is_quarrantined_source = self.is_quarantined[source_nodes]
        is_quarrantined_dest = self.is_quarantined[dest_nodes]

        # recover only edges where both nodes are free
        relevant_edges = relevant_edges[np.logical_not(
            np.logical_or(is_quarrantined_source, is_quarrantined_dest))]

        # recover probs
        self.e_valid[relevant_edges] = 2

    def final_adjacency_matrix(self):
        """ just for backward compatibility """
        return self

    def get_layer_for_edge(self, e):
        return self.e_types[e]

    def set_layer_weights(self, weights):
        logging.info(f"DBG Updating layer weights {weights}")
        np.copyto(self.layer_weights, weights)
        #    print(self.layer_weights[i])
        self.e_layer_weight = self.layer_weights[self.e_types]

        logging.info(f"DBG new weigths {self.layer_weights}")

    def close_layers(self, list_of_layers, coefs=None):
        print(f"Closing {list_of_layers}")
        for idx, name in enumerate(list_of_layers):
            i = self.layer_name.index(name)
            self.layer_weights[i] = 0 if not coefs else coefs[idx]
        print(self.layer_weights)

    def copy(self):
        """
        Optimized version of shallow/deepcopy of self.
        Since most fields never change between runs, we do shallow copies on them.
        :return: Shallow/deep copy of self.
        """
        heavy_fields = ['e_valid', 'layer_weights',
                        'e_active', 'e_layer_weight']
        new = copy(self)
        for key in heavy_fields:
            field = getattr(self, key)
            setattr(new, key, field.copy())
        new.is_quarantined = None
        new.e_valid.fill(2)
        new.e_active.fill(True)
        return new
