import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, lil_matrix

# novej light graph


class LightGraph:

    def __init__(self, random_seed=None):
        if random_seed:
            np.random.seed(random_seed)
        self.edge_repo = None
        self.A = None
        self.invalids = None
        self.quarantined_probs_repo = {}

    def read_csv(self,
                 path_to_nodes='p.csv',
                 path_to_external='e.csv',
                 path_to_layers='etypes.csv',
                 path_to_edges='edges.csv'):

        csv_hacking = {'na_values': 'undef', 'skipinitialspace': True}
        nodes = pd.read_csv(path_to_nodes, **csv_hacking)
        edges = pd.read_csv(path_to_edges, **csv_hacking)
        layers = pd.read_csv(path_to_layers, **csv_hacking)
        external_nodes = pd.read_csv("../data/stabletown/e.csv", **csv_hacking)

        # layer names, ids and weights go to graph
        layers_to_add = layers.to_dict('list')

        self.layer_ids = layers_to_add['id']
        self.layer_name = layers_to_add['name']
        self.layer_weights = np.array(layers_to_add['weight'])

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
        if self.num_nodes > 65535:
            raise ValueError(
                "Number of nodes too high (we are using unit16, change it to unit32 for higher numbers of nodes.")

        self.ignored = set(external_nodes["id"])

        # edges
        # drop self edges
        indexNames = edges[edges['vertex1'] == edges['vertex2']].index
        if len(indexNames):
            print("Warning: dropping self edges!!!!")
            edges.drop(indexNames, inplace=True)

        # fill edges to a graph
        n_edges = len(edges)
        # edges data"
        self.e_types = np.empty(n_edges, dtype="uint8")
        self.e_subtypes = np.empty(n_edges, dtype="int16")
        self.e_probs = np.empty(n_edges, dtype="float32")
        self.e_intensities = np.empty(n_edges, dtype="float32")
        self.e_source = np.empty(n_edges, dtype="uint16")
        self.e_dest = np.empty(n_edges, dtype="uint16")
        # if value == 2 than is valid, other numbers prob in quarantine
        self.e_valid = 2 * np.ones(n_edges, dtype="float32")
        # edges repo
        self.edges_repo = {
            0: None
        }
        self.edges_directions = {
            0: None
        }
        key = 1
        # working matrix
        tmpA = lil_matrix((self.num_nodes, self.num_nodes), dtype="uint32")

        forward_edge = True
        backward_edge = False

        # fill data and get indicies
        for i, row in enumerate(edges.itertuples()):
            self.e_types[i] = row.layer
            self.e_subtypes[i] = row.sublayer
            self.e_probs[i] = row.probability
            self.e_intensities[i] = row.intensity

            if row.vertex1 in self.ignored or row.vertex2 in self.ignored:
                continue

            try:
                i_row = np.where(self.nodes_id == row.vertex1)[0][0]
                i_col = np.where(self.nodes_id == row.vertex2)[0][0]
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
                self.edges_repo[key +
                                1], self.edges_directions[key+1] = [i], backward_edge
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

            if i % 1000 == 0:
                print("\nEdges loaded", i)

        # create matrix (A[i,j] is an index of edge (i,j) in array of edges)
        print("\nConverting lil_matrix A to csr ...", end="")
        self.A = csr_matrix(tmpA)
        print("level done")
        del tmpA

        print("Converting edges_repo to numpy array ...", end="")
        data = [0]
        subedges_counts = [0, 0]
        for i_key in range(1, key):
            value_list = self.edges_repo[i_key]
            # if len(value_list) > 1:
            #     print(i_key)
            data.extend(value_list) 
            subedges_counts.append(len(value_list))
        self.edges_repo = np.array(data, dtype="uint32")
        self.edges_repo_indptr = np.cumsum(subedges_counts) 
        #self.edges_repo = np.array(data, dtype="uint32")
        #        self.edges_e
        print("level done")

        print("Converting edges_directions to numpy bool array ... ", end="")
        data = [False]
        for i_key in range(1, key):
            dir_list = [self.edges_directions[i_key]] * subedges_counts[i_key]
            data.extend(dir_list)
        self.edges_directions = np.array(data, dtype="bool")
        print("level done")

        print("Control check ... ", end="")
        for i_key in range(1, key):
            assert len(self.edges_repo[i_key]) == len(
                self.edges_directions[i_key])
        print("ok")
        print("LightGraph is ready to use.")



    def get_edges_by_key(self, keys):
        edge_indices = []
        for k in keys:
            row_indices.extend(range(self.edges_repo_indtr[k], self.edges_repo_indtr[k+1]))
        return self.edges_repo(edge_indices) 

    @property
    def number_of_nodes(self):
        return self.num_nodes

    def get_edges_nodes(self, edges, edges_dirs):
        """ returns source and dest nodes numbers (not ids)
        WARNING: NOT IDs
        """
        sources = self.e_source[edges]
        dests = self.e_dest[edges]
        # sources, dests numpy vectors on node_ids
        # edges_dirs - bool vector
        # if True take source if False take dest
        flags = edges_dirs
        source_nodes = sources * flags + dests * (1 - flags)
        dest_nodes = sources * (1 - flags) + dests * flags
        return source_nodes, dest_nodes

        #    def get_edges_subset(self, source_flags, dest_flags):
        #        active_subset = self.A[source_flags == 1, :][:, dest_flags == 1]
        #        edge_lists = [self.edges_repo[key] for key in active_subset.data]
        #        return subset, sum(edge_lists, [])

    def get_edges(self, source_flags, dest_flags, dirs=True):
        active_subset = self.A[source_flags == 1, :][:, dest_flags == 1]
        active_edges_indices = active_subset.data
        if len(active_edges_indices) == 0:
            return np.array([]), np.array([])
        # edge_lists = self.edges_repo[active_edges_indices]
        edge_lists = self.get_edges_by_key(active_edges_indices)
        result = np.concatenate(edge_lists)
        if dirs:
            dirs_lists = self.edges_directions[active_edges_indices]
            result_dirs = np.concatenate(dirs_lists)
            return result, result_dirs
        return result

    def get_nodes_edges(self, nodes):
        active_subset = self.A[nodes]
        active_edges_indices = active_subset.data
        if len(active_edges_indices) == 0:
            print("Warning: no edges for nodes", nodes)
            return np.array([])
        edge_lists = self.edges_repo[active_edges_indices]
        result = np.concatenate(edge_lists)
        return result

    def get_edges_probs(self, edges):
        assert type(edges) == np.ndarray
        # multiply by layer weight! TODO
        layer_types = self.e_types[edges]
        probs = self.e_probs[edges] * (self.e_valid[edges] == 2)
        probs += self.e_valid[edges] * (self.e_valid[edges] != 2)
        weights = self.layer_weights[layer_types]
        return probs * weights

    def get_edges_intensities(self, edges):
        assert type(edges) == np.ndarray
        return self.e_intensities[edges]

    def modify_layers_for_nodes(self, node_id_list, what_by_what, is_quarrantined=None):
        """ changes edges' weights """

        if not what_by_what:
            return
        relevant_edges = np.unique(self.get_nodes_edges(node_id_list))
        valid = self.e_valid[relevant_edges]
        relevant_edges = relevant_edges[valid == 2]
        edges_types = self.e_types[relevant_edges]

        print("relevant edges", len(relevant_edges))
        # print(edges_types)

        for layer_type, coef in what_by_what.items():
            # print(layer_type)
            edges_on_this_layer = relevant_edges[edges_types == layer_type]
            # modify their probabilities
            self.e_valid[edges_on_this_layer] = self.e_probs[edges_on_this_layer] * coef
            # use out in clip to work inplace
            np.clip(self.e_valid[edges_on_this_layer], 0.0,
                    1.0, out=self.e_valid[edges_on_this_layer])

    def recover_edges_for_nodes(self, release, normal_life, is_quarrantined):

        relevant_edges = np.unique(self.get_nodes_edges(release))
        if len(relevant_edges) == 0:
            print("Warning: recovering nodes with no edges")
            return
        # from source and dest nodes select those who are not in quarantine
        source_nodes = self.e_source[relevant_edges]
        dest_nodes = self.e_dest[relevant_edges]

        is_quarrantined_source = is_quarrantined[source_nodes]
        is_quarrantined_dest = is_quarrantined[dest_nodes]

        # leave only edges where both nodes are free
        relevant_edges = relevant_edges[np.logical_not(
            np.logical_or(is_quarrantined_source, is_quarrantined_dest))]

        # recover probs
        self.e_valid[relevant_edges] = 2

    def final_adjacency_matrix(self):
        """ just for backward compatibility """
        return self

    def get_layer_for_edge(self, e):
        return self.e_types[e]

    def close_layers(self, list_of_layers, coefs=None):
        for idx, name in enumerate(list_of_layers):
            print(f"Closing {name}")
            i = self.layer_name.index(name)
            self.layer_weights[i] = 0 if not coefs else coefs[idx]
