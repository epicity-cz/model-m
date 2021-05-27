import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix as sparse_matrix
from scipy.sparse import lil_matrix
from pprint import pprint
from sparse_utils import multiply_col, multiply_row, multiply_zeros_as_ones


def sum_sparse(m):
    x = sparse_matrix(m[0].shape)
    for a in m:
        x += a
    return x


def magic_formula(m, probs):
    assert len(m) == len(probs)
    N = m[0].shape[0]  # m must have at least one member

    prob_no_contact = sparse_matrix((N, N))
    for a, p in zip(m, probs):
        if len(a.data) == 0 or p == 0:
            continue
        pa = a.multiply(p)
        not_a = pa
        not_a.data = 1.0 - not_a.data
        prob_no_contact = multiply_zeros_as_ones(prob_no_contact, not_a)
        del pa
    prob_of_contact = prob_no_contact
    prob_of_contact.data = 1.0 - prob_no_contact.data
    return prob_of_contact


class LightGraph():

    def __init__(self,
                 path_to_nodes='nodes.csv',
                 path_to_edges='edges.csv',
                 path_to_layers='etypes.csv'
                 ):

        csv_hacking = {'na_values': 'undef', 'skipinitialspace': True}
        nodes = pd.read_csv(path_to_nodes, index_col="type", **csv_hacking)
        edges = pd.read_csv(path_to_edges, **csv_hacking)
        layers = pd.read_csv(path_to_layers, **csv_hacking)

        # TODO: nebude treba
        indexNames = edges[edges['vertex1'] == edges['vertex2']].index
        if len(indexNames):
            print("Warning: dropping self edges!!!!")
            edges.drop(indexNames, inplace=True)

        layers_to_add = layers.to_dict('list')
        self.layer_names = layers_to_add['id']
        self.layer_probs = layers_to_add['weight']

        # fill the nodes
        cat_columns = nodes.select_dtypes(['object']).columns
        nodes[cat_columns] = nodes[cat_columns].apply(
            lambda x: x.astype('category'))

        cat_table = {
            col: list(nodes[col].cat.categories)
            for col in cat_columns
        }
        #print(cat_table)

        nodes[cat_columns] = nodes[cat_columns].apply(
            lambda x: x.cat.codes)
        #pprint(nodes)

        # just test of conversion back
        # print(cat_columns)
        # for col in list(cat_columns):
        #     nodes[[col]] = nodes[[col]].apply(
        #         lambda x: pd.Categorical.from_codes(
        #             x, categories=cat_table[col])
        #     )
        # pprint(nodes)

        for col in nodes.columns:
            setattr(self, col, np.array(nodes[col]))

        self.nodes = np.array(nodes.index)
        self.num_nodes = len(self.nodes)

        # fill the edges
        edges["e"] = edges.apply(
            lambda row: (
                (int(row.vertex1), int(row.vertex2))
                if int(row.vertex1) < int(row.vertex2)
                else (int(row.vertex2), int(row.vertex1))
            ),
            axis=1
        )
        edges["t"] = edges.apply(
            lambda row: (int(row.type), int(row.subtype)),
            axis=1
        )
        edges.drop(columns=["type", "subtype"], inplace=True)
        # TODO duplicity by tam mit nemeli!!!! hlidat
        edges.drop_duplicates(inplace=True, subset=["t", "e"])
#        print(edges)

        g = edges.groupby("t").groups
        # for k, v in g.items():
        #     print(k, v)

        def create_matrix(index):
            my_edges = edges.loc[index]
            # print(my_edges)
            # TODO: what to do with symmetry?
            rows = pd.concat([my_edges["vertex1"], my_edges["vertex2"]])
            cols = pd.concat([my_edges["vertex2"], my_edges["vertex1"]])
            data = pd.concat([my_edges["weight"], my_edges["weight"]])
            a = sparse_matrix((data, (rows, cols)),
                              shape=(self.num_nodes, self.num_nodes))
            # print("matrix", a)
            return a

        # misto dictu udelat neco jako nd.array ?
        # pocet vrstev x podvrstvy x NxN sparse matice
        self.A_layered = {
            k: create_matrix(v)
            for k, v in g.items()
        }
        self.A = self.computeA()
        #        print(self.A)

    def number_of_nodes(self):
        return self.num_nodes

    def computeA(self):
        # compute weights
        # cumullate over layers
        A_high_level = {}

        for l in self.layer_names:
            sub_matrices = [
                a
                for k, a in self.A_layered.items()
                if k[0] == l
            ]
            if sub_matrices:
                A_high_level[l] = sum_sparse(sub_matrices)
                del sub_matrices
            else:
                A_high_level[l] = sparse_matrix(
                    (self.num_nodes, self.num_nodes))

        #        print(A_high_level)
        A = magic_formula(
            [A_high_level[l] for l in self.layer_names],
            self.layer_probs
        )
        del A_high_level
        return A

    def __str__(self):
        return "\n".join([str(e) for e in self.G.edges(data=True)])

    def final_adjacency_matrix(self):
        return self.A


    def modify_layers_for_node(self, node_id, what_by_what):
        """ changes edges' weights """

        if not what_by_what:
            return

        if not node_id in self.nodes:
            print(f"Warning: modify_layer_for_node called for nonexisting node_id {node_id}")
            return

        for layer_type, weight in what_by_what.items():
            matrices_to_change = [
                m
                for (layer, sublayer), m in self.A_layered.items()
                if layer == layer_type and node_id in m.indices
            ]
            for m in matrices_to_change:
                # diagonal is empty!
                multiply_col(m, node_id, weight, trunc=True)
                multiply_row(m, node_id, weight, trunc=True)
        self.A = self.computeA()
