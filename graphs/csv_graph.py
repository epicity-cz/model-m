import pandas as pd
import networkx as nx
from graph_gen import GraphGenerator
# from pprint import pprint


def magic_formula(edge_info, layer_info):
    prob_no_contact = 1

    cummulated_edge_info = {
    }
    for etype, w in edge_info.items():
        if etype[0] in cummulated_edge_info:
            cummulated_edge_info[etype[0]].append(w["weight_on_layer"])
        else:
            cummulated_edge_info[etype[0]] = [w["weight_on_layer"]]

    cummulated_edge_info = {
        key: sum(value)
        for key, value in cummulated_edge_info.items()
    }

    for edge_type, edge_intensity in cummulated_edge_info.items():
        w_edge_type = layer_info[edge_type]  # only type (no subtype)
        no_contact_on_edge = 1.0 - (w_edge_type * edge_intensity)
        prob_no_contact *= no_contact_on_edge
    return 1 - prob_no_contact


class CSVGraph(GraphGenerator):

    def __init__(self, path_to_nodes='nodes.csv', path_to_edges='edges.csv', path_to_layers='etypes.csv', magic_formula=magic_formula, **kwargs):

        # zatim to nevolejme, nechcem multigraf
        #    super().__init__(**kwargs)

        self.G = nx.Graph()

        csv_hacking = {'na_values': 'undef', 'skipinitialspace': True}
        nodes = pd.read_csv(path_to_nodes, **csv_hacking)
        edges = pd.read_csv(path_to_edges, **csv_hacking)
        layers = pd.read_csv(path_to_layers, **csv_hacking)

        # TODO: nebude treba
        indexNames = edges[edges['vertex1'] == edges['vertex2']].index
        if len(indexNames):
            print("Warning: dropping self edges!!!!")
            edges.drop(indexNames, inplace=True)

        #        print(layers)
        # fill the layers
#        layer_names = tuple(zip(layers.loc('id'), layers.loc('id2')))
        layers_to_add = layers.to_dict('list')
        self.layer_names = layers_to_add['id']
#        print(layers_names)
        self.layer_probs = layers_to_add['weight']

        self.G.graph['layer_names'] = self.layer_names
        self.G.graph['layer_probs'] = self.layer_probs

        # fill the nodes
        nodes_to_add = nodes.to_dict('records')
        idx_s = list(range(0, len(nodes_to_add)))
        self.G.add_nodes_from(zip(idx_s, nodes_to_add))

        # fill the edges
#        pprint(edges)
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
        edges.drop(columns=["vertex1", "vertex2",
                            "type", "subtype"], inplace=True)
        # TODO duplicity by tam mit nemeli!!!! hlidat
        edges.drop_duplicates(inplace=True)
        # print(edges)

        # multi_edges = pd.DataFrame()
        # multi_edges["e"] = edges["e"].drop_duplicates()
        # print(multi_edges)

        def group_func(subframe):
            if True:
                # print(subframe)
                edge_info = {}
                for type in subframe["t"]:
                    edge_info[type] = {
                        "weight_on_layer": subframe[subframe["t"] == type]["weight"].values[0]
                    }

                return edge_info
                # print(subframe)
                # print(subframe.groupby("type").groups.keys())
                # print(subframe.groupby("type").groups.values())
                # for k, index in subframe.groupby("type").groups.items():
                #     edge_info[k] = subframe.iloc[int(index)].to_dict()
                #     print(edge_info)


#        multi_edges["edge data"] = multi_edges.groupby("e")
        g = edges.groupby("e").groups
        froms = []
        tos = []
        data = []

        for k, v in g.items():
            # k ... (x, y)   v .. Int64Index
            edge_info = group_func(edges.loc[v])
            froms.append(k[0])
            tos.append(k[1])
            if magic_formula:
                w = magic_formula(edge_info, dict(
                    zip(self.layer_names, self.layer_probs)))
            data.append({"info": edge_info, "weight": w})

        # edges_to_add = edges.to_dict('list')
        # pprint(edges_to_add)
        # froms = edges_to_add['vertex1']
        # tos = edges_to_add['vertex2']
        # datas = [{k: v for k, v in d.items() if k != 'vertex1' and k != 'vertex2'} for d in edges.to_dict('records')]

        self.G.add_edges_from(zip(froms, tos, data))

    def __str__(self):
        return "\n".join([str(e) for e in self.G.edges(data=True)])

    def modify_layers_for_node(self, node_id, what_by_what):
        """ changes edges' weights """

        if not what_by_what:
            return

        if not self.G.has_node(node_id):
            print(f"Warning: modify_layer_for_node called for nonexisting node_id {node_id}")
            return

        for u, v, d in self.G.edges([node_id], data=True):
            #            print(u, v, d)
            layer_info = d["info"]
            for layer_type in what_by_what:
                layer_keys = [(t, s) for (t, s) in layer_info
                              if t == layer_type]
                for k in layer_keys:
                    layer_info[k]['weight_on_layer'] = min(
                        what_by_what[layer_type] *
                        layer_info[k]['weight_on_layer'],
                        1.0
                    )
            d["info"] = layer_info
            d["weight"] = magic_formula(layer_info, dict(
                zip(self.layer_names, self.layer_probs)))

            # layer_label = d["type"]
            # if layer_label in what_by_what:
            #     self.G.edges[(u, v, k)
            #                  ]['weight'] = min(self.G.edges[(u, v, k)]['weight'] * what_by_what[layer_label], 1.0)
