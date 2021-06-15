# -*- coding: utf-8 -*-
import networkx as nx
import numpy as np
import scipy.stats as stats
import pandas as pd
from dataclasses import dataclass

#id,sex, age, ecactivity, worktype, commutetime, town, x, y, apartment
#id,type,code,name


#@dataclass(eq=True)
@dataclass
class NodeObj:
    __slots__ = ['id', 'sex', 'age', 'ecactivity', 'worktype', 'comutetime', 
                 'town', 'x', 'y', 'apartment', 'type', 'code', 'name', 'internal']
    id: int
    sex: bool
    age: int
    ecactivity: str
    worktype: str
    comutetime: str
    town: str
    x: float
    y: float
    apartment: int
    type: int # these three are for external nodes
    code: int
    name: str
    internal: bool # true - internal, false - external node        

    def __hash__(self): return hash(self.id)    

#@dataclass(eq=True)
@dataclass()
class EdgeObj:
    __slots__ = ['layer','sublayer', 'probability', 'intensity']
    layer: int
    sublayer: int
    probability: float
    intensity: float

    def __hash__(self): return hash([self.layer, self.sublayer])    


class NewGraph:

    def __init__(self, random_seed=None):
        self.G = nx.MultiGraph()
        if random_seed:
            np.random.seed(random_seed)

    def read_csv(self, prefix='.', path_to_nodes='p.csv', path_to_external='e.csv', path_to_layers='etypes.csv', path_to_edges='edges.csv'):

        csv_hacking = {'na_values': 'undef', 'skipinitialspace': True}
        nodes = pd.read_csv(prefix + path_to_nodes, **csv_hacking)
        ext_nodes = pd.read_csv(prefix + path_to_external, **csv_hacking)
        edges = pd.read_csv(prefix + path_to_edges, **csv_hacking)
        layers = pd.read_csv(prefix + path_to_layers, **csv_hacking)

        # layer names, ids and weights go to graph
        layers_to_add = layers.to_dict('list')

        self.G.graph['layer_ids'] = layers_to_add['id']
        self.G.graph['layer_names'] = layers_to_add['name']
        self.G.graph['layer_weights'] = layers_to_add['weight']

        # nodes and ext_nodes go to nodes
        # FIXME: ext nodes are ignored now!
        
        #fill NodeObjs from nodes (and ext_nodes) 
        nodes.replace({'M': True, 'F': False}, inplace=True)
        NodeObjs = [ NodeObj(row.id, row.sex, row.age, row.ecactivity, row.worktype, row.commutetime,
                             row.town, row.x, row.y, row.apartment, 0, 0, '', True) 
                    for row in nodes.itertuples() ]
        ExtNodeObjs = [NodeObj(row.id, False, 0, '', '', '', '', 0, 0, 0, 
                               row.type, row.code, row.name, False) 
                    for row in ext_nodes.itertuples() ]
        NodeObjs.extend(ExtNodeObjs)
        
#        nodes_to_add = nodes.to_dict('records')
#        idx_s = list(range(0, len(NodeObjs)))
#        self.G.add_nodes_from(zip(idx_s, nodes_to_add))
#        self.G.add_nodes_from(NodeObjs)
        
        # fill edges to a graph
        for row in edges.itertuples(): 
            e =  EdgeObj(row.layer, row.sublayer, row.probability, row.intensity) 
            self.G.add_edge(row.vertex1, row.vertex2, object=e)
        
    
    def read_pickle(self, path_to_pickle='graph.pickle'):
        self.G = nx.read_gpickle(path_to_pickle)
        
    def write_pickle(self, path_to_pickle='graph.pickle'):
        nx.write_gpickle(self.G, path_to_pickle)
        
    def close_layers(self, list_of_layers, coefs=None):
        for idx, name in enumerate(list_of_layers):
            print(f"Closing {name}")
            i = self.G.graph["layer_names"].index(name)
            self.G.graph["layer_probs"][i] = 0 if not coefs else coefs[idx]
        #self.A_valid = False
        
    @property
    def nodes(self):
        return self.G.nodes

    def number_of_nodes(self):
        return self.G.number_of_nodes()

    def get_edges_for_node(self, node_id):
        """ changes edges' weights """
        if not self.G.has_node(node_id):
            print(f"Warning: modify_layer_for_node called for nonexisting node_id {node_id}")
            return
        return self.G.edges([node_id], data=True, keys=True)

    def modify_layers_for_nodes(self, node_id_list, what_by_what, is_quarrantined=None):
        """ changes edges' weights """
        if not what_by_what:
            return
        changed = set()
        # keep the original list (it is modified in the cycle)
        for u, v, k, d in self.G.edges(set(node_id_list), data=True, keys=True):
            if is_quarrantined is not None and (is_quarrantined[u] > 0 or is_quarrantined[v] > 0):
                # edge is already quarrantined
                continue
            layer_label = d["type"]
            if layer_label in what_by_what:
                assert (u, v, k, d.layer, d.sublayer) not in self.quarantined_edges, \
                    f"edge {(u, v, k, d['type'], d['subtype'])} is already quaratined, {is_quarrantined[u]}{is_quarrantined[v]}"
                s, e = (u, v) if u < v else (v, u)
                self.quarantined_edges[(
                    s, e, k, d.layer, d.sublayer)] = d.probability
                self.G.edges[(u, v, k)
                             ].probability = min(self.G.edges[(u, v, k)].probability * what_by_what[layer_label], 1.0)
                changed.add((u, v))

    def recover_edges_for_nodes(self, release, normal_life, is_quarrantined):
        changed = set()
        for u, v, k, d in self.G.edges(release, data=True, keys=True):
            if is_quarrantined[u] or is_quarrantined[v]:
                # still one of nodes is in quarrantine
                continue
            e_type = d.layer
            e_subtype = d.sublayer
            s, e = (u, v) if u < v else (v, u)
            self.G.edges[(u, v, k)].probability = self.quarantined_edges[(
                s, e, k, e_type, e_subtype)]
            del self.quarantined_edges[(
                s, e, k, e_type, e_subtype)]
            changed.add((s, e))

    def get_layers_info(self):
        """ returns dictionary of layer names and probabilities """
        return dict(zip(self.G.graph["layer_names"], self.G.graph["layer_probs"]))

