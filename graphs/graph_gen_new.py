# -*- coding: utf-8 -*-
import networkx as nx
import numpy as np
import scipy.stats as stats
import pandas as pd
from dataclasses import dataclass

#id,sex, age, ecactivity, worktype, commutetime, town, x, y, apartment
#id,type,code,name


@dataclass(frozen=True, eq=True)
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
    
@dataclass(frozen=True, eq=True)
class EdgeObj:
    __slots__ = ['layer','sublayer', 'probability', 'intensity']
    layer: int
    sublayer: int
    probability: float
    intensity: float


class NewGraphGenerator:

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
        G = nx.read_gpickle(path_to_pickle)
        
    def write_pickle(self, path_to_pickle='graph.pickle'):
        nx.write_gpickle(G, path_to_pickle)
        