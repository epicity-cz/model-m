from _collections import defaultdict

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from config.base_config import BaseConfig
from constants import constants
from lang.mytypes import Tuple, Iterator, Dict, Any, List
from params.cal_param import CalParamIdx
from utils.math_prob import yes_no

Town = Any
WaysType = Dict[Town, Dict[Town, List[Town]]]
ConnectionsType = Iterator[Tuple[Town, Town, float]]


class TravelInfo:
    travels: nx.MultiGraph
    towns: List[Town]
    config: BaseConfig
    region_center: Town

    def __init__(self, config: BaseConfig, towns: List[Town]):
        self.travels = nx.MultiGraph()
        self.towns = towns
        self.travels.add_nodes_from(towns)
        self.region_center = None
        for town in towns:
            if not self.region_center or self.region_center.count_persons() < town.count_persons():
                self.region_center = town
        self.config = config
        tra = np.array(config.TRAVEL_RATIOS_ALL)
        self.ratios = tra[2, :] / np.sum(tra, axis=0)

    # generator ConnectionsType
    def generate_connections(self):
        pass

    def get_ratio(self, type: constants.TravelType):
        return self.ratios[constants.TRAVEL_TYPE_2_RATIO[type]]

    def add_export_travel(self, type: constants.TravelType, person, probability: float):
        self.add_travel(type, person.town, self.region_center, person, probability)

    def add_travel(self, type: constants.TravelType, src, dest, person, probability: float):
        if yes_no(self.get_ratio(type)):
            self.travels.add_edge(src, dest, person=person, probability=probability, type=type)

    def transport(self, renv):
        ways = self.generate_ways(self.generate_connections())
        self.generate_transport(renv, ways)

    def show_graph(self, connections: ConnectionsType, title='', filename=None):
        xs = []
        ys = []
        plt.figure(figsize=(8, 6), dpi=300)
        plt.scatter(xs, ys, marker='.')
        bbox_props = dict(boxstyle="round", fc="w", ec="0.5", alpha=0.9)

        for idx, town in enumerate(self.towns):
            xs.append(-town.location[1])
            ys.append(-town.location[0])
            plt.text(xs[idx], ys[idx], town.name, ha="center", va="center", size=8, bbox=bbox_props)

        ax = plt.gca()
        ax.set_aspect('equal')
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)

        for t1, t2, distance in connections:
            xs = [-t1.location[1], -t2.location[1]]
            ys = [-t1.location[0], -t2.location[0]]
            plt.plot(xs, ys)

        plt.title(title)
        plt.tight_layout()
        if filename:
            plt.savefig(filename, bbox_inches='tight', pad_inches=0.1)
        plt.show()

    def generate_ways(self, connections) -> WaysType:
        g = nx.Graph()
        g.add_nodes_from(self.towns)
        g.add_edges_from([(t1, t2, {'weight': weight}) for t1, t2, weight in connections])
        return dict(nx.all_pairs_dijkstra_path(g, weight='weight'))

    def dump_ways(self, ways: WaysType):
        for src in ways:
            for dest in ways[src]:
                print(f"{src.name} => {dest.name}:", end='')
                print("->".join(tt.name for tt in ways[src][dest]))
            print()

    def generate_transport(self, renv, ways: WaysType):
        # section = {t1: {t2: [] for t2 in self.towns} for t1 in self.towns}
        sections = defaultdict(lambda: defaultdict(list))

        for n1, n2, par in self.travels.edges(data=True):
            last = None
            for node in ways[n1][n2]:
                if last is None:
                    last = node
                else:
                    sections[last][node].append(par)

        for key1 in sections:
            for key2 in sections[key1]:
                persons = []
                probabilities = []
                sum = 0
                for item in sections[key1][key2]:
                    persons.append(item['person'])
                    sum += item['probability']
                    probabilities.append(item['probability'])
                renv.mutual_contacts(persons, constants.Layer.PUBLIC_TRANSPORT, key1.id * 1000 + key2.id,
                                     CalParamIdx.ONE, 0.5,
                                     probabilities,
                                     0.25,
                                     min(3, len(persons)))
