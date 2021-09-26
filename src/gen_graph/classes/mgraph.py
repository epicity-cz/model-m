import networkx as nx

from lang.mytypes import AutoName, auto
from params.cal_param import CalParamIdx
from utils.csv_utils import CsvWriter


class LinkParams(AutoName):
    # @formatter:off
    LAYER       = auto()
    SUBLAYER    = auto()
    INTENSITY   = auto()
    PROBABILITY = auto()
    W_LAYER     = auto()
    W_PARAM     = auto()
    # @formatter:on


class Link:
    def __init__(self, layer, sublayer, probability, intensity, w_layer, w_param):
        self.layer = layer
        self.sublayer = sublayer
        self.probability = probability
        self.intensity = intensity
        self.w_layer = w_layer
        self.w_param = w_param

    def weight(self, parameters):
        return self.intensity * self.probability_real(parameters)

    def probability_real(self, parameters):
        return self.probability * CalParamIdx.calculate(self.w_layer, parameters, self.w_param)

    def dict(self):
        return {
            LinkParams.LAYER.name: self.layer,
            LinkParams.SUBLAYER.name: self.sublayer,
            LinkParams.PROBABILITY.name: self.probability,
            LinkParams.INTENSITY.name: self.intensity,
            LinkParams.W_LAYER.name: self.w_layer,
            LinkParams.W_PARAM.name: self.w_param,
        }

    @classmethod
    def from_dict(cls, dd):
        return Link(
            dd[LinkParams.LAYER.name],
            dd[LinkParams.SUBLAYER.name],
            dd[LinkParams.PROBABILITY.name],
            dd[LinkParams.INTENSITY.name],
            dd[LinkParams.W_LAYER.name],
            dd[LinkParams.W_PARAM.name],
        )


class MGraph(nx.MultiDiGraph):
    def links(self, nbunch=None):
        if nbunch:
            if not nbunch in self.nodes:
                return
            it_edges = self.edges(nbunch, data=True)
        else:
            it_edges = self.edges(data=True)
        for (u, v, edge) in it_edges:
            link = Link.from_dict(edge)  # l
            yield (u, v, link)

    def add_link(self, p1, p2, layer, sublayer, probability: float, w_layer: int, w_param=None, intensity=None):
        return self.add_edge(p1, p2, key=layer,
                             **Link(layer, sublayer, probability, intensity, w_layer, w_param).dict())

    def list_edges(self, filename, scale, parameters):
        with CsvWriter(filename) as f:
            f.writerow(['layer', 'sublayer', 'vertex1', 'vertex2', 'probability', 'intensity'])
            for (p1, p2, link) in self.links():
                prob = link.probability_real(parameters)
                if scale * link.intensity > 1 or prob > 1:
                    raise Exception("Node with excess weight")
                f.writerow([link.layer.value,
                            link.sublayer,
                            p1.id,
                            p2.id,
                            prob,
                            link.intensity])

    def is_edge(self, p1, p2, layer):
        return self.has_edge(p1, p2, layer)
