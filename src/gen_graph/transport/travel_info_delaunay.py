import numpy as np
from scipy.spatial import Delaunay

from transport.travel_info import TravelInfo
from utils.geography import distance


class TravelInfoDelaunay(TravelInfo):
    orders = [(1, 2, 0),
              (0, 2, 1),
              (0, 1, 2)]

    def get_distance(self, i1, i2):
        if self.distances[i1, i2] == 0:
            t1, t2 = self.towns[i1], self.towns[i2]
            self.distances[i1, i2] = distance(t1.location, t2.location)
        return self.distances[i1, i2]

    def generate_connections(self):
        size = len(self.towns)
        locations = np.zeros([size, 2])
        self.distances = np.zeros([size, size])

        for idx, town in enumerate(self.towns):
            locations[idx, :] = town.location

        tri = Delaunay(locations)

        links = np.zeros([size, size])  # -1 ommit, 0 not yet, 1 add

        for sidx in range(tri.simplices.shape[0]):
            self.add_links(tri.simplices[sidx], links)

        for i in range(size):
            for j in range(i + 1, size):
                if links[i, j] == 1:
                    t1, t2 = self.towns[i], self.towns[j]
                    yield t1, t2, self.get_distance(i, j)

    def add_links(self, simplex, links):
        for idxs in self.orders:
            i0, i1, i2 = (simplex[j] for j in idxs)
            if i0 > i1:
                (i0, i1) = (i1, i0)
            links[i0, i1] = 1


class TravelInfoDelaunaySparse(TravelInfoDelaunay):
    def add_links(self, simplex, links):
        d = [self.get_distance(simplex[i1], simplex[i2]) for (i1, i2, i3) in self.orders]
        for idxs in self.orders:
            i0, i1, i2 = (simplex[j] for j in idxs)

            if i0 > i1:
                (i0, i1) = (i1, i0)

            if links[i0, i1] != -1:
                a, b, c = (d[idxs[j]] for j in range(3))
                gam = (a ** 2 + b ** 2 - c ** 2) / (2 * a * b)

                if gam < -.5:  # gamma < PI/2
                    # ommit
                    links[i0, i1] = -1
                else:
                    links[i0, i1] = 1
