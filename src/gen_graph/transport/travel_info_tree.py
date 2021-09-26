import numpy as np

from transport.travel_info import TravelInfo
from utils.math_prob import yes_no


class TravelInfoTree(TravelInfo):
    def generate_connections(self):
        size = len(self.towns)
        population = np.zeros(size)
        locations = np.zeros([size, 2])

        for idx, town in enumerate(self.towns):
            population[idx] = town.inhabitants
            locations[idx, :] = town.location

        distances = np.sqrt(np.sum(np.square(locations[:, None, :] - locations[None, :, :]), axis=2))
        logpop = np.log10(population)
        distances_weighted = distances / (logpop[:, None] * logpop[None, :])
        distances_weighted[np.tril_indices(size)] = np.Inf

        clusters = np.arange(size)

        sorted = np.transpose(np.unravel_index(np.argsort(distances_weighted, axis=None), distances_weighted.shape))
        for (i, j) in sorted:
            print(f"trying:{self.towns[i].name} - {self.towns[j].name}  {distances_weighted[i, j]}")
            if (clusters[i] != clusters[j]) or yes_no(0.0):
                yield self.towns[i], self.towns[j], distances[i, j]
                clusters[clusters == clusters[j]] = clusters[i]
                if np.all(clusters == clusters[0]):
                    break
