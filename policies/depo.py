import numpy as np
import logging

class Depo:

    def __init__(self, size):
        self.depo = np.zeros(size, dtype="uint8")

    @property
    def num_of_prisoners(self):
        return (self.depo > 0).sum()

    def lock_up(self, nodes, duration=14, check_duplicate=False):
        assert isinstance(nodes, np.ndarray) or isinstance(nodes, list), f"real type {type(nodes)}"
        if len(nodes) > 0:
            #logging.debug(f"Nodes in depo: {[n for n in nodes if self.depo[n]>0]}")
            #logging.debug(f"Nodes comming to depo: {nodes}") 
            if 31513 in nodes:
                logging.debug(f"Node 31513 depo value {self.depo[31513]}") 
            assert check_duplicate or np.all(self.depo[nodes] == 0)
            if check_duplicate:
                nodes = np.array(nodes)
                zero_nodes = nodes[(self.depo[nodes] == 0).nonzero()[0]]
                self.depo[zero_nodes] = duration
            else:
                self.depo[nodes] = duration

    def filter_locked(self, candidates):
        return candidates[self.depo[candidates]==0]

    def filter_locked_bitmap(self, candidates):
        return self.depo[candidates] == 0

    def tick_and_get_released(self):
        released = np.nonzero(self.depo == 1)[0]
        self.depo[self.depo >= 1] -= 1
        return released

    def is_locked(self, node_id):
        return self.depo[node_id] > 0
