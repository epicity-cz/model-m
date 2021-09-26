import numpy as np
import logging

class Depo:

    """
    Deposit object.
    To be used in policies to store a node for a given period of time.
    """
    def __init__(self, size):
        self.depo = np.zeros(size, dtype="uint8")

    @property
    def num_of_prisoners(self):
        return (self.depo > 0).sum()

    def lock_up(self, nodes, duration=14, check_duplicate=False):
        """
        Put given nodes (list or numpy array of nodes numbers) to the deposit
        for given duration.

        If check duplicate is True, ignore nodes that are already in depo (their time is not updated)
        """
        assert isinstance(nodes, np.ndarray) or isinstance(nodes, list), f"real type {type(nodes)}"
        if len(nodes) > 0:
            assert check_duplicate or np.all(self.depo[nodes] == 0)

            if check_duplicate:
                nodes = np.array(nodes)
                zero_nodes = nodes[(self.depo[nodes] == 0).nonzero()[0]]
                self.depo[zero_nodes] = duration
            else:
                self.depo[nodes] = duration

    def filter_locked(self, candidates):
        """
        Returns those from candidates that are not yet in depo.
        """
        if len(candidates) > 0:
            return candidates[self.depo[candidates]==0]
        else:
            return candidates
    

    def filter_locked_bitmap(self, candidates):
        """
        Returns those from candidates that are not yet in depo.
        Returns bitmap of the lenth of candidates,
        1 candidate not in depo,
        0 candidate already in depo.
       """
        return self.depo[candidates] == 0

    def tick_and_get_released(self):
        """
        Perform one time step and return released nodes.
        """
        released = np.nonzero(self.depo == 1)[0]
        self.depo[self.depo >= 1] -= 1
        return released

    def is_locked(self, node_id):
        """
        Returns True if the node given by node_id is lockec,
        False otherwise.
        """
        return self.depo[node_id] > 0
