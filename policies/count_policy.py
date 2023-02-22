import numpy as np
from policy import Policy
from functools import partial, reduce
from extended_network_model import STATES as states


class R0(Policy): 

    def __init__(self,
                 graph,
                 model):
        super().__init__(graph, model)
        self.start = None
        self.end = None

    def get_last_day(self):
        """ get the part of model history corresponding to the current (last) day """ 
        current_day = int(self.model.t)
        start = np.searchsorted(
            self.model.tseries[:self.model.tidx+1], current_day, side="left")
        if start == 0:
            start = 1
        end = np.searchsorted(
            self.model.tseries[:self.model.tidx+1], current_day+1, side="left")
        return self.model.history[start:end]
        
        

    def run(self):

        last_day = self.get_last_day()
        
        infected_nodes = [
            node
            for node, _, e in last_day
            if e == states.E
        ]

        infectious = [
            node
            for node, _, e in last_day
            if e in (states.I_n, states.I_a)
        ]
        assert len(infectious) < 2 
        if len(infectious) == 1: 
            assert self.start is None, "maybe seed larger than 1?"
            self.start = self.model.t 

        better = [
            node
            for node, _, e in last_day
            if e in (states.J_n, states.J_s)
        ]
        assert len(better) < 2 
        if len(better) == 1: 
            assert self.end is None
            self.end = self.model.t 

            print(f"DBG I-LENGTH {self.end-self.start}")

        self.model.move_to_R(infected_nodes)
