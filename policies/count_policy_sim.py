import numpy as np
from policy import Policy
from functools import partial, reduce
from agent_based_network_model import STATES 


class R0(Policy): 

    def __init__(self,
                 graph,
                 model):
        super().__init__(graph, model)
        self.start = None
        self.end = None
        
        self.seed_ready = False
        self.nodes = np.arange(model.num_nodes).reshape(-1,1)

        

    def run(self):

        exposed = self.model.memberships[STATES.E] == 1

        infectious = (
            self.model.memberships[STATES.I_a] +
            self.model.memberships[STATES.I_n] +
            self.model.memberships[STATES.I_s]            
        ) == 1

        # if exposed num = 1 and infectious num = 0 
        # we are still waiting for the seed to become infectious 
        if not self.seed_ready:
            if exposed.sum() == 1 and infectious.sum() == 0: 
                return 
            else:
                self.seed_ready = True
        
        #assert infectious.sum() < 2 
        if infectious.sum() > 2:
            print(self.nodes[infectious, 0])
            exit()
        if self.start is None and infectious.sum() == 1: 
            #            assert self.start is None, "maybe seed larger than 1?"
            self.start = self.model.t 

        infected_nodes = list(self.nodes[exposed])
        
        better = np.logical_or(
            self.model.memberships[STATES.J_s],
            self.model.memberships[STATES.J_n]
        )
        better = list(self.nodes[better])

        assert len(better) < 2 
        if self.end is None and len(better) == 1: 
            #            assert self.end is None
            self.end = self.model.t 

            print(f"DBG I-LENGTH {self.end-self.start}")

        self.model.move_to_R(infected_nodes)
