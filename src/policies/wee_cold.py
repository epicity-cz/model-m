import numpy as np
import pandas as pd 
from extended_network_model import STATES as states
from policy import Policy
from depo import Depo 
from history_utils import TimeSeries
from quarantine_coefs import QUARANTINE_COEFS

import logging

class WeeColdPolicy(Policy):

    """ 
    Self isolation policy.
    """
    
    def __init__(self, graph, model):
        super().__init__(graph, model)

        # depo of quarantined nodes 
        self.depo = Depo(graph.number_of_nodes)        
        self.stopped = False
        self.threshold = 0.75
        self.duration = 7
        self.coefs = QUARANTINE_COEFS

    def to_df(self):
        return None 

    def stop(self):
        """ just finish necessary, but do not qurantine new nodes """ 
        self.stopped = True


    def quarantine_nodes(self, detected_nodes):
        """ insert nodes into the depo and make modifications in a graph """ 
        if detected_nodes:
            assert self.coefs is not None 
            self.graph.modify_layers_for_nodes(detected_nodes,
                                               self.coefs)
            self.depo.lock_up(detected_nodes, self.duration)

    def tick(self):
        """ update time and return released """ 
        released = self.depo.tick_and_get_released()
        return released

    def release_nodes(self, released):
        """ update graph for nodes that are released """
        if len(released) > 0:
            logging.info(f"DBG {type(self).__name__} Released nodes: {released}")
            self.graph.recover_edges_for_nodes(released)

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
        

        if self.stopped == True:
            return  


        logging.info(
            f"Hello world! This is the wee cold function speaking.  {'(STOPPED)' if self.stopped else ''}")        
        logging.info(f"QE {type(self).__name__}: Nodes in isolation/quarantine {self.depo.num_of_prisoners}")

        last_day = self.get_last_day()
        
        # those who became symptomatic today
        detected_nodes = [
            node
            for node, s, e in last_day
            if e == states.I_s and not self.depo.is_locked(node) 
        ]
        logging.info(f"GDB Wee cold: got cold {len(detected_nodes)}")
        if len(detected_nodes) > 0: 
            r = np.random.rand(len(detected_nodes))
            responsible = r < self.threshold
            detected_nodes = [
                d
                for i, d in enumerate(detected_nodes)
                if responsible[i]
            ]
        logging.info(f"GDB Wee cold: stay home {len(detected_nodes)}")
                
        # quarantine opens doors
        # newly detected are locked up 
        released = self.tick()
        self.quarantine_nodes(detected_nodes)

        if len(released) > 0:
            still_symptomatic = released[self.is_I_s(released)]
            ready_to_leave = released[self.is_I_s(released) == False]
        
            if len(still_symptomatic) > 0:
                self.depo.lock_up(still_symptomatic, 1)
            if len(ready_to_leave) > 0:
                self.release_nodes(ready_to_leave)
        
      

    def is_I_s(self, node_ids):
        return self.model.memberships[states.I_s].ravel()[node_ids] == 1
