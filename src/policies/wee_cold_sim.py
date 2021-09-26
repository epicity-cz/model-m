import numpy as np
import pandas as pd
from models.agent_based_network_model import STATES
from policies.policy import Policy
from policies.depo import Depo
from utils.history_utils import TimeSeries
from policies.quarantine_coefs import QUARANTINE_COEFS
from utils.config_utils import ConfigFile


import logging


class WeeColdPolicy(Policy):

    """ 
    QuarantinePolicy with contact tracing.
    """

    def __init__(self, graph, model, config_file=None):
        super().__init__(graph, model)

        # depo of quarantined nodes
        self.depo = Depo(graph.number_of_nodes)
        self.stopped = False
        self.threshold = 0.5
        self.duration = 7

        if graph.QUARANTINE_COEFS is not None:
            self.coefs = graph.QUARANTINE_COEFS
        else:
            logging.warning("Using default quarantnine coefs.")
            self.coefs = QUARANTINE_COEFS

        self.symptomatic = np.zeros(model.num_nodes, dtype=bool).reshape(-1, 1)
        self.nodes = np.arange(model.num_nodes).reshape(-1, 1)

        # override defaults if config file provided
        if config_file:
            cf = ConfigFile()
            cf.load(config_file)
            my_config = cf.section_as_dict("SELFISOLATION")

            self.threshold = my_config.get("threshold", self.threshold)
            self.duration = my_config.get("duration", self.duration)

    def to_df(self):
        return None

    def stop(self):
        """ just finish necessary, but do not qurantine new nodes """
        self.stopped = True

    def quarantine_nodes(self, detected_nodes):
        """ insert nodes into the depo and make modifications in a graph """
        if len(detected_nodes) > 0:
            assert self.coefs is not None
            self.graph.modify_layers_for_nodes(detected_nodes,
                                               self.coefs)
            self.depo.lock_up(detected_nodes, self.duration,
                              check_duplicate=True)

    def tick(self):
        """ update time and return released """
        released = self.depo.tick_and_get_released()
        return released

    def release_nodes(self, released):
        """ update graph for nodes that are released """
        if len(released) > 0:
            logging.info(f"DBG {type(self).__name__} Released nodes: {len(released)}")
            self.graph.recover_edges_for_nodes(released)

    def run(self):

        if self.stopped == True:
            return

        logging.info(
            f"Hello world! This is the wee cold function speaking.  {'(STOPPED)' if self.stopped else ''}")
        logging.info(f"QE {type(self).__name__}: Nodes in isolation/quarantine {self.depo.num_of_prisoners}")

        # those who became symptomatic today
        have_symptoms = np.logical_or(
            self.model.memberships[STATES.S_s],
            self.model.memberships[STATES.I_s]
        ).ravel()
        new_symptomatic = np.logical_and(
            have_symptoms,
            self.depo.depo == 0
        )

        # update symptomatic
#        print("symptomatic", self.symptomatic.shape)
#        print("new_symptomatic", new_symptomatic.shape)
        self.symptomatic[np.logical_not(have_symptoms)] = False
        self.symptomatic[new_symptomatic] = True

        detected_nodes = self.nodes[new_symptomatic].ravel()

#        print("detected nodes", detected_nodes)

        logging.info(f"GDB Wee cold: got cold {len(detected_nodes)}")
        if len(detected_nodes) > 0:
            r = np.random.rand(len(detected_nodes))
            responsible = r < self.threshold
            detected_nodes = detected_nodes[responsible.ravel()]
        logging.info(f"GDB Wee cold: stay home {len(detected_nodes)}")

        # quarantine opens doors
        # newly detected are locked up
        released = self.tick()
        self.quarantine_nodes(list(detected_nodes))

        if len(released) > 0:
            still_symptomatic = released[self.is_I_s(released)]
            ready_to_leave = released[self.is_I_s(released) == False]

            if len(still_symptomatic) > 0:
                self.depo.lock_up(still_symptomatic, 1)
            if len(ready_to_leave) > 0:
                self.release_nodes(ready_to_leave)

    def is_I_s(self, node_ids):
        return (self.model.memberships[STATES.I_s][node_ids] +
                self.model.memberships[STATES.S_s][node_ids]).ravel() == 1
