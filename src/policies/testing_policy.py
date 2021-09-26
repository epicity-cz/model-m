import time
import numpy as np
import pandas as pd
import logging

from policies.policy import Policy
from utils.history_utils import TimeSeries
from models.agent_based_network_model import STATES

from policies.depo import Depo
from policies.quarantine_coefs import QUARANTINE_COEFS

from utils.global_configs import monitor
import utils.global_configs as cfgs
from utils.config_utils import ConfigFile


class TestingPolicy(Policy):

    """
    TestingPolicy takes care of testing.
    """

    def __init__(self, graph, model, config_file=None):
        super().__init__(graph, model)

        self.nodes = np.arange(model.num_nodes)

        self.model.node_detected = np.zeros(
            model.num_nodes, dtype=bool)
        # how many times the node was tested
        self.node_tested = np.zeros(
            model.num_nodes, dtype="uint32")
        self.node_active_case = np.zeros(
            model.num_nodes, dtype=bool)

        self.stat_all_tests = TimeSeries(300, dtype=int)
        self.stat_positive_tests = TimeSeries(300, dtype=int)
        self.stat_negative_tests = TimeSeries(300, dtype=int)
        self.stat_Id = TimeSeries(300, dtype=int)  # active cases
        self.stat_cum_Id = TimeSeries(300, dtype=int)

        # day of first symptoms
        self.first_symptoms = np.zeros(model.num_nodes, dtype=int)

        self.first_day = True
        self.stopped = False

        self.duration = 10
        self.auto_recover = False
        self.init_deposit()

        if config_file is not None:
            cf = ConfigFile()
            cf.load(config_file)
            my_config = cf.section_as_dict("ISOLATION")

            self.duration = my_config.get("duration", self.duration)
            self.auto_recover = my_config.get(
                "auto_recover",
                "Yes" if self.auto_recover else "No"
            ) == "Yes"

    def init_deposit(self):
        # depo of quarantined nodes
        self.depo = Depo(self.graph.number_of_nodes)
        if self.graph.QUARANTINE_COEFS is None:
            self.coefs = QUARANTINE_COEFS
        else:
            self.coefs = self.graph.QUARANTINE_COEFS

    def quarantine_nodes(self, detected_nodes, last_contacts=False, duration=None):
        """ insert nodes into the depo and make modifications in a graph """
        if len(detected_nodes) > 0:
            assert self.coefs is not None

            new_detected_nodes = self.depo.filter_locked(detected_nodes)
            self.graph.modify_layers_for_nodes(new_detected_nodes,
                                               self.coefs)

            if duration is None:
                duration = self.duration
            if last_contacts:
                sentences = duration - \
                    (self.model.t - self.last_contact[detected_nodes])
            else:
                sentences = duration

            self.depo.lock_up(detected_nodes, sentences, check_duplicate=True)

            if cfgs.MONITOR_NODE is not None and cfgs.MONITOR_NODE in detected_nodes:
                monitor(self.model.t,
                        f"is beeing locked for {sentences if isinstance(sentences, int) else sentences[np.where(detected_nodes == cfgs.MONITOR_NODE)[0][0]]} days")

    def tick(self):
        """ update time and return released """
        released = self.depo.tick_and_get_released()
        return released

    def release_nodes(self, released):
        """ update graph for nodes that are released """
        if len(released) > 0:
            logging.debug(f"DBG {type(self).__name__} Released nodes: {len(released)}")
            self.graph.recover_edges_for_nodes(released)

            if cfgs.MONITOR_NODE is not None and cfgs.MONITOR_NODE in released:
                monitor(self.model.t,
                        f"is released from quarantine/isolation.")

    def to_df(self):
        """ create dataframe with statistics """
        index = range(0+self.model.start_day-1, self.model.t +
                      self.model.start_day)  # -1 + 1
        policy_name = type(self).__name__
        columns = {
            f"all_tests":  self.stat_all_tests[:self.model.t+1],
            f"positive_tests":  self.stat_positive_tests[:self.model.t+1],
            f"negative_tests":  self.stat_negative_tests[:self.model.t+1],
            f"I_d": self.stat_Id[:self.model.t+1],
            f"cum_I_d": self.stat_cum_Id[:self.model.t+1]
        }
        columns["day"] = np.floor(index).astype(int)
        df = pd.DataFrame(columns, index=index)
        df.index.rename('T', inplace=True)
        return df

    def stop(self):
        # just finish necessary, but do not qurantine new nodes
        self.stopped = True

    def process_detected_nodes(self, target_nodes):
        """"
        Put detected nodes into isolation. 

        target nodes ... list of nodes detected today
        """
        released = self.tick()
        # dead are those who died during qurantine/isolation
        released, dead = self.filter_dead(released)

        self.quarantine_nodes(target_nodes)

        if not self.auto_recover:
            recovered, still_ill = self.perform_test(released)
            self.depo.lock_up(still_ill, 2)

            if cfgs.MONITOR_NODE is not None and cfgs.MONITOR_NODE in still_ill:
                monitor(self.model.t,
                        f"will wait in quarantine/isolation another 2 days.")
        else:
            # autorecover
            recovered = released
            self.node_active_case[recovered] = False

        if len(recovered) > 0:
            logging.info(f"releasing {len(recovered)} nodes from isolation")
            self.release_nodes(recovered)

        if len(dead) > 0:
            logging.info(f"releasing {len(dead)} dead nodes from isolation")
            self.release_nodes(dead)
            self.node_active_case[dead] = False

    def first_day_setup(self):
        # fill the days before start by zeros
        self.stat_all_tests[0:self.model.t] = 0
        self.stat_positive_tests[0:self.model.t] = 0
        self.stat_negative_tests[0:self.model.t] = 0
        self.stat_Id[0:self.model.t] = 0
        self.stat_cum_Id[0:self.model.t] = 0

        self.first_day = False

    def update_first_symptoms(self):
        # nodes with symptoms
        is_nodes = (self.model.memberships[STATES.I_s] == 1).ravel()
        # day of first symptoms of nodes with symptoms
        first_symptoms = self.first_symptoms[is_nodes]
        # nodes with new symptoms
        is_nodes[is_nodes] = first_symptoms == 0
        self.first_symptoms[is_nodes] = self.model.t

    def filter_dead(self, nodes):
        is_not_dead = self.model.current_state[nodes, 0] != STATES.D
        return nodes[is_not_dead], nodes[np.logical_not(is_not_dead)]

    def select_test_candidates(self):
        """
        Flips coin for testable nodes and returns those who should be tested today. 
        returns np.array of nodes numbers
        """

        target_nodes = self.nodes[self.model.testable]

        # filter out dead nodes
        target_nodes, _ = self.filter_dead(target_nodes)

        n = len(target_nodes)
        if n == 0:
            return np.array([])

        r = np.random.rand(n)
        to_be_tested = r < self.model.theta_Is[target_nodes, 0]
        nodes_to_be_tested = target_nodes[to_be_tested]

        logging.info(
            f"TESTING: {len(nodes_to_be_tested)} nodes to be tested")

        if cfgs.MONITOR_NODE is not None and cfgs.MONITOR_NODE in nodes_to_be_tested:
            monitor(self.model.t,
                    f"decided to go for test today.")

        return nodes_to_be_tested

    def perform_test(self, nodes):
        """ Take nodes and creates two groups of them - healthy and ill.
            Healthy are those in states S, Ss, R, E (not detectable).
            Ill are those in states Is, In, Ia, Js, Jn.
            Nodes in states EXT and D are not allowed to take a test. 

            returns healthy, ill 
            (both int np.arrays) 
        """

        n = len(nodes)
        if not n > 0:
            return np.array([]), np.array([])

        assert np.all(self.model.current_state[nodes] != STATES.EXT)
        assert np.all(self.model.current_state[nodes] != STATES.D)

        # add number of tests to model statistics
        self.stat_all_tests[self.model.t] += n

        self.node_tested[nodes] += 1
        self.model.testable[nodes] = False

        # choose healthy nodes
        negative_nodes = (
            self.model.memberships[STATES.S, nodes, 0] +
            self.model.memberships[STATES.S_s, nodes, 0] +
            self.model.memberships[STATES.R, nodes, 0] +
            # E is not recognized by test
            self.model.memberships[STATES.E, nodes, 0]
        )

        if np.any(self.model.memberships[STATES.EXT, nodes, 0] == 1):
            raise ValueError("External node goes for test.")

        healthy = nodes[negative_nodes == 1]
        ill = nodes[negative_nodes == 0]

        assert (len(healthy) + len(ill)) == len(nodes)

        # give them flags
        self.stat_positive_tests[self.model.t] += len(ill)
        self.stat_negative_tests[self.model.t] += len(healthy)

        if len(ill) > 0:
            self.model.node_detected[ill] = True
            self.node_active_case[ill] = True

        if len(healthy) > 0:
            self.node_active_case[healthy] = False

        logging.debug(
            f"Tested {len(nodes)}, positive {len(ill)}, negative {len(healthy)}")

        if cfgs.MONITOR_NODE is not None and cfgs.MONITOR_NODE in nodes:
            if cfgs.MONITOR_NODE in ill:
                monitor(self.model.t,
                        f"has positive test.")
            if cfgs.MONITOR_NODE in healthy:
                monitor(self.model.t,
                        f"has negative test.")

        return healthy, ill

    def run(self):

        s = time.time()
        super().run()

        self.stat_all_tests[self.model.t] = 0
        self.stat_positive_tests[self.model.t] = 0
        self.stat_negative_tests[self.model.t] = 0

        self.update_first_symptoms()

        nodes_to_be_tested = self.select_test_candidates()
        healthy, ill = self.perform_test(nodes_to_be_tested)

        self.process_detected_nodes(ill)

        self.stat_Id[self.model.t] = self.node_active_case.sum()
        self.stat_cum_Id[self.model.t] = self.model.node_detected.sum()

        logging.info(f"t = {self.model.t} ACTIVE CASES: {self.stat_Id[self.model.t]}")
        e = time.time()
        logging.info(f"TESTING TIME {e-s}")
