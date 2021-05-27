import numpy as np
import pandas as pd
from policy import Policy
from history_utils import TimeSeries
import logging

from agent_based_network_model import STATES

from depo import Depo
from quarantine_coefs import QUARANTINE_COEFS

from global_configs import monitor
import global_configs as cfgs

from scipy.ndimage.interpolation import shift

logging.basicConfig(level=logging.DEBUG)



class TestingPolicy(Policy):

    """
    TestingPolicy takes care of testing.
    """

    def __init__(self, graph, model):
        super().__init__(graph, model)

        # depo of quarantined nodes
        self.nodes = np.arange(model.num_nodes).reshape(-1, 1)
        self.testable =np.full(model.num_nodes, fill_value=-1, dtype=int).reshape(-1,1)
        self.node_detected = np.zeros(
            model.num_nodes, dtype=bool).reshape(-1, 1)
        self.node_tested = np.zeros(
            model.num_nodes, dtype="uint32").reshape(-1, 1)
        self.node_active_case = np.zeros(
            model.num_nodes, dtype=bool).reshape(-1, 1)

        self.node_waiting_for_test = np.full(
            model.num_nodes, fill_value=-1, dtype=int)

        self.stat_all_tests = TimeSeries(301, dtype=int)
        self.stat_positive_tests = TimeSeries(301, dtype=int)
        self.stat_negative_tests = TimeSeries(301, dtype=int)
        self.stat_Id = TimeSeries(301, dtype=int)
        self.stat_cum_Id = TimeSeries(301, dtype=int)


        self.first_symptoms = np.zeros(model.num_nodes, dtype=int)

        # last contact == 0 means no contact (day 0 is skipped)
        self.last_contact = np.zeros(self.model.num_nodes, dtype="uint8")

        self.first_day = True
        self.stopped = False

        self.test_ceiling = 1000

        self.nodes_mask = np.zeros(model.num_nodes, dtype=bool).reshape(-1, 1)

        self.init_deposit()

    def init_deposit(self):
        # depo of quarantined nodes
        self.depo = Depo(self.graph.number_of_nodes)
        self.threshold = 0.5
        self.duration = 7
        self.coefs = QUARANTINE_COEFS

    def quarantine_nodes(self, detected_nodes):
        """ insert nodes into the depo and make modifications in a graph """
        if len(detected_nodes) > 0:
            assert self.coefs is not None
            self.graph.modify_layers_for_nodes(detected_nodes,
                                               self.coefs)
        
            
            sentences  = self.duration - (self.model.t - self.last_contact[detected_nodes])
        
            self.depo.lock_up(detected_nodes, sentences)
 
            if cfgs.MONITOR_NODE is not None and cfgs.MONITOR_NODE in detected_nodes:
                monitor(self.model.t,
                        f"is beeing locked for {sentences[detected_nodes.index(cfgs.MONITOR_NODE)]} days")


    def tick(self):
        """ update time and return released """
        released = self.depo.tick_and_get_released()
        return released

    def release_nodes(self, released):
        """ update graph for nodes that are released """
        if len(released) > 0:
            logging.debug(f"DBG {type(self).__name__} Released nodes: {released}")
            self.graph.recover_edges_for_nodes(released)

    def test_check(self, released):
        """ Check the given nodes.
        Returns array of healthy (negative test result) and array of ill (positive test result).
        """

        if not released.shape[0] > 0:
            return np.array([]), np.array([])

        # add number of tests to model statistics
        self.stat_all_tests[self.model.t] += len(released)
        self.testable[released, 0] = False 
        
        # chose those that are not ill (np.logical_or doesn't work for three members)
        node_is_R = (
            self.model.memberships[STATES.S, released, 0] +
            self.model.memberships[STATES.S_s, released, 0] +
            self.model.memberships[STATES.R, released, 0]  + 
            self.model.memberships[STATES.E, released, 0]  # E is not recognized by test
        )
        node_is_R = node_is_R > 0

        healthy = released[node_is_R == 1]
        still_ill = released[node_is_R == 0]
        
        # give them flags
        self.stat_positive_tests[self.model.t] += len(still_ill)
        self.stat_negative_tests[self.model.t] += len(healthy)

        if still_ill.shape[0] > 0:
            self.node_detected[still_ill] = True
            self.node_active_case[still_ill] = True 

        return healthy,  still_ill

    def to_df(self):
        index = range(0, self.model.t+1)
        policy_name = type(self).__name__
        inc_id = self.stat_Id[0:self.model.t+1] - shift(self.stat_Id[0:self.model.t+1], 1, cval=0)
        columns = {
            f"{policy_name}_all_tests":  self.stat_all_tests[:self.model.t+1],
            f"I_d": self.stat_Id[:self.model.t+1],
            f"inc_I_d": inc_id
        }
        columns["day"] = np.floor(index).astype(int)
        df = pd.DataFrame(columns, index=index)
        df.index.rename('T', inplace=True)
        return df

    def stop(self):
        """ just finish necessary, but do not qurantine new nodes """
        self.stopped = True

    def do_testing(self, target_nodes, n):

        logging.info(
            f"TESTING: {n} symptomatic nodes")
        r = np.random.rand(n)
        to_be_tested = r < self.model.theta_Is[target_nodes]
        can_be_tested = self.testable[target_nodes] == 1
        
        to_be_tested = np.logical_and(
            to_be_tested,
            can_be_tested
        )

        self.nodes_mask.fill(0)
        self.nodes_mask[target_nodes] = to_be_tested

        logging.info(
            f"TESTING: {self.nodes_mask.sum()} nodes to be tested")

        # testing
        self.node_tested[self.nodes_mask] += 1

        negative_nodes, possitive_nodes = self.test_check(
            self.nodes[self.nodes_mask])

        """
        possitive_nodes = np.logical_and(
            self.model.memberships[STATES.I_s],
            self.nodes_mask
        )

        negative_nodes = np.logical_and(
            np.logical_not(self.model.memberships[STATES.I_s]),
            self.nodes_mask
        )
        """

        return self.nodes_mask, possitive_nodes, negative_nodes

    def stay_home_rutine(self, target_nodes):

        # target nodes were detected today
        # they stay home
        self.quarantine_nodes(target_nodes)

        released = self.tick()
        
        recovered, still_ill = self.test_check(released)
        if len(still_ill) > 0:
            ill_states = self.model.current_state[still_ill].flatten()
            print(" *is ", [self.model.state_str_dict[x]
                            for x in ill_states])

        self.depo.lock_up(still_ill, 2)

        if recovered.shape[0] > 0:
            print("releasing nodes", recovered)
            self.release_nodes(recovered)
            self.node_active_case[recovered] = False

    def first_day_setup(self):
        # fill the days before start by zeros
        self.stat_all_tests[0:self.model.t] = 0
        self.stat_positive_tests[0:self.model.t] = 0
        self.stat_negative_tests[0:self.model.t] = 0
        self.stat_Id[0:self.model.t] = 0
        self.stat_cum_Id[0:self.model.t] = 0

        self.first_day = False

    def run(self):

        if self.first_day:
            self.first_day_setup()

        logging.info(
            f"Hello world! This is the TESTING POLICY function speaking.  {'(STOPPED)' if self.stopped else ''}")

        #        get all symptomatic nodes
        target_nodes = np.logical_or(
            self.model.memberships[STATES.S_s] == 1,
            self.model.memberships[STATES.I_s] == 1
            )

        real_symptoms_start = np.logical_and(
            self.first_symptoms == 0,
            self.model.memberships[STATES.I_s] == 1
            )
        
        self.first_symptoms[real_symptoms_start] = self.model.t

       # target_nodes = self.model.memberships[STATES.I_s] == 1

        target_nodes = np.logical_and(
            target_nodes,
            np.logical_not(self.node_active_case)
        )

        new_symptomatic = np.logical_and(
            target_nodes,
            np.logical_not(self.testable == -1)
        )
        r = np.random.rand(new_symptomatic.sum())
        print("R", r.shape)
        print("test_rate", self.model.test_rate.flatten()) 
        self.testable[new_symptomatic] = r < self.model.test_rate[new_symptomatic]
    

        n = target_nodes.sum()
        if n == 0:
            logging.info(
                f"TESTING: no symptomatic nodes")
            possitive_nodes = []
        else:
            logging.debug(f"n = {n}")

            tested, possitive_nodes, negative_nodes = self.do_testing(
                target_nodes, n)

            logging.debug(
                f"TEST tested {tested.sum()}, positive {len(possitive_nodes)}, negative {len(negative_nodes)}")

        self.stay_home_rutine(possitive_nodes)

        self.stat_Id[self.model.t] = self.node_active_case.sum()
        self.stat_cum_Id[self.model.t] = self.node_detected.sum()
        print(f"t = {self.model.t} ACTIVE CASES: {self.stat_Id[self.model.t]}")
