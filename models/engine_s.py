import gc
import os
import time
import networkx as nx
import scipy.integrate
import scipy as scipy
import numpy as np
import numpy_indexed as npi
import pandas as pd
import json

import logging


# from agent_based_network_model import STATES as s
from engine import BaseEngine
from history_utils import TimeSeries, TransitionHistory, ShortListSeries
from agent_based_network_model import STATES
from random_utils import RandomDuration
from global_configs import monitor
import global_configs


class EngineS(BaseEngine):
    """ should work in the same way like SEIRSPlusLikeEngine
    but makes the state changes only once a day """

    def inicialization(self):
        super().inicialization()
        self.testable = np.zeros(
            shape=(self.graph.number_of_nodes, 1), dtype=bool)

        with open(self.durations_file, "r") as f:
            self.duration_probs = json.load(f)

        self.rngd = {
            label: RandomDuration(probs)
            for label, probs in self.duration_probs.items()
        }

        self.nodes = np.arange(self.graph.number_of_nodes).reshape(-1, 1)

        self.num_nodes = self.graph.num_nodes

        # print(self.duration_probs)
        # exit()

    def update_graph(self, new_G):
        if new_G is not None:
            self.G = new_G  # just for backward compability, TODO: remove G from everywhere and replace by graph
            self.graph = new_G
            self.num_nodes = self.graph.num_nodes

    #     @property
    # def num_nodes(self):
    #     return self.graph.num_nodes

    def setup_series_and_time_keeping(self):

        super().setup_series_and_time_keeping()

        self.expected_num_transitions = 10
        self.expected_num_days = 300

        tseries_len = self.num_transitions * self.num_nodes

        self.tseries = TimeSeries(tseries_len, dtype=float)
        self.meaneprobs = TimeSeries(self.expected_num_days, dtype=float)
        self.medianeprobs = TimeSeries(self.expected_num_days, dtype=float)

        self.history = TransitionHistory(tseries_len)

        # state history
        self.states_history = TransitionHistory(
            self.expected_num_days, width=self.num_nodes)
        
        self.states_durations = {
            s: []
            for s in self.states
        }

        self.durations = np.zeros(self.num_nodes, dtype=int)

        # state_counts ... numbers of inidividuals in given states
        self.state_counts = {
            state: TimeSeries(self.expected_num_days, dtype=int)
            for state in self.states
        }

        self.state_increments = {
            state: TimeSeries(self.expected_num_days, dtype=int)
            for state in self.states
        }

        # N ... actual number of individuals in population
        self.N = TimeSeries(self.expected_num_days, dtype=float)

        # history of contacts for last 14 days
        self.contact_history = ShortListSeries(14)
        self.successfull_source_of_infection = np.zeros(
            self.num_nodes, dtype="uint16")
        # for school experiment: 
        self.infections_archive = []

        self.stat_successfull_layers = {
            layer: TimeSeries(self.expected_num_days, dtype=int)
            for layer in self.graph.layer_ids
        }

    def states_and_counts_init(self):
        super().states_and_counts_init()

        # time to go until I move to the state state_to_go
        self.time_to_go = np.full(
            self.num_nodes, fill_value=-1, dtype="int32").reshape(-1, 1)
        self.state_to_go = np.full(
            self.num_nodes, fill_value=-1, dtype="int32").reshape(-1, 1)

        self.current_state = self.states_history[0].copy().reshape(-1, 1)

        self.infectious_time = np.full(
            self.num_nodes, fill_value=-1, dtype="int32").reshape(-1, 1)
        self.symptomatic_time = np.full(
            self.num_nodes, fill_value=-1, dtype="int32").reshape(-1, 1)
        self.rna_time = np.full(
            self.num_nodes, fill_value=-1, dtype="int32").reshape(-1, 1)

        # need update = need to recalculate time to go and state_to_go
        self.need_update = np.ones(self.num_nodes, dtype=bool)
        # need_check - state that needs regular checkup
        self.need_check = np.logical_or(
            self.memberships[STATES.S],
            self.memberships[STATES.S_s]
        )

        self.time_to_go[(self.memberships[STATES.S] == True).ravel()] = 1
        self.state_to_go[(self.memberships[STATES.S] ==
                          True).ravel()] = STATES.S_s

        self.time_to_go[(self.memberships[STATES.E] == True).ravel()] = 1
        self.state_to_go[(self.memberships[STATES.E] ==
                          True).ravel()] = STATES.I_a

        index = np.random.randint(37, size=10)
        self.time_to_go[index] = -1
        self.state_to_go[index] = -1

        # move all nodes to S and set move
        self.update_plan(np.ones(self.num_nodes, dtype=bool))

        #print("DBG init time to go", self.time_to_go)
        #print("DBG init state to go", self.state_to_go)
        # exit()

    def run(self, T, print_interval=10, verbose=False):

        if global_configs.MONITOR_NODE is not None:
            monitor(0, f" being monitored, now in {self.state_str_dict[self.current_state[global_configs.MONITOR_NODE,0]]}")

        running = True
        self.tidx = 0
        if print_interval >= 0:
            self.print(verbose)

        for self.t in range(1, T+1):

            if __debug__ and print_interval >= 0 and verbose:
                print(flush=True)

            if (self.t >= len(self.state_counts[0])):
                # room has run out in the timeseries storage arrays; double the size of these arrays
                self.increase_data_series_length()

            # reset
            #            self.successfull_source_of_infection.fill(0)

            start = time.time()
            running = self.run_iteration()

            # run periodical update
            if self.periodic_update_callback is not None:
                self.periodic_update_callback.run()

            end = time.time()
            if print_interval > 0:
                print(f"Last day took: {end - start} seconds")

            if print_interval > 0 and (self.t % print_interval == 0):
                self.print(verbose)

            # new infectious may be inserted algoritmicaly
            # Terminate if tmax reached or num infectious and num exposed is 0:
            # numI = sum([self.current_state_count(s)
            #            for s in self.unstable_states
            #            ])
            # if numI == 0:
            #    break

        if self.t < T:
            for t in range(self.t+1, T+1):
                if (t >= len(self.state_counts[0])):
                    self.increase_data_series_length()
                for state in self.states:
                    self.state_counts[state][t] = self.state_counts[state][t-1]
                    self.state_increments[state][t] = 0

        # finalize durations
        # for s in self.states:
        #     durations = self.durations[self.memberships[s].flatten() == 1]
        #     self.states_durations[s].extend(list(durations))

        if print_interval >= 0:
            self.print(verbose)
        self.finalize_data_series()
        return True

    def run_iteration(self):

        logging.debug("DBG run iteration")

        # memberships check
        # try:
        #     assert np.all(self.memberships.sum(axis=0) == 1)
        # except AssertionError:
        #     values = self.memberships.sum(axis=0)
        #     print("AssertionError", values.shape)
        #     print(values)
        #     exit()
        # try:
        #     assert np.all(self.memberships >= 0)
        #     assert np.all(self.memberships <= 1)
        # except AssertionError:
        #     print("AssertionError", (self.memberships < 0).nonzero())
        #     exit()

        # for i in range(self.num_nodes):
        #     its_state = self.current_state[i][0]
        #     #            print(its_state, type(its_state))
        #     assert self.memberships[its_state, i] == 1, f"Node {i}: {self.state_str_dict[its_state]}"

        #        print(self.memberships.sum(axis=0))

        # prepare
        # add timeseries members
        for state in self.states:
            self.state_counts[state][self.t] = self.state_counts[state][self.t-1]
            self.state_increments[state][self.t] = 0
        self.N[self.t] = self.N[self.t-1]

        self.durations += 1 

        self.states_history[self.t] = self.states_history[self.t-1]
        
        #print("DBG Time to go", self.time_to_go)
        #print("DBG State to go", self.state_to_go)

        # update times_to_go and states_to_go and
        # do daily_checkup
        self.daily_update(self.need_check)

        self.time_to_go -= 1
        #print("DBG Time to go", self.time_to_go)
        nodes_to_move = self.time_to_go == 0

        if global_configs.MONITOR_NODE and nodes_to_move[global_configs.MONITOR_NODE]:
            node = global_configs.MONITOR_NODE
            monitor(self.t,
                    f"changing state from {self.state_str_dict[self.current_state[node,0]]} to {self.state_str_dict[self.state_to_go[node,0]]}")
        orig_states = self.current_state[nodes_to_move] 
        durs = self.durations[nodes_to_move.flatten()]
        self.change_states(nodes_to_move)
        self.durations[nodes_to_move.flatten()] = 0

        for s, d in zip(orig_states, durs):
            self.states_durations[s].append(d)

    def _get_target_nodes(self, nodes, state):
        ret = np.logical_and(
            self.memberships[state].flatten(),
            nodes.flatten()
        )
        return ret

    def get_targe_nodes(self, nodes, state):
        return self._get_target_nodes(nodes, state)

    def select_active_edges(self, source_states, source_candidate_states, dest_states, dest_candidate_states):
        assert type(dest_states) == list and type(source_states) == list

        # 1. select active edges
        # candidates for active edges are edges between source_candidate_states and dest_candidate_states
        # source (the one who is going to be infected)
        # dest (the one who can offer infection)

        s = time.time()
        source_candidate_flags = self.memberships[source_candidate_states, :, :].reshape(
            len(source_candidate_states), self.num_nodes).sum(axis=0)
        # source_candidate_indices = source_candidate_flags.nonzero()[0]

        dest_candidate_flags = self.memberships[dest_candidate_states, :, :].reshape(
            len(dest_candidate_states), self.num_nodes).sum(axis=0)
        # dest_candidate_indices = dest_candidate_flags.nonzero()[0]
        e = time.time()
        logging.info(f"Create flags {e-s}")

        s = time.time()
        possibly_active_edges, possibly_active_edges_dirs = self.graph.get_edges(
            source_candidate_flags,
            dest_candidate_flags
        )
        num_possibly_active_edges = len(possibly_active_edges)
        e = time.time()
        logging.info(f"Select possibly active {e-s}")

        s = time.time()

        if num_possibly_active_edges == 0:
            return None, None

        # for each possibly active edge flip coin
        r = np.random.rand(num_possibly_active_edges)
        e = time.time()
        logging.info(f"Random numbers: {e-s}")
        s = time.time()
        # edges probs
        p = self.graph.get_edges_probs(possibly_active_edges)
        e = time.time()
        logging.info(f"Get intensities: {e-s}")

        s = time.time()
        active_indices = (r < p).nonzero()[0]
        num_active_edges = len(active_indices)
        if num_active_edges == 0:
            return None, None

        active_edges = possibly_active_edges[active_indices]
        active_edges_dirs = possibly_active_edges_dirs[active_indices]
        e = time.time()
        logging.info(f"Select active edges: {e-s}")
        return active_edges, active_edges_dirs

    def archive_active_edges(self, active_edges, active_edges_dirs):
        s = time.time()
        # get source and dest nodes for active edges
        # source and dest met today, dest is possibly infectious, source was possibly infected
        source_nodes, dest_nodes = self.graph.get_edges_nodes(
            active_edges, active_edges_dirs)
        # add to contact_history (infectious node goes first!!!)
        contact_indices = list(zip(dest_nodes, source_nodes, active_edges))
        self.contact_history.append(contact_indices)

        if self.t == 6:
            contact_numbers = {}
            for e in active_edges:
                layer_number = self.graph.get_layer_for_edge(e)
                contact_numbers[layer_number] = contact_numbers.get(
                    layer_number, 0) + 1
            contact_num_str = ', '.join([f'{layer_type}:{number}' for layer_type, number in contact_numbers.items()])
            logging.debug(f"DBG contact numbers {contact_num_str}")

        # print("Potkali se u kolina:", contact_indices)
        logging.info(f"Todays contact num:  {len(contact_indices)}")
        e = time.time()
        logging.info(f"Archive active edges: {e-s}")

    def get_relevant_edges(self, active_edges, active_edges_dirs, source_states, dest_states):
        # restrict the selection to only relevant states
        # (ie. candidates can be both E and I, relevant are only I)
        # candidates are those who will be possibly relevant in future
        s = time.time()
        dest_flags = self.memberships[dest_states, :, :].reshape(
            len(dest_states), self.num_nodes).sum(axis=0)
        source_flags = self.memberships[source_states, :, :].reshape(
            len(source_states), self.num_nodes).sum(axis=0)

        relevant_edges, relevant_edges_dirs = self.graph.get_edges(
            source_flags, dest_flags)
        if len(relevant_edges) == 0:
            return None, None

        # get intersection
        active_relevant_edges = np.intersect1d(active_edges, relevant_edges)

        if len(active_relevant_edges) == 0:
            return None, None
        # print(active_relevant_edges.shape)
        # print(relevant_edges.shape)
        # print((active_relevant_edges[:, np.newaxis] == relevant_edges).shape)
        try:
            # this causes exceptin, but only sometimes ???
            # where_indices = (
            #    active_relevant_edges[:, np.newaxis] == relevant_edges).nonzero()[1]
            # lets try npi instead
            where_indices = npi.indices(relevant_edges, active_relevant_edges)
        except AttributeError as e:
            print(e)
            print("Lucky we are ...")
            print(active_relevant_edges.shape)
            np.save("active_relevant_edges", active_relevant_edges)
            print(relevant_edges.shape)
            np.save("relevant_edges", relevant_edges)
            print(active_relevant_edges[:, np.newaxis] == relevant_edges)
            exit()
        #        print(where_indices, len(where_indices))
        # always one index! (sources and dest must be disjunct)
        active_relevant_edges_dirs = relevant_edges_dirs[where_indices]
        e = time.time()
        logging.info(f"Get relevant active edges: {e-s}")
        return active_relevant_edges, active_relevant_edges_dirs

    def prob_of_contact(self, source_states, source_candidate_states, dest_states, dest_candidate_states, beta, beta_in_family):

        active_edges, active_edges_dirs = self.select_active_edges(
            source_states, source_candidate_states, dest_states, dest_candidate_states)
        if active_edges is None:  # we have no active edges today
            return np.zeros((self.num_nodes, 1))

        self.archive_active_edges(active_edges, active_edges_dirs)

        active_relevant_edges, active_relevant_edges_dirs = self.get_relevant_edges(active_edges,
                                                                                    active_edges_dirs,
                                                                                    source_states,
                                                                                    dest_states)
        if active_relevant_edges is None:
            return np.zeros((self.num_nodes, 1))

        s = time.time()
        intensities = self.graph.get_edges_intensities(
            active_relevant_edges).reshape(-1, 1)
        relevant_sources, relevant_dests = self.graph.get_edges_nodes(
            active_relevant_edges, active_relevant_edges_dirs)

        is_family_edge = self.graph.is_family_edge(
            active_relevant_edges).reshape(-1, 1)
        """
        pro experiment s hospodama jenom: 
        """
        # TODO list of no-masks layers into config
        # get rid of this mess
        #        if self.t <= 111:
        if True:
            is_class_edge = self.graph.is_class_edge(
                active_relevant_edges).reshape(-1, 1)
            is_pub_edge = self.graph.is_pub_edge(
                active_relevant_edges).reshape(-1, 1)
            is_super_edge = self.graph.is_super_edge(
                active_relevant_edges).reshape(-1, 1)
            is_family_edge = np.logical_or.reduce((is_family_edge,
                                                   is_class_edge,
                                                   is_super_edge,
                                                   is_pub_edge))
        """
        else:
            is_class_edge = self.graph.is_class_edge(
                active_relevant_edges).reshape(-1, 1)
            is_family_edge = np.logical_or(is_family_edge, is_class_edge)
        """

        #        assert len(relevant_sources) == len(set(relevant_sources))
        # TOD beta - b_ij
        # new beta depands on the one who is going to be infected
        #        b_intensities = beta[relevant_sources]
        #        b_f_intensities = beta_in_family[relevant_sources]

        # reduce asymptomatic
        is_A = self.memberships[STATES.I_n][relevant_dests]

        b_original_intensities = (
            beta_in_family[relevant_sources] * (1 - is_A) +
            self.beta_A_in_family[relevant_sources] * is_A
        )
        b_reduced_intensities = (
            beta[relevant_sources] * (1 - is_A) +
            self.beta_A[relevant_sources] * is_A
        )

        b_intensities = (
            b_original_intensities * is_family_edge +
            b_reduced_intensities * (1 - is_family_edge)
        )

        assert b_intensities.shape == intensities.shape

        # relevant_sources_unique, unique_indices = np.unique(
        #     relevant_sources, return_inverse=True)

        # print(len(relevant_sources_unique))
        # print(b_intensities, b_intensities.shape)
        # print(active_relevant_edges, active_relevant_edges.shape)
        # exit()

        r = np.random.rand(b_intensities.ravel().shape[0]).reshape(-1, 1)
        # print(b_intensities.shape)
        # print(intensities.shape)
        # print((b_intensities*intensities).shape)
        is_exposed = r < (b_intensities * intensities)

        #        print(is_exposed, is_exposed.shape)
        if np.all(is_exposed == False):
            return np.zeros((self.num_nodes, 1))

        is_exposed = is_exposed.ravel()
        exposed_nodes = relevant_sources[is_exposed]
        ret = np.zeros((self.num_nodes, 1))
        ret[exposed_nodes] = 1

        sourced_nodes = relevant_dests[is_exposed]
        dest_nodes = relevant_sources[is_exposed]
        self.successfull_source_of_infection[sourced_nodes] += 1

        self.infections_archive.extend(
            list(zip(sourced_nodes, dest_nodes))
        )

        succesfull_edges = active_relevant_edges[is_exposed]
        successfull_layers = self.graph.get_layer_for_edge(succesfull_edges)
        for e in successfull_layers:
            self.stat_successfull_layers[e][self.t] += 1

        return ret
        # no_infection = (1 - b_intensities * intensities).ravel()

        # res = np.ones(len(relevant_sources_unique), dtype='float32')
        # for i in range(len(unique_indices)):
        #     res[unique_indices[i]] = res[unique_indices[i]]*no_infection[i]
        # prob_of_no_infection = res
        # # prob_of_no_infection2 = np.fromiter((np.prod(no_infection, where=(relevant_sources==v).T)
        # #                         for v in relevant_sources_unique), dtype='float32')

        # result = np.zeros(self.num_nodes)
        # result[relevant_sources_unique] = 1 - prob_of_no_infection
        # e = time.time()
        # print("Comp probability", e-s)
        #        return result.reshape(self.num_nodes, 1)

    def move_to_R(self, nodes):
        target_nodes = np.zeros(self.num_nodes, dtype=bool)
        target_nodes[nodes] = True
        self.change_states(target_nodes, target_state=STATES.R)

    def move_to_E(self, num):
        
        #nodes_supply = [
        #    x
        #    for x in self.graph.nodes
        #    if (
        #            (self.graph.is_quarantined is None or not self.graph.is_quarantined[x])
        #            and
        #            self.memberships[STATES.R][x] != 1
        #    )
        #]
        s_or_ss = np.logical_or(
            self.memberships[STATES.S],
            self.memberships[STATES.S_s]
        ).ravel()

        #s_or_ss = np.logical_and(
        #    s_or_ss,
        #    self.graph.nodes_age <= 20 # ucitele nenakazujeme 
        #)  
        
        nodes_supply = self.graph.nodes[s_or_ss]
        if  len(nodes_supply) == 0:
            logging.warning("No nodes to infect.")
            return 
        nodes = np.random.choice(nodes_supply, num, replace=False)
                        
        target_nodes = np.zeros(self.num_nodes, dtype=bool)
        target_nodes[nodes] = True
        self.change_states(target_nodes, target_state=STATES.E)
        


        
    def print(self, verbose=False):
        print("t = %.2f" % self.t)
        if verbose:
            for state in self.states:
                print(f"\t {self.state_str_dict[state]} = {self.state_counts[state][self.t]}")

    def df_source_infection(self):
        df = pd.DataFrame(index=range(0, self.t))
        for i in self.graph.layer_ids:
            df[self.graph.layer_name[i]] = self.stat_successfull_layers[i].asarray()[
                :self.t]
        return df

    def save_durations(self, f):
        for s in self.states:
            line = ",".join([str(x) for x in self.states_durations[s]])
            print(f"{self.state_str_dict[s]},{line}", file=f)

    def df_source_nodes(self):
        self.successfull_source_of_infection =  self.successfull_source_of_infection[
                np.logical_and(
                    np.logical_and(
                        self.current_state != STATES.S,
                        self.current_state != STATES.S_s),
                    self.current_state != STATES.E).flatten()
            ]
        
        df = pd.Series(self.successfull_source_of_infection)
        return df

    def df_infections_archive(self):
        return pd.DataFrame(self.infections_archive,
                            columns=["source of infection", "infected node"])
    

    def save_node_states(self, filename):
        index = range(0, self.t+1)
        columns = self.states_history.values
        df = pd.DataFrame(columns, index=index)
        df.to_csv(filename)
        # df = df.replace(self.state_str_dict)
        # df.to_csv(filename)
        # print(df)
