import json
import numpy as np
import pandas as pd

import time
import logging


from models.engine import BaseEngine
from models.prob_infection import prob_of_contact
from utils.random_utils import RandomDuration
from utils.random_utils import gen_tuple
from utils.history_utils import TimeSeries, TransitionHistory, ShortListSeries
from models.states import STATES, state_codes

from utils.global_configs import monitor
import utils.global_configs as global_configs


class SimulationDrivenModel(BaseEngine):

    states = [
        STATES.S,
        STATES.S_s,
        STATES.E,
        STATES.I_n,
        STATES.I_a,
        STATES.I_s,
        STATES.J_s,
        STATES.J_n,
        STATES.R,
        STATES.D,
        STATES.EXT
    ]

    num_states = len(states)

    state_str_dict = state_codes

    transitions = [
        (STATES.S_s,  STATES.E),  # 0
        (STATES.S_s,  STATES.S),  # 1

        (STATES.S, STATES.E),  # 3
        (STATES.S, STATES.S_s),  # 4

        (STATES.E, STATES.I_n),  # 7
        (STATES.E, STATES.I_a),  # 8

        (STATES.I_n, STATES.J_n),
        (STATES.I_a, STATES.I_s),
        (STATES.I_s, STATES.J_s),

        (STATES.J_s, STATES.R),
        (STATES.J_s, STATES.D),

        (STATES.J_n, STATES.R)
    ]

    num_transitions = len(transitions)

    final_states = [
        STATES.R,
        STATES.D
    ]

    invisible_states = [
        STATES.D,
        STATES.EXT
    ]

    unstable_states = [
        STATES.E,
        STATES.I_n,
        STATES.I_a,
        STATES.I_s,
        STATES.J_n,
        STATES.J_s
    ]

    fixed_model_parameters = {
        "p": (0, "probability of interaction outside adjacent nodes"),
        "q": (0, " probability of detected individuals interaction outside adjacent nodes"),
        "mu": (0, "rate of infection-related death"),
        "false_symptoms_rate": (0, ""),
        "false_symptoms_recovery_rate": (1., ""),
        "save_nodes": (False, ""),
        "durations_file": ("../config/duration_probs.json", "file with probs for durations"),
        "prob_death_file": ("../data/prob_death.csv", "file with probs for durations"),
        "ext_epi": (0, "prob of beeing infectious for external nodes"),
        "beta_reduction": (0,  "reduction of beta for asymptomatic multiplier")
    }

    model_parameters = {
        "beta": (0,  "rate of transmission (exposure)"),
        "beta_in_family": (0, "hidden parameter"),
        "beta_A": (0, "hidden parameter"),
        "beta_A_in_family": (0, "hidden parameter"),
        "theta_E": (0, "rate of baseline testing for exposed individuals"),
        "theta_Ia": (0, "rate of baseline testing for Ia individuals"),
        "theta_Is": (0, "rate of baseline testing for Is individuals"),
        "theta_In": (0, "rate of baseline testing for In individuals"),
        "test_rate": (1.0, "test rate"),
        "psi_E": (0, "probability of positive test results for exposed individuals"),
        "psi_Ia": (0, "probability of positive test results for Ia individuals"),
        "psi_Is": (0, "probability of positive test results for Is individuals"),
        "psi_In": (0, "probability of positive test results for In individuals"),
        "asymptomatic_rate": (0, "asymptomatic rate"),
        "symptomatic_time": (-1, "time_from first_symptom  - do not setup"),
        "infectious_time": (-1, "time_from first_symptom  - do not setup"),
    }

    common_arguments = {
        "random_seed": (None, "random seed value"),
        "start_day": (1, "day to start")
    }

    def __init__(self, G, **kwargs):

        self.G = G  # backward compatibility
        self.graph = G

        self.init_kwargs = kwargs

        # 2. model initialization
        self.inicialization()

        # 3. time and history setup
        self.setup_series_and_time_keeping()

        # 4. init states and their counts
        self.states_and_counts_init(ext_nodes=self.num_ext_nodes,
                                    ext_code=STATES.EXT)

        # 5. set callback to None
        self.periodic_update_callback = None

        self.T = self.start_day - 1

    def update_graph(self, new_G):
        if new_G is not None:
            self.G = new_G  # just for backward compability
            self.graph = new_G
            self.num_nodes = self.graph.num_nodes
            try:
                self.num_ext_nodes = self.graph.num_nodes - self.graph.num_base_nodes
            except AttributeError:
                #  for saved old graph
                self.num_ext_nodes = 0
            self.nodes = np.arange(self.graph.number_of_nodes).reshape(-1, 1)

    def inicialization(self):

        self.init_kwargs["beta_in_family"] = self.init_kwargs["beta"]
        self.init_kwargs["beta_A"] = self.init_kwargs["beta"] * \
            self.init_kwargs["beta_reduction"]
        self.init_kwargs["beta_A_in_family"] = self.init_kwargs["beta_A"]

        super().inicialization()

        self.testable = np.zeros(
            shape=(self.graph.number_of_nodes, 1), dtype=bool)

        self.will_die = np.zeros(
            shape=(self.graph.number_of_nodes,), dtype=bool)

        # initialize random generators for durations
        with open(self.durations_file, "r") as f:
            self.duration_probs = json.load(f)

        self.rngd = {
            label: RandomDuration(probs)
            for label, probs in self.duration_probs.items()
        }

        # load death probs
        # first get codes for M an F
        self.MALE = self.graph.cat_table["sex"].index("M")
        self.FEMALE = self.graph.cat_table["sex"].index("F")

        # read and convert to dictionary for better lookup
        df = pd.read_csv(self.prob_death_file)
        df = df.set_index("age")
        df.rename(columns={"F": self.FEMALE,
                           "M": self.MALE},
                  inplace=True)
        self.death_probs = {
            self.FEMALE: df[self.FEMALE].to_numpy(),
            self.MALE: df[self.MALE].to_numpy()
        }

        # #test
        # nodes = self.graph.nodes[:10]
        # print(nodes)
        # age = self.graph.nodes_age[:10]
        # sex = self.graph.nodes_sex[:10]

        # print(age)
        # print(sex)

        # probs = np.zeros(len(nodes), dtype=float)

        # male = sex == self.MALE
        # probs[male] = self.death_probs[self.MALE][age[male]]

        # female = sex == self.FEMALE
        # probs[female] = self.death_probs[self.FEMALE][age[female]]

        # print(probs)

        # exit()

        # node indexes
        self.nodes = np.arange(self.graph.num_nodes).reshape(-1, 1)
        self.num_nodes = self.graph.num_nodes

    def setup_series_and_time_keeping(self):

        super().setup_series_and_time_keeping()

        self.expected_num_transitions = 10
        self.expected_num_days = 300

        tseries_len = self.num_transitions * self.num_nodes

        self.tseries = TimeSeries(tseries_len, dtype=float)
        self.history = TransitionHistory(tseries_len)

        # state history
        if global_configs.SAVE_NODES:
            history_len = self.expected_num_days
        else:
            history_len = 1
        self.states_history = TransitionHistory(
            history_len, width=self.num_nodes)

        if global_configs.SAVE_DURATIONS:
            self.states_durations = {
                s: []
                for s in self.states
            }

        self.durations = np.zeros(self.num_nodes, dtype=int)
        self.infect_time = np.zeros(self.num_nodes, dtype=int)

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
        for i in range(14):
            self.contact_history.append(None)
        self.successfull_source_of_infection = np.zeros(
            self.num_nodes, dtype="uint16")

        self.stat_successfull_layers = {
            layer: TimeSeries(self.expected_num_days, dtype=int)
            for layer in self.graph.layer_ids
        }

    def states_and_counts_init(self, ext_nodes=None, ext_code=None):
        super().states_and_counts_init(ext_nodes, ext_code)

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

        self.time_to_die = np.full(
            self.num_nodes, fill_value=-1, dtype="int32").reshape(-1, 1)

        self.testable = np.zeros(
            self.num_nodes, dtype=bool)

        # need update = need to recalculate time to go and state_to_go
        self.need_update = np.ones(self.num_nodes, dtype=bool)
        # need_check - state that needs regular checkup
        self.need_check = np.logical_or(
            self.memberships[STATES.S],
            self.memberships[STATES.S_s]
        )

        # self.time_to_go[(self.memberships[STATES.S] == True).ravel()] = 1
        # self.state_to_go[(self.memberships[STATES.S] ==
        #                   True).ravel()] = STATES.S_s

        # self.time_to_go[(self.memberships[STATES.E] == True).ravel()] = 1
        # self.state_to_go[(self.memberships[STATES.E] ==
        #                   True).ravel()] = STATES.I_a

        index = np.random.randint(37, size=10)
        self.time_to_go[index] = -1
        self.state_to_go[index] = -1

        # move all nodes to S and set move
        self.update_plan(np.ones(self.num_nodes, dtype=bool))

    def daily_update(self, nodes):
        """
        Everyday checkup
        """

        # S, S_s
        target_nodes = np.logical_or(
            self._get_target_nodes(nodes, STATES.S),
            self._get_target_nodes(nodes, STATES.S_s)
        )

        # if we have external nodes
        if self.num_ext_nodes > 0:
            self.flip_coin_for_external_edges()

        # try infection (may rewrite S/Ss moves)
        P_infection = prob_of_contact(self,
                                      [STATES.S_s, STATES.S],
                                      [STATES.S,
                                          STATES.S_s,
                                          STATES.E,
                                          STATES.I_n,
                                          STATES.I_a,
                                       STATES.I_s
                                       ],
                                      [STATES.I_n, STATES.I_a,
                                          STATES.I_s, STATES.EXT],
                                      [STATES.I_n, STATES.I_a,
                                          STATES.I_s, STATES.E],
                                      self.beta, self.beta_in_family
                                      ).flatten()

        #    r = np.random.rand(target_nodes.sum())
        exposed = P_infection[target_nodes]
        # print(exposed, exposed.shape)
        # exit()

        exposed_mask = np.zeros(self.num_nodes, dtype=bool)
        exposed_mask[target_nodes] = exposed

        self.time_to_go[exposed_mask] = 1
        self.state_to_go[exposed_mask] = STATES.E

    def change_states(self, nodes, target_state=None):
        """
        nodes that just entered a new state, update plan
        """
        # discard current state
        self.memberships[:, nodes == True] = 0

    #    print("DBG nodes", nodes == True)

        for node in nodes.nonzero()[0]:
            # print()
            # print(self.state_to_go.shape)
            # print(self.state_to_go)
            # exit()
            if target_state is None:
                new_state = self.state_to_go[node][0]
            else:
                new_state = target_state
            old_state = self.current_state[node, 0]
            # print(f"{new_state} {new_state.shape}")
            self.memberships[new_state, node] = 1
            self.state_counts[new_state][self.t] += 1
            self.state_counts[old_state][self.t] -= 1
            self.state_increments[new_state][self.t] += 1
            if global_configs.SAVE_NODES:
                self.states_history[self.t][node] = new_state

        if target_state is None:
            self.current_state[nodes] = self.state_to_go[nodes]
        else:
            self.current_state[nodes] = target_state
        self.update_plan(nodes)

    def update_plan(self, nodes):
        """ This is done for nodes that  just changed thier states.
        New plans are generated according the state."""

        # update plan
        # STATES.S:     "S",
        target_nodes = self._get_target_nodes(nodes, STATES.S)
        # print("---")
        # print(target_nodes.shape)
        # print(self.time_to_go.shape)

        self.time_to_go[target_nodes] = -1
        self.state_to_go[target_nodes] = STATES.S
        self.need_check[target_nodes] = True

        # STATES.S_s:   "S_s",
        # target_nodes = self._get_target_nodes(nodes, STATES.S_s)
        # assert target_nodes.sum() == 0, "S_s
        # self.time_to_go[target_nodes] = 7
        # self.state_to_go[target_nodes] = STATES.S
        # self.need_check[target_nodes] = True

        # STATES.E:     "E",
        target_nodes = self._get_target_nodes(nodes, STATES.E)
        # print(f"target nodes {target_nodes.shape}")
        # print(f"self.time_to_go {self.time_to_go.shape}")

        # asymptotic or symptomatic branch?
        r = np.random.rand(target_nodes.sum())
        asymptomatic = r < self.asymptomatic_rate[target_nodes, 0]

        asymptomatic_nodes = target_nodes.copy()
        asymptomatic_nodes[target_nodes] = asymptomatic

        symptomatic_nodes = target_nodes.copy()
        symptomatic_nodes[target_nodes] = np.logical_not(asymptomatic)

        self.time_to_go[target_nodes] = self.rngd["E"].get(
            n=(target_nodes.sum(), 1))
        self.state_to_go[asymptomatic_nodes] = STATES.I_n
        self.state_to_go[symptomatic_nodes] = STATES.I_a
        self.need_check[target_nodes] = False

        # STATES.I_n:   "I_n",
        # need to generate I duratin and J durations
        target_nodes = self._get_target_nodes(nodes, STATES.I_n)
        n = target_nodes.sum()
        if n > 0:
            expected_i_time, expected_j_time = gen_tuple(
                2,
                (n, 1),
                self.rngd["I"],
                self.rngd["RNA"]
            )

            self.infectious_time[target_nodes] = expected_i_time
            self.rna_time[target_nodes] = expected_j_time
            self.time_to_go[target_nodes] = expected_i_time
            self.state_to_go[target_nodes] = STATES.J_n
            self.need_check[target_nodes] = False

        # STATES.I_a:   "I_a",
        target_nodes = self._get_target_nodes(nodes, STATES.I_a)

        # current infectious time (part of total infectious time)
        expected_a_time, expected_i_time, expected_j_time = gen_tuple(
            3,
            (target_nodes.sum(), 1),
            self.rngd["A"],
            self.rngd["I"],
            self.rngd["RNA"]
        )

        self.infectious_time[target_nodes] = expected_i_time
        assert np.all(expected_a_time < expected_i_time)
        self.symptomatic_time[target_nodes] = expected_i_time - expected_a_time
        self.rna_time[target_nodes] = expected_j_time

        self.time_to_go[target_nodes] = expected_a_time
        self.state_to_go[target_nodes] = STATES.I_s
        self.need_check[target_nodes] = False

        # STATES.I_s:   "I_s",
        target_nodes = self._get_target_nodes(nodes, STATES.I_s)

        # decide for testing (testing policy must be ON to be tested)
        n = target_nodes.sum()
        if n > 0:
            r = np.random.rand(n)
            self.testable[target_nodes] = r < self.test_rate[target_nodes, 0]

            self.will_die[target_nodes] = self.die_or_not_to_die(target_nodes)
        target_nodes_to_die = np.logical_and(
            target_nodes,
            self.will_die
        )
        self.time_to_die[target_nodes_to_die,
                         0] = self.get_time_to_die(target_nodes_to_die)

        nodes_to_die_now = np.zeros(len(target_nodes), dtype=bool)
        nodes_to_die_now[target_nodes_to_die] = self.time_to_die[target_nodes_to_die,
                                                                 0] <= self.symptomatic_time[target_nodes_to_die, 0]
        nodes_to_live_now = target_nodes
        nodes_to_live_now[nodes_to_die_now] = False

        # -> D
        self.time_to_go[nodes_to_die_now] = self.time_to_die[nodes_to_die_now]
        self.state_to_go[nodes_to_die_now] = STATES.D
        self.need_check[nodes_to_die_now] = False

        # -> J_s
        assert np.all(self.symptomatic_time[target_nodes] > 0)
        self.time_to_go[nodes_to_live_now] = self.symptomatic_time[nodes_to_live_now]
        self.state_to_go[nodes_to_live_now] = STATES.J_s
        self.need_check[nodes_to_live_now] = False
        self.time_to_die[nodes_to_live_now] = self.time_to_go[nodes_to_live_now]

        # STATES.J_s:   "J_s",
        target_nodes = self._get_target_nodes(nodes, STATES.J_s)

        nodes_to_die = np.logical_and(
            target_nodes,
            self.will_die
        )
        target_nodes[nodes_to_die] = False
        left_rna_positivity = self.rna_time[target_nodes] - \
            self.infectious_time[target_nodes]

        # -> D
        self.time_to_go[nodes_to_die] = self.time_to_die[nodes_to_die]
        self.state_to_go[nodes_to_die] = STATES.D
        self.need_check[nodes_to_die] = False

        # -> R
        assert np.all(self.symptomatic_time[target_nodes] > 0)
        self.time_to_go[target_nodes] = left_rna_positivity
        self.state_to_go[target_nodes] = STATES.R
        self.need_check[target_nodes] = False

        # STATES.J_n:   "J_n",
        target_nodes = self._get_target_nodes(nodes, STATES.J_n)

        left_rna_positivity = self.rna_time[target_nodes] - \
            self.infectious_time[target_nodes]

        self.time_to_go[target_nodes] = left_rna_positivity
        self.state_to_go[target_nodes] = STATES.R
        self.need_check[target_nodes] = False

        # STATES.R:   "R",
        target_nodes = self._get_target_nodes(nodes, STATES.R)
        self.testable[target_nodes] = False
        self.time_to_go[target_nodes] = -1
        self.state_to_go[target_nodes] = -1
        self.need_check[target_nodes] = False

        # STATES.D:   "D",
        target_nodes = self._get_target_nodes(nodes, STATES.D)
        self.testable[target_nodes] = False
        self.time_to_go[target_nodes] = -1
        self.state_to_go[target_nodes] = -1
        self.need_check[target_nodes] = False

    def _get_target_nodes(self, nodes, state):
        ret = nodes.copy().ravel()
        is_target_state = self.memberships[state, ret, 0]
        ret[nodes.flatten()] = is_target_state
        # ret = np.logical_and(
        #     self.memberships[state].flatten(),
        #     nodes.flatten()
        # )
        return ret

    def run(self, T, print_interval=10, verbose=False):

        if global_configs.MONITOR_NODE is not None:
            monitor(0, f" being monitored, now in {self.state_str_dict[self.current_state[global_configs.MONITOR_NODE,0]]}")

        running = True
        self.tidx = 0
        self.T = self.start_day - 1
        if print_interval >= 0:
            self.print(verbose)

        for self.t in range(1, T+1):

            if self.num_ext_nodes > 0 and __debug__:
                # check that ext nodes are still ext nodes
                assert np.all(
                    self.memberships[STATES.EXT, self.nodes[:-self.num_ext_nodes], 0] == 0)
                assert np.all(
                    self.memberships[STATES.EXT, self.nodes[-self.num_ext_nodes:], 0] == 1)

                # check that ext nodes are not in quarantine
                if self.graph.is_quarantined is not None:
                    assert np.all(
                        self.graph.is_quarantined[self.nodes[-self.num_ext_nodes:]] == 0)

            self.T = self.start_day + self.t - 1

            if __debug__ and print_interval >= 0 and verbose:
                print(flush=True)

            if (self.t >= len(self.state_counts[0])):
                # room has run out in the timeseries storage arrays; double the size of these arrays
                self.increase_data_series_length()

            # reset
            #            self.successfull_source_of_infection.fill(0)

            if print_interval > 0 and verbose:
                start = time.time()
            running = self.run_iteration()

            # run periodical update
            if self.periodic_update_callback is not None:
                self.periodic_update_callback.run()

            if print_interval > 0 and (self.t % print_interval == 0):
                self.print(verbose)
                if verbose:
                    end = time.time()
                    print(f"Last day took: {end - start} seconds")

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
        if global_configs.SAVE_DURATIONS:
            for s in self.states:
                durations = self.durations[self.memberships[s].flatten() == 1]
                durations = durations[durations != 0]
                self.states_durations[s].extend(list(durations))

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

        for _, s in self.stat_successfull_layers.items():
            s[self.t] = 0

        self.durations += 1
        infectious_nodes = (
            self.memberships[STATES.I_a] +
            self.memberships[STATES.I_s] +
            self.memberships[STATES.I_n]
        ).ravel()
        self.infect_time[infectious_nodes == 1] += 1

        if global_configs.SAVE_NODES:
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

        if global_configs.SAVE_DURATIONS:
            for s, d in zip(orig_states, durs):
                assert(d > 0)
                self.states_durations[s].append(d)

    def move_to_R(self, nodes):
        target_nodes = np.zeros(self.num_nodes, dtype=bool)
        target_nodes[nodes] = True
        self.change_states(target_nodes, target_state=STATES.R)

    def move_target_nodes_to_R(self, target_nodes):
        """ same as move_to_R, but nodes given by bitmap target_nodes """
        self.change_states(target_nodes, target_state=STATES.R)

    def move_target_nodes_to_S(self, target_nodes):
        """ same as move_to_R, but nodes given by bitmap target_nodes """
        self.change_states(target_nodes, target_state=STATES.S)

    def move_to_E(self, num):

        # nodes_supply = [
        #    x
        #    for x in self.graph.nodes
        #    if (
        #            (self.graph.is_quarantined is None or not self.graph.is_quarantined[x])
        #            and
        #            self.memberships[STATES.R][x] != 1
        #    )
        # ]
        s_or_ss = np.logical_or(
            self.memberships[STATES.S],
            self.memberships[STATES.S_s]
        ).ravel()

        # s_or_ss = np.logical_and(
        #    s_or_ss,
        #    self.graph.nodes_age <= 20 # ucitele nenakazujeme
        # )

        nodes_supply = self.graph.nodes[s_or_ss]
        if len(nodes_supply) == 0:
            logging.warning("No nodes to infect.")
            return
        nodes = np.random.choice(nodes_supply, num, replace=False)

        target_nodes = np.zeros(self.num_nodes, dtype=bool)
        target_nodes[nodes] = True
        self.change_states(target_nodes, target_state=STATES.E)

    def print(self, verbose=False):
        print(f"T = {self.T} ({self.t})")
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
        self.successfull_source_of_infection = self.successfull_source_of_infection[
            np.logical_and(
                np.logical_and(
                    self.current_state != STATES.S,
                    self.current_state != STATES.S_s),
                self.current_state != STATES.E).flatten()
        ]

        df = pd.Series(self.successfull_source_of_infection)
        return df

    def save_node_states(self, filename):
        if global_configs.SAVE_NODES is False:
            logging.warning(
                "Nodes states were not saved, returning empty data frame.")
            return pd.DataFrame()
        index = range(0, self.t+1)
        columns = self.states_history.values
        df = pd.DataFrame(columns, index=index)
        df.to_csv(filename)
        # df = df.replace(self.state_str_dict)
        # df.to_csv(filename)
        # print(df)

    def die_or_not_to_die(self, target_nodes):
        """ decides whether node dies or not
        accept bitmap of target_nodes
        """
        n_target_nodes = target_nodes.sum()
        if n_target_nodes == 0:
            return np.array([], dtype=float)
        sex = self.graph.nodes_sex[target_nodes]
        age = self.graph.nodes_age[target_nodes].astype(int)
        probs = np.zeros(n_target_nodes, dtype=float)

        for sex_type in self.MALE, self.FEMALE:
            sel = sex == sex_type
            probs[sel] = self.mu * self.death_probs[sex_type][age[sel]]

        r = np.random.rand(n_target_nodes)
        return r < probs

    def get_time_to_die(self, target_nodes):
        """ Determine the time after which the node dies. """

        # 1. Vygeneruj U z R(0,1)
        # 2. Pokud U < 0.571, pak X=ceil(U / 0.571)
        # 3. Jinak X= round(4+ln(1-U)/-0.13))

        n_target_nodes = target_nodes.sum()
        self.dtime_coef = 0.571
        random_u = np.random.rand(n_target_nodes)
        lower = random_u < self.dtime_coef
        higher = np.logical_not(lower)

        time_X = np.zeros(n_target_nodes, dtype=int)
        time_X[lower] = np.ceil(10*random_u[lower] / self.dtime_coef)
        time_X[higher] = np.round(4-np.log(1-random_u[higher])/0.13)

        assert np.all(self.time_to_die[target_nodes, 0] == -1)
        return time_X

    def to_df(self):

        df = super().to_df()
        if self.start_day != 1:
            df["day"] = self.start_day + df["day"] - 1
            df.index = self.start_day + df.index - 1
        return df

    def get_dead(self):
        alld = self.state_counts[STATES.D][-1]
        dead_nodes = (self.memberships[STATES.D] == 1).flatten()
        ages = self.graph.nodes_age[dead_nodes]
        young = (ages < 65).sum()
        old1 = np.logical_and(ages >= 65, ages <= 79).sum()
        old2 = (ages >= 80).sum()
        return alld, young, old1, old2

    def flip_coin_for_external_edges(self):
        ext_nodes = self.nodes[-self.num_ext_nodes:].ravel()
        ext_edges = self.graph.get_nodes_edges(list(ext_nodes))
        self.graph.switch_on_edges(ext_edges)  # recover from the last time

        r = np.random.rand(len(ext_edges))
        ext_edges_off = np.array(ext_edges)[r >= self.ext_epi]
        self.graph.switch_off_edges(list(ext_edges_off))
