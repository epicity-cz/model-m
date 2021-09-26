import pandas as pd
import numpy as np
import scipy as scipy
import scipy.integrate
import networkx as nx
import time

from utils.history_utils import TimeSeries, TransitionHistory, ShortListSeries
from utils.sparse_utils import prop_of_row
from models.engine import BaseEngine


class SeirsPlusLikeEngine(BaseEngine):

    def inicialization(self):
        """ model inicialization """

        for argdict in (self.fixed_model_parameters,
                        self.common_arguments,
                        self.model_parameters):
            for name, definition in argdict.items():
                value = self.init_kwargs.get(name, definition[0])
                setattr(self, name, value)

        if self.random_seed:
            np.random.seed(self.random_seed)

        # setup adjacency matrix
        self.update_graph(self.G)

        # create arrays for model params
        for param_name in self.model_parameters:
            param = self.__getattribute__(param_name)
            if isinstance(param, (list, np.ndarray)):
                setattr(self, param_name,
                        np.array(param).reshape((self.num_nodes, 1)))
            else:
                setattr(self, param_name,
                        np.full(fill_value=param, shape=(self.num_nodes, 1)))
            # print(param_name, getattr(self, param_name))
#        exit()

    def setup_series_and_time_keeping(self):

        self.expected_num_transitions = 10  # TO: change to our situation
        self.expected_num_days = 301
        tseries_len = (self.num_transitions + 1) * self.num_nodes

        self.tseries = TimeSeries(tseries_len, dtype=float)
        self.meaneprobs = TimeSeries(self.expected_num_days, dtype=float)
        self.medianeprobs = TimeSeries(self.expected_num_days, dtype=float)

        self.history = TransitionHistory(tseries_len)

        self.states_history = TransitionHistory(
            1, width=self.num_nodes)

        self.states_durations = {
            s: []
            for s in self.states
        }

        # state_counts ... numbers of inidividuals in given states
        self.state_counts = {
            state: TimeSeries(self.expected_num_days, dtype=int)
            for state in self.states
        }

        self.num_tests = TimeSeries(self.expected_num_days, dtype=int)
        self.w_times = TimeSeries(self.expected_num_days, dtype=int)
        self.all_positive_tests = TimeSeries(self.expected_num_days, dtype=int)
        self.num_qtests = TimeSeries(self.expected_num_days, dtype=int)

        self.state_increments = {
            state: TimeSeries(self.expected_num_days, dtype=int)
            for state in self.states
        }

        # self.propensities_repo = {
        #     transition: TimeSeries(tseries_len, dtype=float)
        #     for transition in self.transitions
        # }

        # N ... actual number of individuals in population
        self.N = TimeSeries(self.expected_num_days, dtype=float)

        self.contact_history = ShortListSeries(14)

        # float time
        self.t = 0
        self.tmax = 0  # will be set when run() is called
        self.tidx = 0  # time index to time series
        self.tseries[0] = 0

    def states_and_counts_init(self):
        """ Initialize Counts of inidividuals with each state """

        self.init_state_counts = {
            s: self.init_kwargs.get(f"init_{self.state_str_dict[s]}", 0)
            for s in self.states
        }

        for state, init_value in self.init_state_counts.items():
            self.state_counts[state][0] = init_value

        for state in self.init_state_counts.keys():
            self.state_increments[state][0] = 0

        nodes_left = self.num_nodes - sum(
            [self.state_counts[s][0] for s in self.states]
        )

        self.state_counts[self.states[0]][0] += nodes_left

        # invisible nodes does not count to population (death ones)
        self.N[0] = self.num_nodes - sum(
            [self.state_counts[s][0] for s in self.invisible_states]
        )

        # self.states_history[0] ... initial array of states
        start = 0
        for state, count in self.state_counts.items():
            self.states_history[0][start:start+count[0]].fill(state)
            start += count[0]
        # distribute the states randomly
        np.random.shuffle(self.states_history[0])

#        ecko = np.arange(self.num_nodes)[self.states_history[0] == 2]
#        print("the winner is", ecko)
#        exit()

        # 0/1 num_states x num_nodes
        self.memberships = np.vstack(
            [self.states_history[0] == s
             for s in self.states]
        )
        self.memberships = np.expand_dims(self.memberships, axis=2).astype(int)
        # print(self.memberships.shape)
        # print(np.all(self.memberships.sum(axis=0) == 1))
        # print(self.memberships.sum(axis=1))
        # exit()

        self.durations = np.zeros(self.num_nodes, dtype="uint16")
        self.infect_start = np.zeros(self.num_nodes, dtype="uint16")
        self.infect_time = np.zeros(self.num_nodes, dtype="uint16")

        self.test_waiting = np.zeros(self.num_nodes, dtype=int)

    def node_degrees(self, Amat):
        """ return number of degrees of  nodes,
        i.e. sums of adj matrix cols """
        return Amat.sum(axis=0).reshape(self.num_nodes, 1)

    def update_scenario_flags(self):
        testing_infected = np.any(self.theta_Ia) or np.any(
            self.theta_Is) or np.any(self.phi_Ia) or np.any(self.phi_Is)
        positive_test_for_I = np.any(self.psi_Ia) or np.any(self.psi_Is)

        testing_exposed = np.any(self.theta_E) or np.any(self.phi_E)
        positive_test_for_E = np.any(self.psi_E)

        self.testing_scenario = (
            (positive_test_for_I and testing_infected) or
            (positive_test_for_E and testing_exposed)
        )

        tracing_E = np.any(self.phi_E)
        tracing_I = np.any(self.phi_Ia) or np.any(self.phi_Is)
        self.tracing_scenario = (
            (positive_test_for_E and tracing_E) or
            (positive_test_for_I and tracing_I)
        )

    def num_contacts(self, state):
        """ return numbers of contacts from given state
        if state is a list, it is sum over all states """

        if type(state) == int:
            # if TF_ENABLED:
            #     with tf.device('/GPU:' + "0"):
            #         x = tf.Variable(self.X == state, dtype="float32")
            #         return tf.sparse.sparse_dense_matmul(self.A, x)
            # else:
            return np.asarray(
                scipy.sparse.csr_matrix.dot(self.A, self.memberships[state]))

        elif type(state) == list:
            state_flags = self.memberships[state, :, :].reshape(
                len(state), self.num_nodes)
            # if TF_ENABLED:
            #     with tf.device('/GPU:' + "0"):
            #         x = tf.Variable(state_flags, dtype="float32")
            #         nums = tf.sparse.sparse_dense_matmul(self.A, x)
            # else:
            nums = scipy.sparse.csr_matrix.dot(state_flags, self.A)
            return np.sum(nums, axis=0).reshape(-1, 1)
        else:
            raise TypeException(
                "num_contacts(state) accepts str or list of strings")

    def prob_of_contact(self, source_states, source_candidate_states, dest_states, dest_candidate_states, beta):
        #        print(states)
        # for i in states:
        #    print(self.current_state_count(i))
        assert type(dest_states) == list and type(source_states) == list
        source_candidate_flags = self.memberships[source_candidate_states, :, :].reshape(
            len(source_candidate_states), self.num_nodes).sum(axis=0)
        source_candidate_indices = source_candidate_flags.nonzero()[0]

        dest_candidate_flags = self.memberships[dest_candidate_states, :, :].reshape(
            len(dest_candidate_states), self.num_nodes).sum(axis=0)
        dest_candidate_indices = dest_candidate_flags.nonzero()[0]

        vysek = self.A[source_candidate_flags ==
                       1, :][:, dest_candidate_flags == 1]
        vysek.eliminate_zeros()
        #        print(vysek.shape)
        if vysek.shape[0] == 0 or vysek.shape[1] == 0:
            return np.zeros((self.num_nodes, 1))
        # for each active edge flip coin
        r = np.random.rand(len(vysek.data))
        # set to 0/1 according the coin
        vysek.data = (vysek.data > r).astype(int)
        contact_indices = vysek.nonzero()
        # print(source_candidate_indices, contact_indices[0])
        # print(source_candidate_indices[contact_indices[0]])
        # covert vysek indicies to node numbers
        active_dest_indices = dest_candidate_indices[contact_indices[1]]
        active_source_indices = source_candidate_indices[contact_indices[0]]
        contact_indices = list(zip(active_dest_indices, active_source_indices))
        # the first element of couple is the infected one !
        self.contact_history.append(contact_indices)
        #        print("-->", self.contact_history.values)

        # print(contact_indices)
        dest_flags = self.memberships[dest_states, :, :].reshape(
            len(dest_states), self.num_nodes).sum(axis=0)
        source_flags = self.memberships[source_states, :, :].reshape(
            len(source_states), self.num_nodes).sum(axis=0)

        A_actual = scipy.sparse.csr_matrix(
            (np.ones(len(active_source_indices)),
             (active_source_indices, active_dest_indices)),
            shape=(self.num_nodes, self.num_nodes)
        )
        A_actual = A_actual[source_flags == 1, :][:, dest_flags == 1]
        A_actual.eliminate_zeros()
        if A_actual.shape[0] == 0 or A_actual.shape[1] == 0:
            return np.zeros((self.num_nodes, 1))
        beta = np.tile(beta[dest_flags == 1].ravel(),
                       (A_actual.shape[0], 1))
        not_prob_contact = scipy.sparse.csr_matrix(A_actual.multiply(beta))
        del A_actual
        not_prob_contact.data = 1.0 - not_prob_contact.data
        result = np.zeros(self.num_nodes)
        result[source_flags == 1] = 1 - prop_of_row(not_prob_contact)
        return result.reshape(self.num_nodes, 1)

        # if False:
        #     source_flags = self.memberships[source_states, :, :].reshape(
        #         len(source_states), self.num_nodes).sum(axis=0)
        #     #        print("state flags", state_flags.shape)
        #     #        print(self.node_ids[state_flags == 1])
        #     dest_flags = self.memberships[dest_states, :, :].reshape(
        #         len(dest_states), self.num_nodes).sum(axis=0)

        #     vysek = self.A[source_flags == 1, :][:, dest_flags == 1]
        #     vysek.eliminate_zeros()
        #     #        print(vysek.shape)
        #     if vysek.shape[0] == 0:
        #         return np.zeros((self.num_nodes, 1))
        #     not_prob_contact = scipy.sparse.csr_matrix(vysek)
        #     assert np.all(not_prob_contact.data >= 0) and np.all(
        #         not_prob_contact.data <= 1)
        #     #        print(not_prob_contact)
        #     beta = np.tile(beta[dest_flags == 1].ravel(),
        #                    (not_prob_contact.shape[0], 1))
        #     # print(not_prob_contact.shape, beta.shape)
        #     not_prob_contact = scipy.sparse.csr_matrix(
        #         not_prob_contact.multiply(beta))
        #     not_prob_contact.data = 1.0 - not_prob_contact.data
        #     # print("**** == 1", np.all(not_prob_contact.data == 1))
        #     # print("**** prop columns", prop_of_column(not_prob_contact), np.all(prop_of_column(not_prob_contact)==1))
        #     # print("*** contact ",  1 - prop_of_column(not_prob_contact))
        #     result = np.zeros(self.num_nodes)
        #     # print((1 - prop_of_row(not_prob_contact)).shape, result.shape,
        #     #      result[source_flags == 1].shape)
        #     result[source_flags == 1] = 1 - prop_of_row(not_prob_contact)
        #     return result.reshape(self.num_nodes, 1)

        # not_prob_contact = prob_contact
        # not_prob_contact.data = 1.0 - not_prob_contact.data
        #  product over columns

    def current_state_count(self, state):
        return self.state_counts[state][self.tidx]

    def current_N(self):
        return self.N[self.tidx]

    def increase_data_series_length(self):
        self.expected_num_transitions = 10  # TO: change to our situation
        tseries_len = (self.expected_num_transitions + 1) * self.num_nodes

        self.tseries.bloat(tseries_len)
        self.history.bloat(tseries_len)
        for state in self.states:
            self.state_counts[state].bloat(self.expected_num_days)
            self.state_increments[state].bloat(self.expected_num_days)
        # for tran in self.transitions:
        #     self.propensities_repo[tran].bloat(tseries_len)
        self.N.bloat(self.expected_num_days)

    def finalize_data_series(self):

        self.tseries.finalize(self.tidx)
        self.history.finalize(self.tidx)
        for state in self.states:
            self.state_counts[state].finalize(self.tidx)
            self.state_incrementss[state].finalize(self.tidx)
        self.N.finalize(self.tidx)

    def save(self, file_or_filename):
        index = self.tseries
        col_increments = {
            "inc_" + self.state_str_dict[x]: col_inc
            for x, col_inc in self.state_increments
        }
        col_states = {
            self.state_str_dic[x]: count
            for x, count in self.state_counts
        }
        columns = {**col_states, **col_increments}
        columns["day"] = np.floor(index).astype(int)
        df = pd.DataFrame(columns, index=index)
        df.index.rename('T', inplace=True)
        df.to_csv(file_or_filename)
        print(df)

    def run_iteration(self):

        if (self.tidx >= self.tseries.len()-1):
            # Room has run out in the timeseries storage arrays; double the size of these arrays
            self.increase_data_series_length()

        # 1. Generate 2 random numbers uniformly distributed in (0,1)
        r1 = np.random.rand()
        r2 = np.random.rand()

        # 2. Calculate propensities
        propensities = np.hstack(self.calc_propensities())
        transition_types = self.transitions
        alpha = propensities.sum()

        # Terminate when probability of all events is 0:
        if alpha <= 0.0:
            self.finalize_data_series()
            return False

        # 3. Calculate alpha
        propensities = propensities.ravel(order="F")
        cumsum = propensities.cumsum()

        # 4. Compute the time until the next event takes place
        tau = (1/alpha)*np.log(float(1/r1))
        self.t += tau

        # 5. Compute which event takes place
        transition_idx = np.searchsorted(cumsum, r2*alpha)
        transition_node = transition_idx % self.num_nodes
        transition_type = transition_types[int(transition_idx/self.num_nodes)]

        # 6. Update node states and data series
        assert (self.memberships[transition_type[0], transition_node] == 1), (f"Assertion error: Node {transition_node} has unexpected current state, given the intended transition of {transition_type}.")

        self.memberships[transition_type[0], transition_node] = 0
        self.memberships[transition_type[1], transition_node] = 1

        self.tidx += 1
        self.tseries[self.tidx] = self.t
        self.history[self.tidx] = (transition_node, *transition_type)

        for state in self.states:
            self.state_counts[state][self.tidx] = self.state_counts[state][self.tidx-1]
        self.state_counts[transition_type[0]][self.tidx] -= 1
        self.state_counts[transition_type[1]][self.tidx] += 1

        self.N[self.tidx] = self.N[self.tidx-1]
        # if node died
        if transition_type[1] in (self.invisible_states):
            self.N[self.tidx] = self.N[self.tidx-1] - 1

        # Terminate if tmax reached or num infectious and num exposed is 0:
        numI = sum([self.current_state_count(s)
                    for s in self.unstable_states
                    ])

        if self.t >= self.tmax or numI < 1:
            self.finalize_data_series()
            return False

        return True

    def run(self, T, print_interval=10, verbose=False):

        if not T > 0:
            return False

        self.tmax += T

        running = True
        day = -1
        if print_interval > 0 and verbose:
            start = time.time()

        while running:

            running = self.run_iteration()

            # true after the first event after midnight
            day_changed = day != int(self.t)
            day = int(self.t)
            if day_changed and print_interval > 0 and verbose:
                end = time.time()
                print("Last day took: ", end - start, "seconds")
                start = time.time()

            # run periodical update
            if self.periodic_update_callback and day != 0 and day_changed:
                print(self.periodic_update_callback)
                changes = self.periodic_update_callback(
                    self.history, self.tseries[: self.tidx+1], self.t)

                if "graph" in changes:
                    print("CHANGING GRAPH")
                    self.update_graph(changes["graph"])

            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # print only if print_interval is set
            # prints always at the beginning of a new day
            if print_interval > 0 or not running:
                if day_changed:
                    day = int(self.t)

                if not running or (day_changed and (day % print_interval == 0)):
                    print("t = %.2f" % self.t)
                    if verbose or not running:
                        for state in self.states:
                            print(f"\t {self.state_str_dict[state]} = {self.current_state_count(state)}")
                    print(flush=True)

        return True

        pass
