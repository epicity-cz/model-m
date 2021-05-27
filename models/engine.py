import pandas as pd
import numpy as np
import scipy as scipy
import scipy.integrate
import networkx as nx
from pprint import pprint
from history_utils import TimeSeries, TransitionHistory


class BaseEngine():

    def setup_model_params(self, model_params_dict):
        # create arrays for model params
        for param_name, param in model_params_dict.items():
            if isinstance(param, (list, np.ndarray)):
                setattr(self, param_name,
                        np.array(param).reshape((self.num_nodes, 1)))
            else:
                setattr(self, param_name,
                        np.full(fill_value=param, shape=(self.num_nodes, 1)))

    def set_seed(self, random_seed):
        print("set_seed", random_seed)
        np.random.seed(random_seed)
        self.random_seed = random_seed

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

        model_params_dict = {
            param_name: self.__getattribute__(param_name)
            for param_name in self.model_parameters
        }
        self.setup_model_params(model_params_dict)

    def setup_series_and_time_keeping(self):

        self.t = 0
        tseries_len = 0
        self.expected_num_days = 30

        self.tseries = None
        self.meaneprobs = None
        self.medianeprobs = None

        self.history = None

        self.states_history = None
        self.states_durations = None

        # state_counts ... numbers of inidividuals in given states
        self.state_counts = None

        self.state_counts = {
            state: None
            for state in self.states
        }

        self.state_increments = {
            state: None
            for state in self.states
        }

        # N ... actual number of individuals in population
        self.N = TimeSeries(self.expected_num_days, dtype=float)

        # float time
        self.t = 0
        self.tmax = 0  # will be set when run() is called
        self.tidx = 0  # time index to time series

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

        # add the rest of nodes to first state (S)
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

        # 0/1 num_states x num_nodes
        self.memberships = np.vstack(
            [self.states_history[0] == s
             for s in self.states]
        )
        self.memberships = np.expand_dims(self.memberships, axis=2).astype(int)
        # print(self.memberships.shape)
        # print(np.all(self.memberships.sum(axis=0) == 1))
        # print(self.memberships.sum(axis=1))

        # print(self.states_history[0])
        # exit()

        self.durations = np.zeros(self.num_nodes, dtype="uint16")
        self.infect_start = np.zeros(self.num_nodes, dtype="uint16")
        self.infect_time = np.zeros(self.num_nodes, dtype="uint16")

    def update_graph(self, new_G):
        """ create adjacency matrix for G """
        self.G = new_G

        if isinstance(new_G, scipy.sparse.csr_matrix):
            self.A = new_G
        elif isinstance(new_G, np.ndarray):
            self.A = scipy.sparse.csr_matrix(new_G)
        elif type(new_G) == nx.classes.graph.Graph:
            # adj_matrix gives scipy.sparse csr_matrix
            self.A = nx.adj_matrix(new_G)
        else:
            # print(type(new_G))
            raise TypeError(
                "Input an adjacency matrix or networkx object only.")

        self.num_nodes = self.A.shape[1]
        self.degree = np.asarray(self.node_degrees(self.A)).astype(float)

        # if TF_ENABLED:
        #     self.A = to_sparse_tensor(self.A)

    def node_degrees(self, Amat):
        """ return number of degrees of  nodes,
        i.e. sums of adj matrix cols """
        # TODO FIX ME
        return Amat.sum(axis=0).reshape(self.num_nodes, 1)

    def set_periodic_update(self, callback):
        """ set callback function
        callback function is called every midnigh """
        self.periodic_update_callback = callback
        #        print(f"DBD callback set {callback.graph}")

    # TODO: need this???

    def update_scenario_flags(self):
        pass

    def num_contacts(self, state):
        """ return numbers of contacts from given state
        if state is a list, it is sum over all states """

        print("Warning: deprecated, do not use this method in newer engines.")

        if type(state) == str:
            # if TF_ENABLED:
            #     with tf.device('/GPU:' + "0"):
            #         x = tf.Variable(self.X == state, dtype="float32")
            #         return tf.sparse.sparse_dense_matmul(self.A, x)
            # else:
            return np.asarray(
                scipy.sparse.csr_matrix.dot(self.A, self.X == state))

        elif type(state) == list:
            state_flags = np.hstack(
                [np.array(self.X == s, dtype=int) for s in state]
            )
            # if TF_ENABLED:
            #     with tf.device('/GPU:' + "0"):
            #         x = tf.Variable(state_flags, dtype="float32")
            #         nums = tf.sparse.sparse_dense_matmul(self.A, x)
            # else:

            nums = scipy.sparse.csr_matrix.dot(self.A, state_flags)
            return np.sum(nums, axis=1).reshape((self.num_nodes, 1))
        else:
            raise TypeException(
                "num_contacts(state) accepts str or list of strings")

    def current_state_count(self, state):
        if self.state_counts[state] is None:
            return None
        return self.state_counts[state][self.tidx]

    def current_N(self):
        return self.N[self.tidx]

    def increase_data_series_length(self):
        pass

    def finalize_data_series(self):
        pass

    def run_iteration(self):
        pass

    def run(self, T, print_interval=10, verbose=False):
        pass

    def to_df(self):
        index = range(0, self.t+1)
        col_increments = {
            "inc_" + self.state_str_dict[x]: col_inc
            for x, col_inc in self.state_increments.items()
        }
        col_states = {
            self.state_str_dict[x]: count
            for x, count in self.state_counts.items()
        }

        columns = {**col_states, **col_increments}
        columns["day"] = np.floor(index).astype(int)
        columns["mean_p_infection"] = self.meaneprobs
        columns["median_p_infection"] = self.medianeprobs
        df = pd.DataFrame(columns, index=index)
        df.index.rename('T', inplace=True)
        return df

    def save(self, file_or_filename):
        """ Save timeseries. They have different format than in BaseEngine,
        so I redefined save method here """
        df = self.to_df()
        df.to_csv(file_or_filename)
        print(df)

    def save_durations(self, file_or_filename):
        print("Warning: self durations not implemented YET")

    def increase_data_series_length(self):
        for state in self.states:
            self.state_counts[state].bloat(100)
            self.state_increments[state].bloat(100)

        self.N.bloat(100)
        self.meaneprobs.bloat(100)
        self.medianeprobs.bloat(100)
        self.states_history.bloat(100)
        
    def increase_history_len(self):
        self.tseries.bloat(10*self.num_nodes)
        self.history.bloat(10*self.num_nodes)
        
        
    def finalize_data_series(self):
        self.tseries.finalize(self.tidx)
        self.history.finalize(self.tidx)

        
        for state in self.states:
            self.state_counts[state].finalize(self.t)
            self.state_increments[state].finalize(self.t)
        self.N.finalize(self.t)
        self.meaneprobs.finalize(self.t)
        self.medianeprobs.finalize(self.t)
        self.states_history.finalize(self.t)
        
    def print(self, verbose=False):
        print("t = %.2f" % self.t)
        if verbose:
            for state in self.states:
                print(f"\t {self.state_str_dict[state]} = {self.current_state_count(state)}")
