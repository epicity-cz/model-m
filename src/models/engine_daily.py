import numpy as np
import scipy as scipy
import scipy.integrate
import networkx as nx
import time

from utils.history_utils import TimeSeries, TransitionHistory
from models.engine_seirspluslike import SeirsPlusLikeEngine


class DailyEngine(SeirsPlusLikeEngine):
    """ should work in the same way like SEIRSPlusLikeEngine
    but makes the state changes only once a day """

    def inicialization(self):

        self.todo_list = []
        self.todo_t = []

        super().inicialization()

    def run_iteration(self, alpha, cumsum, transition_types):

        # 1. Generate 2 random numbers uniformly distributed in (0,1)
        r1 = np.random.rand()
        r2 = np.random.rand()

        # 2. Calculate propensities
        #        propensities, transition_types = self.calc_propensities()

        # Terminate when probability of all events is 0:
        # if propensities.sum() <= 0.0:
        #     self.finalize_data_series()
        #     return False

        # 4. Compute the time until the next event takes place
        tau = (1/alpha)*np.log(float(1/r1))
        self.t += tau

        # 5. Compute which event takes place
        transition_idx = np.searchsorted(cumsum, r2*alpha)
        transition_node = transition_idx % self.num_nodes
        transition_type = transition_types[int(transition_idx/self.num_nodes)]

        if transition_node not in [x[0] for x in self.todo_list]:
            #        if (transition_node, transition_type) not in self.todo_list:
            self.todo_t.append(self.t)
            self.todo_list.append((transition_node, transition_type))

        return True

    def update_states(self):
        #        print("updating states")
        # for t, (transition_node, transition_type) in zip(self.todo_t, self.todo_list):
        #     print(t, transition_node, "-->", transition_type)
        # 6. Update node states and data series
        for t, (transition_node, transition_type) in zip(self.todo_t, self.todo_list):
            self.tidx += 1
            if (self.tidx >= self.tseries.len()-1):
                # Room has run out in the timeseries storage arrays; double the size of these arrays
                self.increase_data_series_length()

            assert (self.memberships[transition_type[0], transition_node] == 1), (f"Assertion error: Node {transition_node} has unexpected current state, given the intended transition of {transition_type}.")

            self.memberships[transition_type[0], transition_node] = 0
            self.memberships[transition_type[1], transition_node] = 1
            self.tseries[self.tidx] = t
            self.history[self.tidx] = (transition_node, *transition_type)

            for state in self.states:
                self.state_counts[state][self.tidx] = self.state_counts[state][self.tidx-1]
            self.state_counts[transition_type[0]][self.tidx] -= 1
            self.state_counts[transition_type[1]][self.tidx] += 1

            self.N[self.tidx] = self.N[self.tidx-1]
            # if node died
            if transition_type[1] in (self.invisible_states):
                self.N[self.tidx] = self.N[self.tidx-1] - 1

        del self.todo_list
        del self.todo_t
        self.todo_list = []
        self.todo_t = []

    def midnight(self, verbose):
        self.update_states()

        # run periodical update
        if self.periodic_update_callback:
            changes = self.periodic_update_callback(
                self.history, self.tseries[:self.tidx+1], self.t)

            if "graph" in changes:
                print("CHANGING GRAPH")
                self.update_graph(changes["graph"])

        return self.propensities_recalc()

    def print(self, verbose=False):
        print("t = %.2f" % self.t)
        if verbose:
            for state in self.states:
                print(f"\t {self.state_str_dict[state]} = {self.current_state_count(state)}")
                print(flush=True)

    def propensities_recalc(self):
        # 2. Calculate propensities
        propensities = np.hstack(self.calc_propensities())
        transition_types = self.transitions

        # 3. Calculate alpha
        # nebylo by rychlejsi order=C a prohodi // a % ?
        propensities_flat = propensities.ravel(order="F")
        cumsum = propensities_flat.cumsum()
        alpha = propensities_flat.sum()
        return alpha, cumsum, propensities.sum() > 0.0, transition_types

    def run(self, T, print_interval=10, verbose=False):

        if not T > 0:
            return False

        self.tmax += T

        running = True
        day = -1

        self.print(verbose=True)
        start = time.time()

        alpha, cumsum, running, transition_types = self.propensities_recalc()

        while running:

            running = self.run_iteration(alpha, cumsum, transition_types)

            # true after the first event after midnight
            day_changed = day != int(self.t)
            day = int(self.t)
            if day_changed and day != 0:
                alpha, cumsum, running, transition_types = self.midnight(
                    verbose)
                if print_interval and (day % print_interval == 0):
                    self.print(verbose)
                end = time.time()
                print("Last day took: ", end - start, "seconds")
                start = time.time()

                # Terminate if tmax reached or num infectious and num exposed is 0:
                numI = sum([self.current_state_count(s)
                            for s in self.unstable_states
                            ])

                if self.t >= self.tmax or numI < 1:
                    self.finalize_data_series()
                    running = False

                day = int(self.t)

        self.print(verbose)
        self.finalize_data_series()
        return True

    # def increase_data_series_length(self):
    #     self.tseries.bloat()
    #     self.history.bloat()
    #     for state in self.states:
    #         self.state_counts[state].bloat()
    #     self.N.bloat()

    # def finalize_data_series(self):
    #     self.tseries.finalize(self.tidx)
    #     self.history.finalize(self.tidx)
    #     for state in self.states:
    #         self.state_counts[state].finalize(self.tidx)
    #     self.N.finalize(self.tidx)
