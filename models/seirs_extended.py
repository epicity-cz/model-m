from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import networkx as nx
import numpy as np
import scipy as scipy
import scipy.integrate

TF_ENABLED = False
if TF_ENABLED:
    import tensorflow as tf
    from tf_utils import to_sparse_tensor


class ExtendedNetworkModel():

    """
    A class to simulate the SEIRS Stochastic Network Model
    extended with additional states
    ===================================================
    Possible states:
       S  - healthy
       SS - healthy with symptoms
       E  -  exposed (not yet infectious)
       I_n - asymptomatic and infectious
       I_a - symptomatic and infectious  (+ no manifest of symptoms)
       I_s - symptomatic and manifest of symptoms
       I_d - detected
       R  - recovered
       D  - death

    Params:
            G       Network adjacency matrix (numpy array) or Networkx graph object

    Original Params:
            beta    Rate of transmission (exposure)
            sigma   Rate of infection (upon exposure)
            gamma   Rate of recovery (upon infection)
            xi      Rate of re-susceptibility (upon recovery)
            mu_I    Rate of infection-related death
            mu_0    Rate of baseline death
            nu      Rate of baseline birth
            p       Probability of interaction outside adjacent nodes

            Q       Quarantine adjacency matrix (numpy array) or Networkx graph object.
            beta_D  Rate of transmission (exposure) for individuals with detected infections
            sigma_D Rate of infection (upon exposure) for individuals with detected infections
            gamma_D Rate of recovery (upon infection) for individuals with detected infections
            mu_D    Rate of infection-related death for individuals with detected infections
            theta_E Rate of baseline testing for exposed individuals
            theta_I Rate of baseline testing for infectious individuals
            phi_E   Rate of contact tracing testing for exposed individuals
            phi_I   Rate of contact tracing testing for infectious individuals
            psi_E   Probability of positive test results for exposed individuals
            psi_I   Probability of positive test results for exposed individuals
            q       Probability of quarantined individuals interaction outside adjacent nodes

            initE   Init number of exposed individuals
            initI   Init number of infectious individuals
            initD_E Init number of detected infectious individuals
            initD_I Init number of detected infectious individuals
            initR   Init number of recovered individuals
            initF   Init number of infection-related fatalities
                    (all remaining nodes initialized susceptible)


    Params of states not included in original SEIRS:
         false_symptoms_rate                healthy individuals with symptoms rate 
         false_symptoms_recovery rate       loosing false symptoms 
         asymptomatic_rate                  rate of infectous individuals without symptoms
         symptoms_manifest_rate             controls manifest of symptoms (from I_a to I_s)
         asymptomatic_testing_rate          detection of asymptomatic individual
         symptomatic_testing_rate           detection of asymptomatic individual
    """

    # Define node states
    states = (
        "S",
        "S_s",
        "E",
        "I_n",
        "I_a",
        "I_s",
        "I_d",
        "R_d",
        "R_u",
        "D_d",
        "D_u"
    )

    final_states = (
        "R_d",
        "R_u",
        "D_d",
        "D_u"
    )

    transitions = (
        ("S", "S_s"),
        ("S", "E"),
        ("S_s", "S"),
        ("S_s", "E"),
        ("E", "I_n"),
        ("E", "I_a"),
        ("I_n", "R_u"),
        ("I_a", "I_s"),
        ("I_s", "R_u"),
        ("I_s", "D_u"),
        ("I_s", "I_d"),
        ("I_d", "R_d"),
        ("I_d", "D_d"),
        ("I_a", "I_d"),
        ("E", "I_d")
    )

    def __init__(self, G,
                 beta, sigma, gamma, mu_I=0, p=0,
                 beta_D=0, gamma_D=0, mu_D=0,
                 theta_E=0, theta_Ia=0, theta_Is=0,
                 phi_E=0, phi_Ia=0, phi_Is=0,
                 psi_E=0, psi_Ia=0, psi_Is=0,
                 q=0,
                 false_symptoms_rate=0, false_symptoms_recovery_rate=1, asymptomatic_rate=0, symptoms_manifest_rate=1.0,
                 init_S=0, init_S_s=0, init_E=0, init_I_n=0, init_I_a=0, init_I_s=0, init_I_d=0, init_R_u=0, init_R_d=0, init_D_u=0, init_D_d=0,
                 random_seed=None):

        if random_seed:
            np.random.seed(random_seed)

        self.periodic_update_callback = None

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Setup Adjacency matrix:
        self.update_G(G)

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Model Parameters:
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        model_param_names = ("beta", "sigma", "gamma", "mu_I", "p", "beta_D", "gamma_D", "mu_D",
                             "theta_E", "theta_Ia", "theta_Is", "phi_E", "phi_Ia", "phi_Is",
                             "psi_E", "psi_Ia", "psi_Is", "q", "false_symptoms_rate",
                             "false_symptoms_recovery_rate",
                             "asymptomatic_rate", "symptoms_manifest_rate")

        for param_name in model_param_names:
            param = locals()[param_name]
            if isinstance(param, (list, np.ndarray)):
                setattr(self, param_name,
                        np.array(param).reshape((self.numNodes, 1)))
            else:
                setattr(self, param_name,
                        np.full(fill_value=param, shape=(self.numNodes, 1)))

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Each node can undergo up to 4 transitions (sans vitality/re-susceptibility returns to S state),
        # so there are ~numNodes*4 events/timesteps expected; initialize numNodes*5 timestep slots to start
        # (will be expanded during run if needed)
        self.num_transitions = 100  # TO: change to our situation
        tseries_len = (self.num_transitions + 1) * self.numNodes
        self.tseries = np.zeros(tseries_len)

        # history of events
        max_state_len = max([len(s) for s in self.states])
        self.history = np.chararray((tseries_len, 3), itemsize=5)

        # instead of original numE, numI, etc.
        # state_counts ... numbers of inidividuals in given states
        self.state_counts = dict()
        for state in self.states:
            self.state_counts[state] = np.zeros(
                tseries_len, dtype=int)

        # N ... actual number of individuals in population
        self.N = np.zeros(tseries_len)

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize Timekeeping:
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        self.t = 0
        self.tmax = 0  # will be set when run() is called
        self.tidx = 0
        self.tseries[0] = 0

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize Counts of inidividuals with each state:
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # self.numE[0] = int(initE)
        # self.numI[0] = int(initI)
        # self.numD_E[0] = int(initD_E)
        # self.numD_I[0] = int(initD_I)
        # self.numR[0] = int(initR)
        # self.numF[0] = int(initF)
        init_values = (init_S, init_S_s, init_E, init_I_n, init_I_a,
                       init_I_s, init_I_d, init_R_u, init_R_d, init_D_u, init_D_d)
        for state, init_value in zip(self.states, init_values):
            self.state_counts[state][0] = int(init_value)

        nodes_left = self.numNodes - sum(init_values)
        self.state_counts["S"][0] += nodes_left
        # self.numS[0] = self.numNodes - self.numE[0] - self.numI[0] - \
        #     self.numD_E[0] - self.numD_I[0] - self.numR[0] - self.numF[0]

        # all individuals except death ones
        self.N[0] = self.numNodes - init_D_u - init_D_d

        # X ... array of states
        tempX = []
        for state, count in self.state_counts.items():
            tempX.extend([state]*count[0])
        self.X = np.array(tempX).reshape((self.numNodes, 1))

        # self.X = np.array([self.S]*int(self.numS[0])
        #                      + [self.E]*int(self.numE[0])
        #                      + [self.I]*int(self.numI[0])
        #                      + [self.D_E]*int(self.numD_E[0])
        #                      + [self.D_I]*int(self.numD_I[0])
        #                      + [self.R]*int(self.numR[0])
        #                      + [self.F]*int(self.numF[0])
        # ).reshape((self.numNodes, 1))

        np.random.shuffle(self.X)

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize scenario flags:
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        self.update_scenario_flags()

    # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    def set_periodic_update(callback):
        self.periodic_update_callback = callback

    def node_degrees(self, Amat):
        # sums of adj matrix cols
        return Amat.sum(axis=0).reshape(self.numNodes, 1)

    # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    def update_G(self, new_G):
        if isinstance(new_G, np.ndarray):
            self.A = scipy.sparse.csr_matrix(new_G)
        elif type(new_G) == nx.classes.graph.Graph:
            # adj_matrix gives scipy.sparse csr_matrix
            self.A = nx.adj_matrix(new_G)
        else:
            raise TypeError(
                "Input an adjacency matrix or networkx object only.")

        self.numNodes = int(self.A.shape[1])
        self.degree = np.asarray(self.node_degrees(self.A)).astype(float)

        if TF_ENABLED:
            self.A = to_sparse_tensor(self.A)

    # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

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

    # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    # return numbers of contacts in given state

    def num_contacts(self, state):

        if type(state) == str:
            if TF_ENABLED:
                with tf.device('/GPU:' + "0"):
                    x = tf.Variable(self.X == state, dtype="float32")
                    return tf.sparse.sparse_dense_matmul(self.A, x)
            else:
                return np.asarray(
                    scipy.sparse.csr_matrix.dot(self.A, self.X == state))

        elif type(state) == list:
            state_flags = np.hstack(
                [np.array(self.X == s, dtype=int) for s in state]
            )
            if TF_ENABLED:
                with tf.device('/GPU:' + "0"):
                    x = tf.Variable(state_flags, dtype="float32")
                    nums = tf.sparse.sparse_dense_matmul(self.A, x)
            else:
                nums = scipy.sparse.csr_matrix.dot(self.A, state_flags)
                return np.sum(nums, axis=1).reshape((self.numNodes, 1))
            return np.sum(nums, axis=1).reshape((self.numNodes, 1))

        else:
            raise TypeException(
                "num_contacts(state) accepts str or list of strings")

    def current_state_count(self, state):
        return self.state_counts[state][self.tidx]

    def current_N(self):
        return self.N[self.tidx]

    def calc_propensities(self):

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Pre-calculate matrix multiplication terms that may be used in multiple propensity calculations,
        # and check to see if their computation is necessary before doing the multiplication

        # number of infectious nondetected contacts
        # sum of all I states
        numContacts_I = np.zeros(shape=(self.numNodes, 1))
        if any(self.beta):
            infected = [
                s for s in ("I_n", "I_a", "I_s")
                if self.current_state_count(s)
            ]
            if infected:
                numContacts_I = self.num_contacts(infected)

        numContacts_Id = np.zeros(shape=(self.numNodes, 1))
        if any(self.beta_D):
            numContacts_Id = self.num_contacts("I_d")

        # numQuarantineContacts_DI = np.zeros(shape=(self.numNodes, 1))
        # if(self.testing_scenario
        #         and np.any(self.numD_I[self.tidx])
        #         and np.any(self.beta_D)):
        #     numQuarantineContacts_DI = np.asarray(
        #         scipy.sparse.csr_matrix.dot(self.A_Q, self.X == self.D_I))

        # numContacts_D = np.zeros(shape=(self.numNodes, 1))
        # if(self.tracing_scenario
        #         and (np.any(self.numD_E[self.tidx]) or np.any(self.numD_I[self.tidx]))):
        #     numContacts_D = np.asarray(scipy.sparse.csr_matrix.dot(self.A, self.X == self.D_E)
        #                                + scipy.sparse.csr_matrix.dot(self.A, self.X == self.D_I))

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        propensities = dict()

        #  "S" ->  "S_s"
        propensities[("S", "S_s")] = self.false_symptoms_rate*(self.X == "S")

        #  "S" -> "E"
        numI = self.current_state_count(
            "I_n") + self.current_state_count("I_a") + self.current_state_count("I_s")

        S_to_E_koef = (
            self.p * (
                self.beta * numI +
                self.q * self.beta_D * self.current_state_count("I_d")
            ) / self.current_N()
            +
            (1 - self.p) * np.divide(
                self.beta * numContacts_I +
                self.beta_D * numContacts_Id, self.degree, out=np.zeros_like(self.degree), where=self.degree != 0
            )
        )
        propensities[("S", "E")] = S_to_E_koef * (self.X == "S")

        # propensities_StoE = (self.p*((self.beta*self.numI[self.tidx] + self.q*self.beta_D*self.numD_I[self.tidx])/self.N[self.tidx])
        #                      + (1-self.p)*np.divide((self.beta*numContacts_I + self.beta_D*numQuarantineContacts_DI),
        #                                             self.degree, out=np.zeros_like(self.degree), where=self.degree != 0)
        #                      )*(self.X == self.S)

        propensities[("S_s", "S")
                     ] = self.false_symptoms_recovery_rate*(self.X == "S_s")

        # becoming exposed does not depend on unrelated symptoms
        propensities[("S_s", "E")] = S_to_E_koef * (self.X == "S_s")

        exposed = self.X == "E"
        propensities[("E", "I_n")] = self.asymptomatic_rate * \
            self.sigma * exposed
        propensities[("E", "I_a")] = (
            1-self.asymptomatic_rate) * self.sigma * exposed

        propensities[("I_n", "R_u")] = self.gamma * (self.X == "I_n")

        asymptomatic = self.X == "I_a"
        propensities[("I_a", "I_s")
                     ] = self.symptoms_manifest_rate * asymptomatic

        symptomatic = self.X == "I_s"
        propensities[("I_s", "R_u")] = self.gamma * symptomatic
        propensities[("I_s", "D_u")] = self.mu_I * symptomatic

        detected = self.X == "I_d"
        propensities[("I_d", "R_d")] = self.gamma_D * detected
        propensities[("I_d", "D_d")] = self.mu_D * detected

        # testing  TODO
        propensities[("I_a", "I_d")] = (
            self.theta_Ia + self.phi_Ia * numContacts_Id) * self.psi_Ia * asymptomatic

        propensities[("I_s", "I_d")] = (
            self.theta_Is + self.phi_Is * numContacts_Id) * self.psi_Is * symptomatic

        propensities[("E", "I_d")] = (
            self.theta_E + self.phi_E * numContacts_Id) * self.psi_E * exposed

        propensities_list = []
        for t in self.transitions:
            propensities_list.append(propensities[t])

        stacked_propensities = np.hstack(propensities_list)

        return stacked_propensities, self.transitions

    def increase_data_series_length(self):
        self.tseries = np.pad(
            self.tseries,
            [(0, (self.num_transitions+1)*self.numNodes)],
            mode='constant', constant_values=0)

        new_history = np.empty(((self.num_transitions+1)*self.numNodes, 3))
        self.history = np.vstack([self.history, new_history])

        for state in self.states:
            self.state_counts[state] = np.pad(
                self.state_counts[state], [
                    (0, (self.num_transtions+1)*self.numNodes)],
                mode='constant', constant_values=int(0))

        self.N = np.pad(
            self.N, [(0, (self.num_transitions+1)*self.numNodes)], mode='constant', constant_values=0)

        return None

# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    def finalize_data_series(self):
        """ throw away ending zeros """
        self.tseries = np.array(self.tseries, dtype=float)[:self.tidx+1]

        self.history = self.history[:self.tidx+1]

        for state in self.states:
            self.state_counts[state] = np.array(self.state_counts[state],
                                                dtype=int)[:self.tidx+1]

        self.N = np.array(self.N, dtype=int)[:self.tidx+1]
        return None

# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    def run_iteration(self):

        if(self.tidx >= len(self.tseries)-1):
            # Room has run out in the timeseries storage arrays; double the size of these arrays:
            self.increase_data_series_length()

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # 1. Generate 2 random numbers uniformly distributed in (0,1)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        r1 = np.random.rand()
        r2 = np.random.rand()

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # 2. Calculate propensities
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        propensities, transitionTypes = self.calc_propensities()

        # Terminate when probability of all events is 0:
        if propensities.sum() <= 0.0:
            self.finalize_data_series()
            return False

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # 3. Calculate alpha
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        propensities_flat = propensities.ravel(order='F')
        cumsum = propensities_flat.cumsum()
        alpha = propensities_flat.sum()

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # 4. Compute the time until the next event takes place
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        tau = (1/alpha)*np.log(float(1/r1))
        self.t += tau

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # 5. Compute which event takes place
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        transitionIdx = np.searchsorted(cumsum, r2*alpha)
        transitionNode = transitionIdx % self.numNodes
        transitionType = transitionTypes[int(transitionIdx/self.numNodes)]

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # 6. Update node states and data series
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        assert(self.X[transitionNode] == transitionType[0] and self.X[transitionNode] not in self.final_states), "Assertion error: Node " + \
            str(transitionNode)+" has unexpected current state " + \
            str(self.X[transitionNode]) + \
            " given the intended transition of "+str(transitionType)+"."

        self.X[transitionNode] = transitionType[1]
        self.tidx += 1

        self.tseries[self.tidx] = self.t

        self.history[self.tidx] = [transitionNode,
                                   transitionType[0], transitionType[1]]

        for state in self.states:
            self.state_counts[state][self.tidx] = self.state_counts[state][self.tidx-1]

        self.state_counts[transitionType[0]][self.tidx] -= 1
        self.state_counts[transitionType[1]][self.tidx] += 1

        # if somebody died
        if transitionType[1] in ("D_u", "D_d"):
            self.N[self.tidx] = self.N[self.tidx-1] - 1
        else:
            self.N[self.tidx] = self.N[self.tidx-1]

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Terminate if tmax reached or num infectious and num exposed is 0:
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        numI = (self.current_state_count("I_n") +
                self.current_state_count("I_a") +
                self.current_state_count("I_s") +
                self.current_state_count("I_d")
                )

        if self.t >= self.tmax or (numI < 1 and self.current_state_count("E") < 1):
            self.finalize_data_series()
            return False

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        return True


# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    def run(self, T, checkpoints=None, print_interval=10, verbose=False):
        if(T > 0):
            self.tmax += T
        else:
            return False

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Pre-process checkpoint values:
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # if(checkpoints):
        #     numCheckpoints=len(checkpoints['t'])
        #     paramNames=['G', 'beta', 'sigma', 'gamma', 'xi', 'mu_I', 'mu_0', 'nu', 'p',
        #                   'Q', 'beta_D', 'sigma_D', 'gamma_D', 'mu_D', 'q',
        #                   'theta_E', 'theta_I', 'phi_E', 'phi_I', 'psi_E', 'psi_I']
        #     for chkpt_param, chkpt_values in checkpoints.items():
        #         assert(isinstance(chkpt_values, (list, np.ndarray)) and len(chkpt_values) ==
        #                numCheckpoints), "Expecting a list of values with length equal to number of checkpoint times ("+str(numCheckpoints)+") for each checkpoint parameter."
        #     # Finds 1st index in list greater than given val
        #     checkpointIdx=np.searchsorted(checkpoints['t'], self.t)
        #     if(checkpointIdx >= numCheckpoints):
        #         # We are out of checkpoints, stop checking them:
        #         checkpoints=None
        #     else:
        #         checkpointTime=checkpoints['t'][checkpointIdx]

        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        # Run the simulation loop:
        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        running = True
        day = -1

        while running:

            running = self.run_iteration()

            # true after the first event after midnight
            day_changed = day != int(self.t)

            # run periodical update
            if self.periodic_update_callback and day != 1 and day_changed:
                self.periodic_update_callback()

            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # print only if print_interval is set
            # prints always at the beginning of a new day
            if print_interval or not running:
                if day_changed:
                    day = int(self.t)

                if not running or (day_changed and (day % print_interval == 0)):
                    print("t = %.2f" % self.t)
                    if verbose or not running:
                        for state in self.states:
                            print(f"\t {state} = {self.current_state_count(state)}")
                    print(flush=True)

        return True


# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

#     def plot(self, ax=None,  plot_S='line', plot_E='line', plot_I='line', plot_R='line', plot_F='line',
#              plot_D_E='line', plot_D_I='line', combine_D=True,
#              color_S='tab:green', color_E='orange', color_I='crimson', color_R='tab:blue', color_F='black',
#              color_D_E='mediumorchid', color_D_I='mediumorchid', color_reference='#E0E0E0',
#              dashed_reference_results=None, dashed_reference_label='reference',
#              shaded_reference_results=None, shaded_reference_label='reference',
#              vlines=[], vline_colors=[], vline_styles=[], vline_labels=[],
#              ylim=None, xlim=None, legend=True, title=None, side_title=None, plot_percentages=True):

#         import matplotlib.pyplot as pyplot

#         # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#         # Create an Axes object if None provided:
#         # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#         if(not ax):
#             fig, ax=pyplot.subplots()

#         # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#         # Prepare data series to be plotted:
#         # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#         Fseries=self.numF/self.numNodes if plot_percentages else self.numF
#         Eseries=self.numE/self.numNodes if plot_percentages else self.numE
#         Dseries=(self.numD_E+self.numD_I) / \
#             self.numNodes if plot_percentages else (self.numD_E+self.numD_I)
#         D_Eseries=self.numD_E/self.numNodes if plot_percentages else self.numD_E
#         D_Iseries=self.numD_I/self.numNodes if plot_percentages else self.numD_I
#         Iseries=self.numI/self.numNodes if plot_percentages else self.numI
#         Rseries=self.numR/self.numNodes if plot_percentages else self.numR
#         Sseries=self.numS/self.numNodes if plot_percentages else self.numS

#         # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#         # Draw the reference data:
#         # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#         if(dashed_reference_results):
#             dashedReference_tseries=dashed_reference_results.tseries[::int(
#                 self.numNodes/100)]
#             dashedReference_IDEstack=(dashed_reference_results.numI + dashed_reference_results.numD_I + dashed_reference_results.numD_E +
#                                         dashed_reference_results.numE)[::int(self.numNodes/100)] / (self.numNodes if plot_percentages else 1)
#             ax.plot(dashedReference_tseries, dashedReference_IDEstack, color='#E0E0E0',
#                     linestyle='--', label='$I+D+E$ ('+dashed_reference_label+')', zorder=0)
#         if(shaded_reference_results):
#             shadedReference_tseries=shaded_reference_results.tseries
#             shadedReference_IDEstack=(shaded_reference_results.numI + shaded_reference_results.numD_I +
#                                         shaded_reference_results.numD_E + shaded_reference_results.numE) / (self.numNodes if plot_percentages else 1)
#             ax.fill_between(shaded_reference_results.tseries, shadedReference_IDEstack,
#                             0, color='#EFEFEF', label='$I+D+E$ ('+shaded_reference_label+')', zorder=0)
#             ax.plot(shaded_reference_results.tseries,
#                     shadedReference_IDEstack, color='#E0E0E0', zorder=1)

#         # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#         # Draw the stacked variables:
#         # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#         topstack=np.zeros_like(self.tseries)
#         if(any(Fseries) and plot_F == 'stacked'):
#             ax.fill_between(np.ma.masked_where(Fseries <= 0, self.tseries), np.ma.masked_where(
#                 Fseries <= 0, topstack+Fseries), topstack, color=color_F, alpha=0.5, label='$F$', zorder=2)
#             ax.plot(np.ma.masked_where(Fseries <= 0, self.tseries), np.ma.masked_where(
#                 Fseries <= 0, topstack+Fseries),           color=color_F, zorder=3)
#             topstack=topstack+Fseries
#         if(any(Eseries) and plot_E == 'stacked'):
#             ax.fill_between(np.ma.masked_where(Eseries <= 0, self.tseries), np.ma.masked_where(
#                 Eseries <= 0, topstack+Eseries), topstack, color=color_E, alpha=0.5, label='$E$', zorder=2)
#             ax.plot(np.ma.masked_where(Eseries <= 0, self.tseries), np.ma.masked_where(
#                 Eseries <= 0, topstack+Eseries),           color=color_E, zorder=3)
#             topstack=topstack+Eseries
#         if(combine_D and plot_D_E == 'stacked' and plot_D_I == 'stacked'):
#             ax.fill_between(np.ma.masked_where(Dseries <= 0, self.tseries), np.ma.masked_where(
#                 Dseries <= 0, topstack+Dseries), topstack, color=color_D_E, alpha=0.5, label='$D_{all}$', zorder=2)
#             ax.plot(np.ma.masked_where(Dseries <= 0, self.tseries), np.ma.masked_where(
#                 Dseries <= 0, topstack+Dseries),           color=color_D_E, zorder=3)
#             topstack=topstack+Dseries
#         else:
#             if(any(D_Eseries) and plot_D_E == 'stacked'):
#                 ax.fill_between(np.ma.masked_where(D_Eseries <= 0, self.tseries), np.ma.masked_where(
#                     D_Eseries <= 0, topstack+D_Eseries), topstack, color=color_D_E, alpha=0.5, label='$D_E$', zorder=2)
#                 ax.plot(np.ma.masked_where(D_Eseries <= 0, self.tseries), np.ma.masked_where(
#                     D_Eseries <= 0, topstack+D_Eseries),           color=color_D_E, zorder=3)
#                 topstack=topstack+D_Eseries
#             if(any(D_Iseries) and plot_D_I == 'stacked'):
#                 ax.fill_between(np.ma.masked_where(D_Iseries <= 0, self.tseries), np.ma.masked_where(
#                     D_Iseries <= 0, topstack+D_Iseries), topstack, color=color_D_I, alpha=0.5, label='$D_I$', zorder=2)
#                 ax.plot(np.ma.masked_where(D_Iseries <= 0, self.tseries), np.ma.masked_where(
#                     D_Iseries <= 0, topstack+D_Iseries),           color=color_D_I, zorder=3)
#                 topstack=topstack+D_Iseries
#         if(any(Iseries) and plot_I == 'stacked'):
#             ax.fill_between(np.ma.masked_where(Iseries <= 0, self.tseries), np.ma.masked_where(
#                 Iseries <= 0, topstack+Iseries), topstack, color=color_I, alpha=0.5, label='$I$', zorder=2)
#             ax.plot(np.ma.masked_where(Iseries <= 0, self.tseries), np.ma.masked_where(
#                 Iseries <= 0, topstack+Iseries),           color=color_I, zorder=3)
#             topstack=topstack+Iseries
#         if(any(Rseries) and plot_R == 'stacked'):
#             ax.fill_between(np.ma.masked_where(Rseries <= 0, self.tseries), np.ma.masked_where(
#                 Rseries <= 0, topstack+Rseries), topstack, color=color_R, alpha=0.5, label='$R$', zorder=2)
#             ax.plot(np.ma.masked_where(Rseries <= 0, self.tseries), np.ma.masked_where(
#                 Rseries <= 0, topstack+Rseries),           color=color_R, zorder=3)
#             topstack=topstack+Rseries
#         if(any(Sseries) and plot_S == 'stacked'):
#             ax.fill_between(np.ma.masked_where(Sseries <= 0, self.tseries), np.ma.masked_where(
#                 Sseries <= 0, topstack+Sseries), topstack, color=color_S, alpha=0.5, label='$S$', zorder=2)
#             ax.plot(np.ma.masked_where(Sseries <= 0, self.tseries), np.ma.masked_where(
#                 Sseries <= 0, topstack+Sseries),           color=color_S, zorder=3)
#             topstack=topstack+Sseries

#         # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#         # Draw the shaded variables:
#         # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#         if(any(Fseries) and plot_F == 'shaded'):
#             ax.fill_between(np.ma.masked_where(Fseries <= 0, self.tseries), np.ma.masked_where(
#                 Fseries <= 0, Fseries), 0, color=color_F, alpha=0.5, label='$F$', zorder=4)
#             ax.plot(np.ma.masked_where(Fseries <= 0, self.tseries), np.ma.masked_where(
#                 Fseries <= 0, Fseries),    color=color_F, zorder=5)
#         if(any(Eseries) and plot_E == 'shaded'):
#             ax.fill_between(np.ma.masked_where(Eseries <= 0, self.tseries), np.ma.masked_where(
#                 Eseries <= 0, Eseries), 0, color=color_E, alpha=0.5, label='$E$', zorder=4)
#             ax.plot(np.ma.masked_where(Eseries <= 0, self.tseries), np.ma.masked_where(
#                 Eseries <= 0, Eseries),    color=color_E, zorder=5)
#         if(combine_D and (any(Dseries) and plot_D_E == 'shaded' and plot_D_E == 'shaded')):
#             ax.fill_between(np.ma.masked_where(Dseries <= 0, self.tseries), np.ma.masked_where(
#                 Dseries <= 0, Dseries), 0, color=color_D_E, alpha=0.5, label='$D_{all}$', zorder=4)
#             ax.plot(np.ma.masked_where(Dseries <= 0, self.tseries), np.ma.masked_where(
#                 Dseries <= 0, Dseries),    color=color_D_E, zorder=5)
#         else:
#             if(any(D_Eseries) and plot_D_E == 'shaded'):
#                 ax.fill_between(np.ma.masked_where(D_Eseries <= 0, self.tseries), np.ma.masked_where(
#                     D_Eseries <= 0, D_Eseries), 0, color=color_D_E, alpha=0.5, label='$D_E$', zorder=4)
#                 ax.plot(np.ma.masked_where(D_Eseries <= 0, self.tseries), np.ma.masked_where(
#                     D_Eseries <= 0, D_Eseries),    color=color_D_E, zorder=5)
#             if(any(D_Iseries) and plot_D_I == 'shaded'):
#                 ax.fill_between(np.ma.masked_where(D_Iseries <= 0, self.tseries), np.ma.masked_where(
#                     D_Iseries <= 0, D_Iseries), 0, color=color_D_I, alpha=0.5, label='$D_I$', zorder=4)
#                 ax.plot(np.ma.masked_where(D_Iseries <= 0, self.tseries), np.ma.masked_where(
#                     D_Iseries <= 0, D_Iseries),    color=color_D_I, zorder=5)
#         if(any(Iseries) and plot_I == 'shaded'):
#             ax.fill_between(np.ma.masked_where(Iseries <= 0, self.tseries), np.ma.masked_where(
#                 Iseries <= 0, Iseries), 0, color=color_I, alpha=0.5, label='$I$', zorder=4)
#             ax.plot(np.ma.masked_where(Iseries <= 0, self.tseries), np.ma.masked_where(
#                 Iseries <= 0, Iseries),    color=color_I, zorder=5)
#         if(any(Sseries) and plot_S == 'shaded'):
#             ax.fill_between(np.ma.masked_where(Sseries <= 0, self.tseries), np.ma.masked_where(
#                 Sseries <= 0, Sseries), 0, color=color_S, alpha=0.5, label='$S$', zorder=4)
#             ax.plot(np.ma.masked_where(Sseries <= 0, self.tseries), np.ma.masked_where(
#                 Sseries <= 0, Sseries),    color=color_S, zorder=5)
#         if(any(Rseries) and plot_R == 'shaded'):
#             ax.fill_between(np.ma.masked_where(Rseries <= 0, self.tseries), np.ma.masked_where(
#                 Rseries <= 0, Rseries), 0, color=color_R, alpha=0.5, label='$R$', zorder=4)
#             ax.plot(np.ma.masked_where(Rseries <= 0, self.tseries), np.ma.masked_where(
#                 Rseries <= 0, Rseries),    color=color_R, zorder=5)

#         # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#         # Draw the line variables:
#         # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#         if(any(Fseries) and plot_F == 'line'):
#             ax.plot(np.ma.masked_where(Fseries <= 0, self.tseries), np.ma.masked_where(
#                 Fseries <= 0, Fseries), color=color_F, label='$F$', zorder=6)
#         if(any(Eseries) and plot_E == 'line'):
#             ax.plot(np.ma.masked_where(Eseries <= 0, self.tseries), np.ma.masked_where(
#                 Eseries <= 0, Eseries), color=color_E, label='$E$', zorder=6)
#         if(combine_D and (any(Dseries) and plot_D_E == 'line' and plot_D_E == 'line')):
#             ax.plot(np.ma.masked_where(Dseries <= 0, self.tseries), np.ma.masked_where(
#                 Dseries <= 0, Dseries), color=color_D_E, label='$D_{all}$', zorder=6)
#         else:
#             if(any(D_Eseries) and plot_D_E == 'line'):
#                 ax.plot(np.ma.masked_where(D_Eseries <= 0, self.tseries), np.ma.masked_where(
#                     D_Eseries <= 0, D_Eseries), color=color_D_E, label='$D_E$', zorder=6)
#             if(any(D_Iseries) and plot_D_I == 'line'):
#                 ax.plot(np.ma.masked_where(D_Iseries <= 0, self.tseries), np.ma.masked_where(
#                     D_Iseries <= 0, D_Iseries), color=color_D_I, label='$D_I$', zorder=6)
#         if(any(Iseries) and plot_I == 'line'):
#             ax.plot(np.ma.masked_where(Iseries <= 0, self.tseries), np.ma.masked_where(
#                 Iseries <= 0, Iseries), color=color_I, label='$I$', zorder=6)
#         if(any(Sseries) and plot_S == 'line'):
#             ax.plot(np.ma.masked_where(Sseries <= 0, self.tseries), np.ma.masked_where(
#                 Sseries <= 0, Sseries), color=color_S, label='$S$', zorder=6)
#         if(any(Rseries) and plot_R == 'line'):
#             ax.plot(np.ma.masked_where(Rseries <= 0, self.tseries), np.ma.masked_where(
#                 Rseries <= 0, Rseries), color=color_R, label='$R$', zorder=6)

#         # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#         # Draw the vertical line annotations:
#         # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#         if(len(vlines) > 0 and len(vline_colors) == 0):
#             vline_colors=['gray']*len(vlines)
#         if(len(vlines) > 0 and len(vline_labels) == 0):
#             vline_labels=[None]*len(vlines)
#         if(len(vlines) > 0 and len(vline_styles) == 0):
#             vline_styles=[':']*len(vlines)
#         for vline_x, vline_color, vline_style, vline_label in zip(vlines, vline_colors, vline_styles, vline_labels):
#             if(vline_x is not None):
#                 ax.axvline(x=vline_x, color=vline_color,
#                            linestyle=vline_style, alpha=1, label=vline_label)

#         # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#         # Draw the plot labels:
#         # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#         ax.set_xlabel('days')
#         ax.set_ylabel(
#             'percent of population' if plot_percentages else 'number of individuals')
#         ax.set_xlim(0, (max(self.tseries) if not xlim else xlim))
#         ax.set_ylim(0, ylim)
#         if(plot_percentages):
#             ax.set_yticklabels(['{:,.0%}'.format(y) for y in ax.get_yticks()])
#         if(legend):
#             legend_handles, legend_labels=ax.get_legend_handles_labels()
#             ax.legend(legend_handles[::-1], legend_labels[::-1], loc='upper right',
#                       facecolor='white', edgecolor='none', framealpha=0.9, prop={'size': 8})
#         if(title):
#             ax.set_title(title, size=12)
#         if(side_title):
#             ax.annotate(side_title, (0, 0.5), xytext=(-45, 0), ha='right', va='center',
#                         size=12, rotation=90, xycoords='axes fraction', textcoords='offset points')

#         return ax


# # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

#     def figure_basic(self, plot_S='line', plot_E='line', plot_I='line', plot_R='line', plot_F='line',
#                      plot_D_E='line', plot_D_I='line', combine_D=True,
#                      color_S='tab:green', color_E='orange', color_I='crimson', color_R='tab:blue', color_F='black',
#                      color_D_E='mediumorchid', color_D_I='mediumorchid', color_reference='#E0E0E0',
#                      dashed_reference_results=None, dashed_reference_label='reference',
#                      shaded_reference_results=None, shaded_reference_label='reference',
#                      vlines=[], vline_colors=[], vline_styles=[], vline_labels=[],
#                      ylim=None, xlim=None, legend=True, title=None, side_title=None, plot_percentages=True,
#                      figsize=(12, 8), use_seaborn=True):

#         import matplotlib.pyplot as pyplot

#         fig, ax=pyplot.subplots(figsize=figsize)

#         if(use_seaborn):
#             import seaborn
#             seaborn.set_style('ticks')
#             seaborn.despine()

#         self.plot(ax=ax, plot_S=plot_S, plot_E=plot_E, plot_I=plot_I, plot_R=plot_R, plot_F=plot_F,
#                   plot_D_E=plot_D_E, plot_D_I=plot_D_I, combine_D=combine_D,
#                   color_S=color_S, color_E=color_E, color_I=color_I, color_R=color_R, color_F=color_F,
#                   color_D_E=color_D_E, color_D_I=color_D_I, color_reference=color_reference,
#                   dashed_reference_results=dashed_reference_results, dashed_reference_label=dashed_reference_label,
#                   shaded_reference_results=shaded_reference_results, shaded_reference_label=shaded_reference_label,
#                   vlines=vlines, vline_colors=vline_colors, vline_styles=vline_styles, vline_labels=vline_labels,
#                   ylim=ylim, xlim=xlim, legend=legend, title=title, side_title=side_title, plot_percentages=plot_percentages)

#         pyplot.show()


# # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

#     def figure_infections(self, plot_S=False, plot_E='stacked', plot_I='stacked', plot_R=False, plot_F=False,
#                           plot_D_E='stacked', plot_D_I='stacked', combine_D=True,
#                           color_S='tab:green', color_E='orange', color_I='crimson', color_R='tab:blue', color_F='black',
#                           color_D_E='mediumorchid', color_D_I='mediumorchid', color_reference='#E0E0E0',
#                           dashed_reference_results=None, dashed_reference_label='reference',
#                           shaded_reference_results=None, shaded_reference_label='reference',
#                           vlines=[], vline_colors=[], vline_styles=[], vline_labels=[],
#                           ylim=None, xlim=None, legend=True, title=None, side_title=None, plot_percentages=True,
#                           figsize=(12, 8), use_seaborn=True):

#         import matplotlib.pyplot as pyplot

#         fig, ax=pyplot.subplots(figsize=figsize)

#         if(use_seaborn):
#             import seaborn
#             seaborn.set_style('ticks')
#             seaborn.despine()

#         self.plot(ax=ax, plot_S=plot_S, plot_E=plot_E, plot_I=plot_I, plot_R=plot_R, plot_F=plot_F,
#                   plot_D_E=plot_D_E, plot_D_I=plot_D_I, combine_D=combine_D,
#                   color_S=color_S, color_E=color_E, color_I=color_I, color_R=color_R, color_F=color_F,
#                   color_D_E=color_D_E, color_D_I=color_D_I, color_reference=color_reference,
#                   dashed_reference_results=dashed_reference_results, dashed_reference_label=dashed_reference_label,
#                   shaded_reference_results=shaded_reference_results, shaded_reference_label=shaded_reference_label,
#                   vlines=vlines, vline_colors=vline_colors, vline_styles=vline_styles, vline_labels=vline_labels,
#                   ylim=ylim, xlim=xlim, legend=legend, title=title, side_title=side_title, plot_percentages=plot_percentages)

#         pyplot.show()


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Define a custom method for generating
# power-law-like graphs with exponential tails
# both above and below the degree mean and
# where the mean degree be easily down-shifted
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def custom_exponential_graph(base_graph=None, scale=100, min_num_edges=0, m=9, n=None):
    # Generate a random preferential attachment power law graph as a starting point.
    # By the way this graph is constructed, it is expected to have 1 connected component.
    # Every node is added along with m=8 edges, so the min degree is m=8.
    if(base_graph):
        graph = base_graph.copy()
    else:
        assert(n is not None), "Argument n (number of nodes) must be provided when no base graph is given."
        graph = nx.barabasi_albert_graph(n=n, m=m)

    # To get a graph with power-law-esque properties but without the fixed minimum degree,
    # We modify the graph by probabilistically dropping some edges from each node.
    for node in graph:
        neighbors = list(graph[node].keys())
        quarantineEdgeNum = int(max(min(np.random.exponential(
            scale=scale, size=1), len(neighbors)), min_num_edges))
        quarantineKeepNeighbors = np.random.choice(
            neighbors, size=quarantineEdgeNum, replace=False)
        for neighbor in neighbors:
            if(neighbor not in quarantineKeepNeighbors):
                graph.remove_edge(node, neighbor)

    return graph

# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


def plot_degree_distn(graph, max_degree=None, show=True, use_seaborn=True):
    import matplotlib.pyplot as pyplot
    if(use_seaborn):
        import seaborn
        seaborn.set_style('ticks')
        seaborn.despine()
    # Get a list of the node degrees:
    if type(graph) == np.ndarray:
        nodeDegrees = graph.sum(axis=0).reshape(
            (graph.shape[0], 1))   # sums of adj matrix cols
    elif type(graph) == nx.classes.graph.Graph:
        nodeDegrees = [d[1] for d in graph.degree()]
    else:
        raise BaseException(
            "Input an adjacency matrix or networkx object only.")
    # Calculate the mean degree:
    meanDegree = np.mean(nodeDegrees)
    # Generate a histogram of the node degrees:
    pyplot.hist(nodeDegrees, bins=range(max(nodeDegrees)), alpha=0.5,
                color='tab:blue', label=('mean degree = %.1f' % meanDegree))
    pyplot.xlim(0, max(nodeDegrees) if not max_degree else max_degree)
    pyplot.xlabel('degree')
    pyplot.ylabel('num nodes')
    pyplot.legend(loc='upper right')
    if(show):
        pyplot.show()
