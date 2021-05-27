import numpy as np
from random_utils import gen_tuple
# models

# model = ENGINE + MODEL DEFINITION

# engine is not cofigurable yet
# you can specify your model definition


class STATES():
    S = 0
    S_s = 1
    E = 2
    I_n = 3
    I_a = 4
    I_s = 5
    J_s = 6
    J_n = 7
    R = 8
    D = 9

    pass


state_codes = {
    STATES.S:     "S",
    STATES.S_s:   "S_s",
    STATES.E:     "E",
    STATES.I_n:   "I_n",
    STATES.I_a:   "I_a",
    STATES.I_s:   "I_s",
    STATES.J_s:   "J_s",
    STATES.J_n:   "J_n",
    STATES.R:   "R",
    STATES.D:   "D",
}


# MODEL DEFINITION

# 1. states, transtion types, parameters
model_definition = {
    # define your model states and transition types
    #
    # define model arguments (arguments of constructor) and parameters (arguments of
    # constuctor)
    # arguments are dictionaries: { arg_name : (default value, description) }
    #  init_arguments   .... model parameters single value
    #                        e.g. "p": (0.2, "probability of external constact")
    #
    #  model_parameters .... model parameters: single value or np.array
    #                        those that can differ for each node
    #                        i.e. "beta": (0.2, "transmission rate")
    #
    # you do note have to define init_{STATE_NAME} arguments, you can use them
    # by default (they define numbers of individuals in individual stats,
    # the rest of population is assing the the first state)

    "states":  [
        STATES.S,
        STATES.S_s,
        STATES.E,
        STATES.I_n,
        STATES.I_a,
        STATES.I_s,
        STATES.J_s,
        STATES.J_n,
        STATES.R,
        STATES.D
    ],

    "state_str_dict": state_codes,

    "transitions":  [
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
    ],

    "final_states": [
        STATES.R,
        STATES.D
    ],

    "invisible_states": [
        STATES.D
    ],

    "unstable_states": [
        STATES.E,
        STATES.I_n,
        STATES.I_a,
        STATES.I_s,
        STATES.J_n,
        STATES.J_s
    ],

    "init_arguments": {
        "p": (0, "probability of interaction outside adjacent nodes"),
        "q": (0, " probability of detected individuals interaction outside adjacent nodes"),
        "false_symptoms_rate": (0, ""),
        "false_symptoms_recovery_rate": (1., ""),
        "asymptomatic_rate": (0, ""),
        "save_nodes": (False, ""),
        "durations_file": ("duration_probs", "file with probs for durations")
    },

    "model_parameters": {
        "beta": (0,  "rate of transmission (exposure)"),
        "beta_reduction": (0,  "todo"),
        "beta_in_family": (0, "todo"),
        "beta_A": (0, "todo"),
        "beta_A_in_family": (0, "todo"),
        "mu": (0, "rate of infection-related death"),
        "theta_E": (0, "rate of baseline testing for exposed individuals"),
        "theta_Ia": (0, "rate of baseline testing for Ia individuals"),
        "theta_Is": (0, "rate of baseline testing for Is individuals"),
        "theta_In": (0, "rate of baseline testing for In individuals"),
        "test_rate": (1.0, ""),
        "psi_E": (0, "probability of positive test results for exposed individuals"),
        "psi_Ia": (0, "probability of positive test results for Ia individuals"),
        "psi_Is": (0, "probability of positive test results for Is individuals"),
        "psi_In": (0, "probability of positive test results for In individuals"),
        "symptomatic_time": (-1, "time_from first_symptom  - do not setup"),
        "infectious_time": (-1, "time_from first_symptom  - do not setup")
    }
}

# 2. step functions


def daily_update(model, nodes):
    """
    Everyday checkup
    """

    # S, S_s
    target_nodes = np.logical_or(
        model._get_target_nodes(nodes, STATES.S),
        model._get_target_nodes(nodes, STATES.S_s)
    )

    # try infection (may rewrite S/Ss moves)
    P_infection = model.prob_of_contact(
        [STATES.S_s, STATES.S],
        [STATES.S,
         STATES.S_s,
         STATES.E,
         STATES.I_n,
         STATES.I_a,
         STATES.I_s
         ],
        [STATES.I_n, STATES.I_a, STATES.I_s],
        [STATES.I_n, STATES.I_a, STATES.I_s, STATES.E],
        model.beta, model.beta_in_family
    ).flatten()

    #    r = np.random.rand(target_nodes.sum())
    exposed = P_infection[target_nodes]
    # print(exposed, exposed.shape)
    # exit()

    exposed_mask = np.zeros(model.num_nodes, dtype=bool)
    exposed_mask[target_nodes] = exposed

    model.time_to_go[exposed_mask] = 1
    model.state_to_go[exposed_mask] = STATES.E

    # print(model.time_to_go.flatten())
    # print(model.state_to_go.flatten())
    # exit()


def testing(model, states):
    """
    Returns True if possitively tested
    """
    return np.logical_or(
        states == STATES.E,
        states == STATES.I_a,
        states == STATES.E,
        states == STATES.I_n,
        states == STATES.I_a,
        states == STATES.I_s,
        states == STATES.J_s,
        states == STATES.J_n
    )


def change_states(model, nodes, target_state=None):
    """
    nodes that just entered a new state, update plan
    """
    # discard current state
    model.memberships[:, nodes == True] = 0

#    print("DBG nodes", nodes == True)

    for node in nodes.nonzero()[0]:
        # print()
        # print(model.state_to_go.shape)
        # print(model.state_to_go)
        # exit()
        if target_state is None:
            new_state = model.state_to_go[node][0]
        else:
            new_state = target_state
        old_state = model.current_state[node, 0]
        # print(f"{new_state} {new_state.shape}")
        model.memberships[new_state, node] = 1
        model.state_counts[new_state][model.t] += 1
        model.state_counts[old_state][model.t] -= 1
        model.states_history[model.t][node] = new_state
        
    if target_state is None:
        model.current_state[nodes] = model.state_to_go[nodes]
    else:
        model.current_state[nodes] = target_state
    update_plan(model, nodes)

    
    # add ones to new states
    # new_states = model.state_to_go[nodes]
    # # # todo refactor to get rid of cycle
    # # for state in new_states:
    # #     model.memberships[state, model.state_to_go == state, 0] == 1

    # # print(model.memberships)


def update_plan(model, nodes):
    # update plan
    # STATES.S:     "S",
    target_nodes = model._get_target_nodes(nodes, STATES.S)
    # print("---")
    # print(target_nodes.shape)
    # print(model.time_to_go.shape)

    model.time_to_go[target_nodes] = np.random.choice(range(1, 600))
    model.state_to_go[target_nodes] = STATES.S_s
    model.need_check[target_nodes] = True

    # STATES.S_s:   "S_s",
    target_nodes = model._get_target_nodes(nodes, STATES.S_s)
    model.time_to_go[target_nodes] = 7
    model.state_to_go[target_nodes] = STATES.S
    model.need_check[target_nodes] = True

    # STATES.E:     "E",
    target_nodes = model._get_target_nodes(nodes, STATES.E)
    # print(f"target nodes {target_nodes.shape}")
    # print(f"model.time_to_go {model.time_to_go.shape}")

    # asymptotic or symptomatic branch?
    r = np.random.rand(target_nodes.sum())
    asymptomatic = r < model.asymptomatic_rate

    asymptomatic_nodes = target_nodes.copy()
    asymptomatic_nodes[target_nodes] = asymptomatic

    symptomatic_nodes = target_nodes.copy()
    symptomatic_nodes[target_nodes] = np.logical_not(asymptomatic)

    model.time_to_go[target_nodes] = model.rngd["E"].get(
        n=(sum(target_nodes), 1))
    model.state_to_go[asymptomatic_nodes] = STATES.I_n
    model.state_to_go[symptomatic_nodes] = STATES.I_a
    model.need_check[target_nodes] = False

    # STATES.I_n:   "I_n",
    # need to generate I duratin and J durations
    target_nodes = model._get_target_nodes(nodes, STATES.I_n)
    n = sum(target_nodes)
    if n > 0:
        expected_i_time, expected_j_time = gen_tuple(
            2,
            (n, 1),
            model.rngd["I"],
            model.rngd["RNA"]
        )

        model.infectious_time[target_nodes] = expected_i_time
        model.rna_time[target_nodes] = expected_j_time
        model.time_to_go[target_nodes] = expected_i_time
        model.state_to_go[target_nodes] = STATES.J_n
        model.need_check[target_nodes] = False

    # STATES.I_a:   "I_a",
    target_nodes = model._get_target_nodes(nodes, STATES.I_a)
    # current infectious time (part of total infectious time)
    expected_a_time, expected_i_time, expected_j_time = gen_tuple(
        3,
        (target_nodes.sum(), 1),
        model.rngd["A"],
        model.rngd["I"],
        model.rngd["RNA"]
    )

    model.infectious_time[target_nodes] = expected_i_time
    assert np.all(expected_a_time < expected_i_time)
    model.symptomatic_time[target_nodes] = expected_i_time - expected_a_time
    model.rna_time[target_nodes] = expected_j_time

    model.time_to_go[target_nodes] = expected_a_time
    model.state_to_go[target_nodes] = STATES.I_s
    model.need_check[target_nodes] = False

    # STATES.I_s:   "I_s",
    target_nodes = model._get_target_nodes(nodes, STATES.I_s)
    assert np.all(model.symptomatic_time[target_nodes] > 0)
    model.time_to_go[target_nodes] = model.symptomatic_time[target_nodes]
    model.state_to_go[target_nodes] = STATES.J_s
    model.need_check[target_nodes] = False

    # STATES.J_s:   "J_s",
    # STATES.J_n:   "J_n",
    target_nodes = np.logical_or(
        model._get_target_nodes(nodes, STATES.J_s),
        model._get_target_nodes(nodes, STATES.J_n)
    )
    # print(target_nodes, target_nodes.shape)
    # print("bye")
    # exit()
    left_rna_positivity = model.rna_time[target_nodes] - \
        model.infectious_time[target_nodes]
#    print("DBG RNA left", left_rna_positivity)
    model.time_to_go[target_nodes] = left_rna_positivity
    model.state_to_go[target_nodes] = STATES.R
    model.need_check[target_nodes] = False

    # STATES.R:   "R",
    target_nodes = model._get_target_nodes(nodes, STATES.R)
    model.time_to_go[target_nodes] = -1
    model.state_to_go[target_nodes] = -1
    model.need_check[target_nodes] = False

    # STATES.D:   "D",
    target_nodes = model._get_target_nodes(nodes, STATES.D)
    model.time_to_go[target_nodes] = -1
    model.state_to_go[target_nodes] = -1
    model.need_check[target_nodes] = False


# 3. model class
# SimulationDrivenModel = create_custom_model("SimulationDrivenModel",
#                                             **model_definition,
#                                             member_functions={
#                                                 "daily_update": daily_update,
#                                                 "testing": testing,
#                                                 "change_states": change_states
#                                             },
#                                             engine=EngineS)
