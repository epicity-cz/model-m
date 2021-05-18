import numpy as np
from model import create_custom_model
from engine_daily import DailyEngine
from engine_sequential import SequentialEngine
from engine_m import EngineM
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
    I_ds = 6
    J_s = 11
    J_n = 12
    E_d = 13
    I_da = 14
    I_dn = 15
    J_ds = 16
    J_dn = 17
    R_d = 7
    R_u = 8
    D_d = 9
    D_u = 10

    detected = {
        I_ds,
        E_d,
        I_da,
        I_dn,
        J_ds,
        J_dn
    }

    pass


state_codes = {
    STATES.S:     "S",
    STATES.S_s:   "S_s",
    STATES.E:     "E",
    STATES.I_n:   "I_n",
    STATES.I_a:   "I_a",
    STATES.I_s:   "I_s",
    STATES.I_ds:  "I_ds",
    STATES.J_s:   "J_s",
    STATES.J_n:   "J_n",
    STATES.E_d:   "E_d",
    STATES.I_da:  "I_da",
    STATES.I_dn:  "I_dn",
    STATES.J_ds:  "J_ds",
    STATES.J_dn:  "J_dn",
    STATES.R_d:   "R_d",
    STATES.R_u:   "R_u",
    STATES.D_d:   "D_d",
    STATES.D_u:   "D_u"
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
        STATES.I_ds,
        STATES.J_s,
        STATES.J_n,
        STATES.E_d,
        STATES.I_da,
        STATES.I_dn,
        STATES.J_ds,
        STATES.J_dn,
        STATES.R_d,
        STATES.R_u,
        STATES.D_d,
        STATES.D_u
    ],

    "state_str_dict": state_codes,

    "transitions":  [
        (STATES.S_s,  STATES.E), #0
        (STATES.S_s,  STATES.S), #1
        (STATES.S_s,  STATES.S_s), #2


        (STATES.S, STATES.E),    #3
        (STATES.S, STATES.S_s),  #4
        (STATES.S, STATES.S),    #5

        (STATES.E, STATES.E_d), #6
        (STATES.E, STATES.I_n),  #7
        (STATES.E, STATES.I_a),  #8
        (STATES.E, STATES.E),    #9


        (STATES.I_n, STATES.J_n),
        (STATES.I_n, STATES.I_dn),
        (STATES.I_n, STATES.I_n),

        (STATES.I_a, STATES.I_da),
        (STATES.I_a, STATES.I_s),
        (STATES.I_a, STATES.I_a),

        (STATES.I_s, STATES.J_s),
        (STATES.I_s, STATES.D_u),
        (STATES.I_s, STATES.I_ds),
        (STATES.I_s, STATES.I_s),

        (STATES.I_ds, STATES.J_ds),
        (STATES.I_ds, STATES.D_d),
        (STATES.I_ds, STATES.I_ds),

        (STATES.I_dn, STATES.J_dn),
        (STATES.I_dn, STATES.D_d),
        (STATES.I_dn, STATES.I_dn),

        (STATES.I_da, STATES.I_ds),
        (STATES.I_da, STATES.I_da),

        (STATES.E_d, STATES.I_dn),
        (STATES.E_d, STATES.I_da),
        (STATES.E_d, STATES.E_d),


        (STATES.J_s, STATES.R_u),
        (STATES.J_s, STATES.D_u),
        (STATES.J_s, STATES.J_ds),
        (STATES.J_s, STATES.J_s),

        (STATES.J_n, STATES.R_u),
        (STATES.J_n, STATES.J_dn),
        (STATES.J_n, STATES.J_n),

        (STATES.J_ds, STATES.R_d),
        (STATES.J_ds, STATES.D_d),
        (STATES.J_ds, STATES.J_ds),

        (STATES.J_dn, STATES.R_d),
        (STATES.J_dn, STATES.J_dn),

        (STATES.R_d, STATES.R_d),
        (STATES.R_u, STATES.R_u),
        (STATES.D_d, STATES.D_d),
        (STATES.D_u, STATES.D_u)

    ],

    "final_states": [
        STATES.R_d,
        STATES.R_u,
        STATES.D_d,
        STATES.D_u
    ],

    "invisible_states": [
        STATES.D_u,
        STATES.D_d
    ],

    "unstable_states": [
        STATES.E,
        STATES.I_n,
        STATES.I_a,
        STATES.I_s,
        STATES.I_ds,
        STATES.I_da,
        STATES.I_dn,
        STATES.E_d
    ],

    "init_arguments": {
        "p": (0, "probability of interaction outside adjacent nodes"),
        "q": (0, " probability of detected individuals interaction outside adjacent nodes"),
        "false_symptoms_rate": (0, ""),
        "false_symptoms_recovery_rate": (1., ""),
        "asymptomatic_rate": (0, ""),
        "symptoms_manifest_rate": (1., ""),
        "save_nodes": (False, "")
    },

    "model_parameters": {
        "beta": (0,  "rate of transmission (exposure)"),
        "beta_reduction": (0,  "todo"),
        "beta_in_family": (0, "todo"),
        "beta_A": (0, "todo"),
        "beta_A_in_family": (0, "todo"),
        "sigma": (0, "rate of infection (upon exposure)"),
        "gamma_In": (0, "rate of recovery (upon infection)"),
        "gamma_Is": (0, "rate of recovery (upon infection)"),
#        "gamma_Id": (0, "rate of recovery (upon infection)"),
        "mu": (0, "rate of infection-related death"),
        "beta_D": (0, "rate of transmission (exposure) for detected inds"),
        "mu_D": (0, "rate of infection-related death for detected inds"),
        "theta_E": (0, "rate of baseline testing for exposed individuals"),
        "theta_Ia": (0, "rate of baseline testing for Ia individuals"),
        "theta_Is": (0, "rate of baseline testing for Is individuals"),
        "theta_In": (0, "rate of baseline testing for In individuals"),
        "test_rate": (1.0, ""),
        "phi_E": (0, "rate of contact tracing testing for exposed individuals"),
        "phi_Ia": (0, "rate of contact tracing testing for Ia individuals"),
        "phi_Is": (0, "rate of contact tracing testing for Is individuals"),
        "psi_E": (0, "probability of positive test results for exposed individuals"),
        "psi_Ia": (0, "probability of positive test results for Ia individuals"),
        "psi_Is": (0, "probability of positive test results for Is individuals"),
        "psi_In": (0, "probability of positive test results for In individuals"),
        "delta_n": (0, "probability of ..."),
        "delta_s": (0, "probability of ...")
    }
}

# 2. propensities function


def calc_propensities(model, use_dict=True):

    # STEP 1
    # pre-calculate matrix multiplication terms that may be used in multiple propensity calculations,
    # and check to see if their computation is necessary before doing the multiplication

    # number of infectious nondetected contacts
    # sum of all I states
    # numContacts_I = np.zeros(shape=(model.num_nodes, 1))
    # if any(model.beta):
    #     infected = [
    #         s for s in (STATES.I_n, STATES.I_a, STATES.I_s)
    #         if model.current_state_count(s)
    #     ]
    #     if infected:
    #         numContacts_I = model.num_contacts(infected)

    # numContacts_Id = np.zeros(shape=(model.num_nodes, 1))
    # if any(model.beta_D):
    #     numContacts_Id = model.num_contacts(STATES.I_d)

    # STEP 2
    # create  propensities
    # transition name: probability values
    # see doc/Propensities.pdf

    # compute P infection
    # first part omit
        # model.p * (
        #     model.beta * numI +
        #     model.q * model.beta_D * model.current_state_count(STATES.I_d)
        # ) / model.current_N()
        # + (1 - model.p)

    P1 = model.prob_of_contact(
        [STATES.S_s, STATES.S],
        [STATES.S,
         STATES.S_s,
         STATES.E,
         STATES.I_n,
         STATES.I_a,
         STATES.I_s
         ],
        [STATES.I_n, STATES.I_a, STATES.I_s, STATES.I_ds, STATES.I_da, STATES.I_dn],
        [STATES.I_n, STATES.I_a, STATES.I_s, STATES.I_ds,
            STATES.I_da, STATES.I_dn, STATES.E, STATES.E_d],
        model.beta, model.beta_in_family
    )

    #    P2 = model.prob_of_no_contact([STATES.I_d], model.beta_D)
    assert(np.all(model.beta_D == 0))

    # print("-->", P1.shape, np.any(P1.flatten() > 0), np.all(P1.flatten() <= 1))
    # print(P2.flatten())
    # assert np.all(P2.flatten() == 0)

    #    P_infection = model.prob_of_no_contact(
    #        ([STATES.I_n, STATES.I_a, STATES.I_s], [STATES.I_d])
    #        (model.beta, model.beta_D)
    #    )
    #    print("-->", P_infection.shape, np.any(P_infection.flatten() > 0), np.all(P_infection.flatten() <= 1))

    N = model.current_N()
    numIn = (
        model.current_state_count(STATES.I_a) + 
        model.current_state_count(STATES.I_n) + 
        model.current_state_count(STATES.I_dn) + 
        model.current_state_count(STATES.I_da)
    )
    numI = model.current_state_count(STATES.I_s) + model.current_state_count(STATES.I_ds)

    P2 = (model.beta * numI/N + model.beta_A * numIn/N)

    P_infection = (1-model.p)*P1 + model.p*P2
    #    P_infection = P1 +

    if model.t == 6:
        is_superspread_edge = model.graph.e_types == 31 
        nodes = np.concatenate(
            (model.graph.e_source[is_superspread_edge], 
             model.graph.e_dest[is_superspread_edge])
        )
        nodes = np.unique(nodes)
        
        ss_infections = P_infection[nodes]
        print(f"DBG SUPERSPREAD INF {ss_infections.mean()} {np.median(ss_infections)}")
        nonzero_nodes = len(ss_infections.nonzero()[0])
        print(f"DBG SUPERSPREAD NONZERO {nonzero_nodes} {ss_infections[ss_infections.nonzero()[0]].mean()}")

    S_or_S_s = model.memberships[STATES.S] + model.memberships[STATES.S_s]
    model.meaneprobs[model.t] = P_infection[S_or_S_s == 1].mean()
    model.medianeprobs[model.t] = np.median(P_infection[S_or_S_s == 1])
    #    print(P_infection.shape)

    not_P_infection = 1 - P_infection
    # assert np.all(P_infection < 1.0)
    # assert np.all((not_P_infection + P_infection) == 1)

    # print(model.memberships[STATES.S_s].shape)
    # print(P_infection.shape)
    # print(not_P_infection.shape)

    # print((model.memberships[STATES.S_s] * not_P_infection).shape)
    # exit()

    #    print(model.memberships[:, 0])
    # state S_s
    propensity_S_s_to_E = model.memberships[STATES.S_s] * P_infection
    propensity_S_s_to_S = (
        model.memberships[STATES.S_s] * not_P_infection * model.false_symptoms_recovery_rate)
    propensity_S_s_to_S_s = np.clip(
        model.memberships[STATES.S_s] * (1.0 - propensity_S_s_to_S - propensity_S_s_to_E),
        0.0, 1.0
    )
    # clip them all before we solve overlap 

    # print(propensity_S_s_to_E.flatten())
    # print(propensity_S_s_to_S.flatten())
    # print(propensity_S_s_to_S_s.flatten())
    # assert np.all((propensity_S_s_to_E + propensity_S_s_to_S +
    #                propensity_S_s_to_S_s)[np.nonzero(model.memberships[STATES.S_s])] == 1)

    # state S
    propensity_S_to_E = model.memberships[STATES.S] * P_infection
    propensity_S_to_S_s = (
        model.memberships[STATES.S] * not_P_infection * model.false_symptoms_rate)
    propensity_S_to_S = np.clip(
        model.memberships[STATES.S] * (1.0 - (propensity_S_to_E + propensity_S_to_S_s)),
        0.0, 1.0
    )
    

    #    print("S->E", propensity_S_to_E[0])
    #    print("S->Ss", propensity_S_to_S_s[0])
    #    print("E+Ss", (propensity_S_to_E + propensity_S_to_S_s)[0])
    #    print("E+Ss", (1.0 - (propensity_S_to_E + propensity_S_to_S_s))[0])
    #    print(model.memberships[STATES.S][0])
    #    print("S->S", propensity_S_to_S[0])

    # state E
    propensity_E_to_E_d = model.memberships[STATES.E] * model.theta_E * model.psi_E
    propensity_E_to_I_n = model.memberships[STATES.E] * model.sigma * model.asymptomatic_rate
    propensity_E_to_I_a = model.memberships[STATES.E] * model.sigma * (1.0 - model.asymptomatic_rate)
    propensity_E_to_E = np.clip(
        model.memberships[STATES.E] * (1.0 - propensity_E_to_E_d -
                                       propensity_E_to_I_n - propensity_E_to_I_a),
        0.0, 1.0
    )
    # state I_n
    propensity_I_n_to_J_n = model.memberships[STATES.I_n] * model.delta_n
    propensity_I_n_to_I_dn = model.memberships[STATES.I_n]  * model.theta_In * model.psi_In
    propensity_I_n_to_I_n = np.clip(
        model.memberships[STATES.I_n] * (
            1.0 - propensity_I_n_to_J_n - propensity_I_n_to_I_dn), 
        0.0, 1.0
    )

    # state I_a
    propensity_I_a_to_I_da = model.memberships[STATES.I_a] * model.theta_Ia * model.psi_Ia
    propensity_I_a_to_I_s =  model.memberships[STATES.I_a] * model.symptoms_manifest_rate
    propensity_I_a_to_I_a = np.clip(
        model.memberships[STATES.I_a] * (
            1.0 - propensity_I_a_to_I_da - propensity_I_a_to_I_s),
        0.0, 1.0
    )

    # state I_s
    propensity_I_s_to_I_ds = model.memberships[STATES.I_s]  * model.testable * model.theta_Is * model.psi_Is
    propensity_I_s_to_J_s = np.clip(
        model.memberships[STATES.I_s] * model.delta_s,
        0.0, 1 - propensity_I_s_to_I_ds)
    propensity_I_s_to_D_u = 0.0 * model.memberships[STATES.I_s] 
    #    not_R_or_D = 1.0 - propensity_I_s_to_J_s - propensity_I_s_to_D_u
    propensity_I_s_to_I_s = np.clip(
        model.memberships[STATES.I_s] * (1.0 - propensity_I_s_to_J_s -
                                         propensity_I_s_to_D_u - propensity_I_s_to_I_ds),
        0.0, 1.0
    )

    #print("DBG test propensity", propensity_I_s_to_I_ds[propensity_I_s_to_I_ds > 0])

    # state I_dn
    propensity_I_dn_to_J_dn = model.memberships[STATES.I_dn] * model.delta_n
    propensity_I_dn_to_D_d = 0.0 * model.memberships[STATES.I_dn] 
    propensity_I_dn_to_I_dn = np.clip(
        model.memberships[STATES.I_dn] * (
            1.0 - propensity_I_dn_to_J_dn - propensity_I_dn_to_D_d),
        0.0, 1.0
    )

    # state I_ds
    propensity_I_ds_to_J_ds = model.memberships[STATES.I_ds] * model.delta_s
    propensity_I_ds_to_D_d = 0.0 * model.memberships[STATES.I_ds] 
    propensity_I_ds_to_I_ds = np.clip(
        model.memberships[STATES.I_ds] * (
            1.0 - propensity_I_ds_to_J_ds - propensity_I_ds_to_D_d),
        0.0, 1.0
    )

    # state I_da
    propensity_I_da_to_I_ds = model.memberships[STATES.I_da] * \
        model.symptoms_manifest_rate
    propensity_I_da_to_I_da = model.memberships[STATES.I_da] * \
        (1.0 - propensity_I_da_to_I_ds)

    # state E_d
    propensity_E_d_to_I_dn = model.memberships[STATES.E_d] * \
        model.sigma * model.asymptomatic_rate
    propensity_E_d_to_I_da = model.memberships[STATES.E_d] * \
        model.sigma * (1.0 - model.asymptomatic_rate)
    propensity_E_d_to_E_d = np.clip(
        model.memberships[STATES.E_d] * (1.0
                                         - propensity_E_d_to_I_dn
                                         - propensity_E_d_to_I_da),
        0.0, 1.0
    )
    
    # state J_s
    propensity_J_s_to_J_ds = (
        model.memberships[STATES.J_s]  * model.testable * model.theta_Is * model.psi_Is)

    propensity_J_s_to_R_u = np.clip(
        model.memberships[STATES.J_s] * model.gamma_Is,
        0.0, 1 - propensity_J_s_to_J_ds)
    propensity_J_s_to_D_u = np.clip(
        model.memberships[STATES.J_s] * model.mu,
        0.0,
        1 - propensity_J_s_to_J_ds - propensity_J_s_to_R_u)
    #    not_R_or_D = 1.0 - propensity_J_s_to_R_u - propensity_J_s_to_D_u
    propensity_J_s_to_J_s = np.clip(
        model.memberships[STATES.J_s] * (1.0 - propensity_J_s_to_R_u -
                                         propensity_J_s_to_D_u - propensity_J_s_to_J_ds),
        0.0, 1.0
    )

    # state J_n
    propensity_J_n_to_R_u = model.memberships[STATES.J_n] * model.gamma_In
    propensity_J_n_to_J_dn = model.memberships[STATES.J_n] * model.theta_In * model.psi_In
    propensity_J_n_to_J_n = np.clip(
        model.memberships[STATES.J_n] * \
        (1.0 - propensity_J_n_to_R_u - propensity_J_n_to_J_dn),
        0.0, 1.0
    )

    # state J_ds
    propensity_J_ds_to_R_d = model.memberships[STATES.J_ds] * model.gamma_Is
    propensity_J_ds_to_D_d = model.memberships[STATES.J_ds] * model.mu
    propensity_J_ds_to_J_ds = np.clip(
        model.memberships[STATES.J_ds] * (
            1.0 - propensity_J_ds_to_R_d - propensity_J_ds_to_D_d),
        0.0, 1.0
    )

    # state J_dn
    propensity_J_dn_to_R_d = model.memberships[STATES.J_dn] * model.gamma_In
    propensity_J_dn_to_J_dn = model.memberships[STATES.J_dn] * (
        1.0 - propensity_J_dn_to_R_d)

    # state R_d, R_u, D_d, D_u
    propensity_R_d_to_R_d = model.memberships[STATES.R_d]
    propensity_R_u_to_R_u = model.memberships[STATES.R_u]
    propensity_D_d_to_D_d = model.memberships[STATES.D_d]
    propensity_D_u_to_D_u = model.memberships[STATES.D_u]

    return [
        propensity_S_s_to_E,
        propensity_S_s_to_S,
        propensity_S_s_to_S_s,

        propensity_S_to_E,
        propensity_S_to_S_s,
        propensity_S_to_S,

        propensity_E_to_E_d,
        propensity_E_to_I_n,
        propensity_E_to_I_a,
        propensity_E_to_E,

        propensity_I_n_to_J_n,
        propensity_I_n_to_I_dn,
        propensity_I_n_to_I_n,

        propensity_I_a_to_I_da,
        propensity_I_a_to_I_s,
        propensity_I_a_to_I_a,

        propensity_I_s_to_J_s,
        propensity_I_s_to_D_u,
        propensity_I_s_to_I_ds,
        propensity_I_s_to_I_s,

        propensity_I_ds_to_J_ds,
        propensity_I_ds_to_D_d,
        propensity_I_ds_to_I_ds,

        propensity_I_dn_to_J_dn,
        propensity_I_dn_to_D_d,
        propensity_I_dn_to_I_dn,

        propensity_I_da_to_I_ds,
        propensity_I_da_to_I_da,

        propensity_E_d_to_I_dn,
        propensity_E_d_to_I_da,
        propensity_E_d_to_E_d,


        propensity_J_s_to_R_u,
        propensity_J_s_to_D_u,
        propensity_J_s_to_J_ds,
        propensity_J_s_to_J_s,

        propensity_J_n_to_R_u,
        propensity_J_n_to_J_dn,
        propensity_J_n_to_J_n,

        propensity_J_ds_to_R_d,
        propensity_J_ds_to_D_d,
        propensity_J_ds_to_J_ds,

        propensity_J_dn_to_R_d,
        propensity_J_dn_to_J_dn,

        propensity_R_d_to_R_d,
        propensity_R_u_to_R_u,
        propensity_D_d_to_D_d,
        propensity_D_u_to_D_u
    ]


# 3. model class
ExtendedNetworkModel = create_custom_model("ExtendedNetworkModel",
                                           **model_definition,
                                           calc_propensities=calc_propensities)

ExtendedDailyNetworkModel = create_custom_model("ExtendedDailyNetworkModel",
                                                **model_definition,
                                                calc_propensities=calc_propensities,
                                                engine=DailyEngine)

ExtendedSequentialNetworkModel = create_custom_model("ExtendedSequentialNetworkModel",
                                                     **model_definition,
                                                     calc_propensities=calc_propensities,
                                                     engine=SequentialEngine)

# TODO: inherit from ExtendedNetworkModel a new model (high level) that includes the workaround
#      about multi-graphs, manages call backs, etc.

TGMNetworkModel = create_custom_model("TGMNetworkModel",
                                      **model_definition,
                                      calc_propensities=calc_propensities,
                                      engine=EngineM)
