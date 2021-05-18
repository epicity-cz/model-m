from engine_seirspluslike import SeirsPlusLikeEngine


def not_implemented_yet(): raise NotImplementedError()


def create_custom_model(clsname, states, state_str_dict, transitions,
                        final_states=[], invisible_states=[],
                        unstable_states=[],
                        init_arguments={},
                        model_parameters={},
                        calc_propensities=not_implemented_yet,
                        engine=SeirsPlusLikeEngine):
    """ Creates base model class

    Params:
         states              list of states
         transitions         list of state couples (possible transitions)
         final_states        list of final states (optional)
         invisible_states    states that are not members of population

    Returns:
        class
    """

    # dictionary of future class variables
    attributes = {
        "states": states,
        "num_states": len(states),
        "state_str_dict": state_str_dict,
        "transitions": transitions,
        "num_transitions": len(transitions),
        "final_states": final_states,
        "invisible_states": invisible_states,
        "unstable_states": unstable_states or states,
        "fixed_model_parameters": init_arguments,
        "model_parameters": model_parameters,
        "common_arguments": {"random_seed": (None, "random seed value")}
    }

    model_cls = type(clsname, (engine,), attributes)
    doc_text = """    A class to simulate the Stochastic Network Model

    Params:
            G       Network adjacency matrix (numpy array) or Networkx graph object \n"""

    for argname in ("fixed_model_parameters",
                    "model_parameters",
                    "common_arguments"):
        for param, definition in attributes[argname].items():
            param_text = f"            {param}       {definition[1]}\n"
            if argname == "model_parameters":
                param_text += f"            (float or np.array)\n"
            doc_text = doc_text + param_text

    model_cls.__doc__ = doc_text

    # __init__ method
    def init_function(self, G,  **kwargs):

        # 1. set member variables acording to init arguments
        # definition is couple (default value, description)
        self.G = G
        self.graph = G
        self.A = None
        self.init_kwargs = kwargs

        # 2. model initialization
        self.inicialization()

        # 3. time and history setup
        self.setup_series_and_time_keeping()

        # 4. init states and their counts
        # print(self.init_state_counts)
        self.states_and_counts_init()

        # 5. set callback to None
        self.periodic_update_callback = None

    # add __init__
    model_cls.__init__ = init_function

    # # add member functions
    # function_list = [inicialization,
    #                  update_graph,
    #                  node_degrees,
    #                  setup_series_and_time_keeping,
    #                  states_and_counts_init,
    #                  set_periodic_update,
    #                  update_scenario_flags,
    #                  num_contacts,
    #                  current_state_count,
    #                  current_N,
    #                  run_iteration,
    #                  run,
    #                  finalize_data_series,
    #                  increase_data_series_length]

    # for function in function_list:
    #     setattr(model_cls, function.__name__, function)

    # def not_implemented_yet(self):
    #     raise NotImplementedError

    if calc_propensities is None:
        calc_propensities = not_implemented_yet
    else:
        model_cls.calc_propensities = calc_propensities

    return model_cls
