import json
from functools import partial
from sklearn.model_selection import ParameterGrid

from multiprocessing import Pool
from typing import Dict

from config_utils import ConfigFile
from hyperparam_search.eval_model import return_func_zoo
from hyperparam_search.search_methods import hyperparam_search_zoo
from model_m import load_model_from_config, ModelM, load_graph


def run_hyperparam_search(model_config: str,
                          hyperparam_config: str,
                          model_random_seed: int = 42,
                          use_policy: str = None,
                          run_n_times: int = 1,
                          n_days: int = None,
                          return_func: str = None,
                          return_func_kwargs: Dict = None,
                          **kwargs):
    """
    Run hyperparameter search on a model loaded from config. Hyperparameters specified in `hyperparam_config`
    overwrite those in `model_config`. Search method is defined in `hyperparam_config` as well.

    A single model run returns the model as a whole. If only a part of info is to be extracted, pass
    `return_func` to this function - a string that corresponds to a specific function in `utils.eval_model`.

    Params:
        model_config: Model config filename (ini)
        hyperparam_config: Hyperparam search filename (json)
        model_random_seed: Initial random seed for every model.
        use_policy: Name of the policy to use.

        run_n_times: For a single model (with specific hyperparams), repeat run multiple times. Random seed
            for i-th run is set to `model_random_seed` + i.

        return_func: String key of return function from `utils.eval_model`
        return_func_kwargs: Additional kwargs to pass to return function (which is specified by `return_func`).
        **kwargs: Additional keyword arguments to pass to the hyperparam search function,
            possibly specific to the given method).

    Returns: Result of hyperparameter optimization (specific to search method).
    """

    cf = ConfigFile()
    cf.load(model_config)

    graph = load_graph(cf)
    base_model = load_model_from_config(cf, use_policy, model_random_seed, preloaded_graph=graph)

    # wrapper for running one model same time with different seed
    model_load_func = partial(_run_models_from_config,
                              cf,
                              preloaded_graph=graph, 
                              preloaded_model=base_model, 
                              model_random_seed=model_random_seed,
                              run_n_times=run_n_times,
                              use_policy=use_policy,
                              n_days=n_days,
                              return_func=return_func,
                              return_func_kwargs=return_func_kwargs)

    hyperparam_search_func = _init_hyperparam_search(hyperparam_config)
    return hyperparam_search_func(model_func=model_load_func, **kwargs)


def run_single_model(model, T, print_interval=10, verbose=False):
    model.run(T=T, verbose=verbose, print_interval=print_interval)
    return model


def _run_models_from_config(cf: ConfigFile,
                            preloaded_graph: None, 
                            preloaded_model: None,
                            hyperparams: Dict = None,
                            model_random_seed: int = 42,
                            run_n_times: int = 1,
                            n_days: int = None,
                            use_policy: str = None,
                            return_func: str = None,
                            return_func_kwargs: Dict = None):
    # copy model
    ndays = n_days if n_days is not None else cf.section_as_dict("TASK").get("duration_in_days", 60)
    print_interval = cf.section_as_dict("TASK").get("print_interval", 1)
    verbose = cf.section_as_dict("TASK").get("verbose", "Yes") == "Yes"

    if "theta" in hyperparams:
        hyperparams.update({"theta_E": hyperparams["theta"],
                            "theta_Ia": hyperparams["theta"],
                            "theta_In": hyperparams["theta"]
                        })
        del hyperparams["theta"]

    if "a_reduction" in hyperparams:
        hyperparams["beta_A"] = hyperparams["beta"]*hyperparams["a_reduction"]
        del hyperparams["a_reduction"] 

    hyperparams["beta_in_family"] = hyperparams["beta"] 
    if "beta_A" in hyperparams:
        hyperparams["beta_A_in_family"] = hyperparams["beta_A"] 

    if preloaded_model is None:
        model = load_model_from_config(cf, use_policy, model_random_seed, preloaded_graph=preloaded_graph, hyperparams=hyperparams)
    else:
        model = preloaded_model.duplicate(model_random_seed, hyperparams)
    

    # for different seeds
    def _run_one_model(seed):
        model.reset(random_seed=seed)
        ret = run_single_model(model, T=ndays, print_interval=print_interval, verbose=verbose)
        return ret

    # specific return function or identity
    func = return_func_zoo[return_func] if return_func is not None else lambda m, **kwargs: m

    if run_n_times > 1:
        # add 1 to seed each run
        res = [func(_run_one_model(seed), **return_func_kwargs)
               for seed in range(model_random_seed, model_random_seed + run_n_times)]
    else:
        res = func(_run_one_model(model_random_seed), **return_func_kwargs)

    # optionally return additional run info
    return {
        "result": res,
        "hyperparams": hyperparams,
        "seed": model_random_seed
    }


def _init_hyperparam_search(hyperparam_file: str):
    with open(hyperparam_file, 'r') as json_file:
        config = json.load(json_file)

    return partial(hyperparam_search_zoo[config["method"]], hyperparam_config=config)
