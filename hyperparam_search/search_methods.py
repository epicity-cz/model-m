import warnings

import cma
import numpy as np
import scipy.special
from functools import partial
from multiprocessing import Pool

from cma.optimization_tools import EvalParallel2
from sklearn.model_selection import ParameterGrid
import gc

def _run_model_with_hyperparams(model_func, hyperparams, output_file=None):
    print(f"Running with hyperparams: {hyperparams}", flush=True)

    res = model_func(hyperparams=hyperparams)

    fitness = np.mean(res["result"]) 
    _log_inidividual(output_file, hyperparams.values(), fitness, 0)
    print(f"Finished run with hyperparams: {hyperparams}")
    return res


def perform_gridsearch(model_func, hyperparam_config, n_jobs=1, output_file=None):
    grid = hyperparam_config["MODEL"]
    param_grid = ParameterGrid(grid)

    header = grid.keys()
    _init_output_file(output_file, header)

    run_model = partial(_run_model_with_hyperparams, model_func, output_file=output_file)
    with Pool(processes=n_jobs) as pool:
        res = pool.map(run_model, param_grid)
    return res


def evaluate_with_params(param_array: np.ndarray, model_func, param_keys, param_ranges=None):
    assert len(param_array) == len(param_keys)

    hyperparam_dict = _compile_individual(param_array, param_keys=param_keys, param_ranges=param_ranges)
    model_res = model_func(hyperparams=hyperparam_dict)["result"]

    return np.mean(model_res)


def _keys_with_evolved_vals(evolved_vals, keys):
    return {k: v for k, v in zip(keys, evolved_vals)}


def _init_output_file(output_file, header):
    if output_file is not None:
        with open(output_file, 'w+') as of:
            key_string = ','.join(header)
            of.write(f"gen,{key_string},fitness\n")


def _log_inidividual(output_file, x, fitness, gen):
    if output_file is not None:
        with open(output_file, 'a') as of:
            of.write(f'{gen},{",".join(str(val) for val in x)},{fitness}\n')  # joined hyperparam values and fitness


def _inverse_sigmoid(x):
    x = np.clip(x, 0+np.finfo(np.float32).eps, 1-np.finfo(np.float32).eps)
    return np.log(x/(1-x))


def _init_values(value_dict, ranges=None):
    if ranges is None:
        initial_vals = np.array([v for v in value_dict.values()])
        return _inverse_sigmoid(initial_vals)

    def scale(x, lower, upper):
        return (x - lower) / (upper - lower)

    res_vals = []
    for k, v in value_dict.items():
        if k not in ranges:
            res_vals.append(v)
        else:
            low, up = ranges[k]
            res_vals.append(scale(v, low, up))

    return _inverse_sigmoid(np.array(res_vals))


def _compile_individual(x, param_keys=None, param_ranges=None, with_keys=True):
    def scale_back(val, key):
        val = scipy.special.expit(val)
        low, up = param_ranges[key]
        return val * (up - low) + low

    # optional rescaling
    if param_ranges is None:
        ind = scipy.special.expit(x).tolist()
    else:
        ind = []
        for key, val in zip(param_keys, x):
            ind.append(val if key not in param_ranges else scale_back(val, key))

    # return with keys or not
    return _keys_with_evolved_vals(ind, param_keys) if with_keys else ind



def cma_es(model_func, hyperparam_config: dict, return_only_best=False, output_file=None, n_jobs=1):
    initial_kwargs = hyperparam_config["MODEL"]
    _init_output_file(output_file, initial_kwargs.keys())

    param_ranges = hyperparam_config.get("param_ranges", None)
    initial_vals = _init_values(initial_kwargs, ranges=param_ranges)

    sigma = hyperparam_config["SIGMA"]
    cma_kwargs = hyperparam_config["CMA"]

    eval_func = partial(evaluate_with_params, model_func=model_func,
                        param_keys=list(initial_kwargs.keys()), param_ranges=param_ranges)

    es = cma.CMAEvolutionStrategy(initial_vals, sigma, cma_kwargs)
    with EvalParallel2(fitness_function=eval_func, number_of_processes=n_jobs) as eval_all:
        gen_n = 0
        while not es.stop():
            X = es.ask()
            fitnesses = eval_all(X)
            es.tell(X, fitnesses)
            es.disp()

            for x, f in zip(X, fitnesses):
                ind = _compile_individual(x, param_keys=initial_kwargs.keys(),
                                          param_ranges=param_ranges, with_keys=False)
                _log_inidividual(output_file, list(ind), f, gen_n)

            gen_n += 1
            gc.collect()

    res = es.result
    x = _compile_individual(res[0], initial_kwargs.keys(), param_ranges=param_ranges)

    if return_only_best:
        return {"hyperparams": x, "result": res[1]}  # best evaluated solution, its objective function value
    return {"hyperparams": x, "result": res[1], "es_data": res[2:]}  # full result


hyperparam_search_zoo = {
    'GridSearch': perform_gridsearch,
    'CMA-ES': cma_es
}
