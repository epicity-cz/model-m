import timeit
import time
import click
import random
import matplotlib.pyplot as plt
import numpy as np
import copy 

from config_utils import ConfigFile
from graph_gen import GraphGenerator

from model_zoo import model_zoo


from model_m import ModelM, load_model_from_config, load_graph

from pool import Pool


def evaluate_model(model, setup):

    my_model = model

    idx, random_seed, test_id, config, args = setup 
    ndays, print_interval, verbose = args

    suffix = "" if not test_id else "_" + test_id


    if random_seed is not None:
        my_model.reset(random_seed=random_seed)

    try:
        my_model.run(ndays, print_interval=print_interval, verbose=verbose)
    except AssertionError:
        file_name = f"history{suffix}.FAILED"
        with open(file_name, "w") as f:
            import sys
            import traceback
            _, _, tb = sys.exc_info()
            traceback.print_tb(tb) # Fixed format
            tb_info = traceback.extract_tb(tb)
            filename, line, func, text = tb_info[-1]
            f.write(f'An error occurred on line {line} in statement {text}')
        return idx 

    # save history
    file_name = f"history{suffix}.csv"
    config.save(file_name)
    cfg_string = ""
    with open(file_name, "r") as f:
        cfg_string = "#" + "#".join(f.readlines())
    with open(file_name, "w") as f:
        f.write(cfg_string)
        f.write(f"# RANDOM_SEED = {my_model.model.random_seed}\n")
        my_model.save_history(f)

    # with open(f"durations{suffix}.csv", "w") as f:
    #     my_model.model.save_durations(f)

    return idx
#    del my_model 



def demo(filename, test_id=None, model_random_seed=42, use_policy=None, print_interval=1, n_repeat=1, n_jobs=1):

    cf = ConfigFile()
    cf.load(filename)
    graph = load_graph(cf) 

    ndays = cf.section_as_dict("TASK").get("duration_in_days", 60)
    print_interval = cf.section_as_dict("TASK").get("print_interval", 1)
    verbose = cf.section_as_dict("TASK").get("verbose", "Yes") == "Yes"

    if not isinstance(model_random_seed, list):
        model_random_seed = [ model_random_seed + i
                              for i in range(n_repeat)
        ]
    
    # create model
    model = load_model_from_config(cf, use_policy, model_random_seed[0], preloaded_graph=graph)
    print("model loaded", flush=True) 

    # run parameters

    if test_id is None:
        test_id = ""

    models = [ model ]  
    for i in range(1, n_jobs):
        print(f"{i} copy", flush=True) 
        models.append(model.duplicate(random_seed=model_random_seed[i])) 



    pool = Pool(processors=n_jobs, evalfunc=evaluate_model, models=models)    
    for i in range(n_jobs):
        pool.putQuerry((i, None, f"{test_id}_{i}", cf, (ndays, print_interval, verbose)))
        
    i = n_jobs
    answers = 0
    while i < n_repeat:
        idx = pool.getAnswer()
        answers += 1 
        pool.putQuerry((idx, model_random_seed[i], 
                        f"{test_id}_{i}", cf, (ndays, print_interval, verbose)))
        i += 1

    for i in range(answers, n_repeat):
        pool.getAnswer()

    pool.close()
    print("finished")



@click.command()
@click.option('--const-random-seed/--no-random-seed', ' /-r', default=True, help="Use -r to run with different random seed each time." )
@click.option('--user-random-seeds', '-R', default=None, help="File with random seeds.") 
@click.option('--print_interval',  default=1, help="How often to print the info message.")
@click.option('--n_repeat',  default=1, help="Number repetition of the experiment.")
@click.option('--n_jobs', default=1, help="Number of processes.") 
@click.argument('filename', default="example.ini")
@click.argument('test_id', default="")
def test(const_random_seed,  user_random_seeds,  print_interval, n_repeat, n_jobs, filename, test_id):
    """ Run the experiment given by the config file FILENAME.                                                                                
    \b
    FILENAME  config file with the experiment definition                                                                                 
    TEST_ID   ID is appended to  names of all output files   
    """

    random_seed = 42 if const_random_seed else random.randint(0, 10000)
    if user_random_seeds is not None:
        with open(user_random_seeds, "r") as f:
            random_seeds = [] 
            for line in f:
                random_seeds.append(int(line.strip())) 
            random_seed = random_seeds 
    
    def demo_fce(): return demo(filename, test_id,
                                model_random_seed=random_seed, use_policy=None, print_interval=print_interval, 
                                n_repeat=n_repeat, n_jobs=n_jobs)
    print(timeit.timeit(demo_fce, number=1))


if __name__ == "__main__":
    test()
