import os
import timeit
import time
import click
import random
import matplotlib.pyplot as plt
import numpy as np
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))

import utils.global_configs as global_configs
from utils.config_utils import ConfigFile
from graphs.graph_gen import GraphGenerator

from models.model_zoo import model_zoo


from model_m.model_m import ModelM, load_model_from_config
import logging


# logging.basicConfig(format='%(levelname)s:%(module)s:%(lineno)d: %(message)s',
#                     level=logging.DEBUG)

# def tell_the_story(history, graph):

#     story = ["Once upon a time ..."]

#     states = {
#         "S": "be healthy",  "S_s":  "have flue symptoms",
#         "E": "be exposed", "I_n": "be infectious without symptoms",
#         "I_a":  "be symptomatic and infectious with no  manifest of symptoms",
#         "I_s": "manifest symptoms",
#         "I_d": "be as famous as a taxidriver",
#         "R_d": "be an expert on epidemy",
#         "R_u":  "be healthy again",
#         "D_d":  "push up daisies",
#         "D_u":  "pine for the fjords"
#     }

#     if isinstance(graph, GraphGenerator):
#         sexes = graph.get_attr_list("sex")
#         names = graph.get_attr_list("label")
#     else:
#         sexes = None
#         names = None

#     for t in range(1, len(history)):
#         node, src, dst = history[t]
#         node, src, dst = node.decode(), src.decode(), dst.decode()

#         if sexes:
#             who = "A lady" if sexes[int(node)] else "A gentleman"
#         else:
#             who = "A node"

#         if names:
#             name = names[int(node)]
#         else:
#             name = node

#         src_s, dst_s = states.get(src, src), states.get(dst, dst)
#         story.append(f"{who} {name} stopped to {src_s} and started to {dst_s}.")

#     story.append(
#         "Well! I never wanted to do this in the first place. I wanted to be... an epidemiologist!")

#     return "\n".join(story)


def demo(filename, test_id=None, model_random_seed=42,  print_interval=1, n_repeat=1):

    cf = ConfigFile()
    cf.load(filename)

    # create model
    model = load_model_from_config(cf, model_random_seed)

    # run parameters
    ndays = cf.section_as_dict("TASK").get("duration_in_days", 60)
    print_interval = cf.section_as_dict("TASK").get("print_interval", 1)
    verbose = cf.section_as_dict("TASK").get("verbose", "Yes") == "Yes"
    monitor_node = cf.section_as_dict("TASK").get("monitor_node", None)
    save_nodes = cf.section_as_dict("TASK").get(
        "save_node_states", "No") == "Yes"
    output_dir = cf.section_as_dict("TASK").get(
        "output_dir", None) 

    if monitor_node is not None:
        global_configs.MONITOR_NODE = monitor_node
    if save_nodes is not None:
        global_configs.SAVE_NODES = save_nodes

    if test_id is None:
        test_id = ""

    for i in range(n_repeat):

        test_id_i = f"{test_id}_{i}" if n_repeat > 1 else test_id
        if i > 0:
            model.reset()
        model.run(ndays, print_interval=print_interval, verbose=verbose)

        # storyfile = cf.section_as_dict("OUTPUT").get("story", None)
        # if storyfile:
        #     story = tell_the_story(model.history, model.G)
        #     with open(storyfile, "w") as f:
        #         f.write(story)

        # save history
        suffix = "" if not test_id_i else "_" + test_id_i
        file_name = f"history{suffix}.csv"
        if output_dir is not None:
            file_name = os.path.join(output_dir, file_name)
        cf.save(file_name)
        cfg_string = ""
        with open(file_name, "r") as f:
            cfg_string = "#" + "#".join(f.readlines())
        with open(file_name, "w") as f:
            f.write(cfg_string)
            f.write(f"# RANDOM_SEED = {model_random_seed}\n")
            model.save_history(f)

#        with open(f"durations{suffix}.csv", "w") as f:
#            model.model.save_durations(f)

        if save_nodes:
            save_nodes_filename = f"node_states{suffix}.csv"
            if output_dir is not None:
                save_nodes_filename = os.path.join(output_dir, save_nodes_filename)
            model.save_node_states(save_nodes_filename)

        save_source_infection = False
        if save_source_infection:
            source_filename = f"sources{suffix}.csv"
            if output_dir is not None:
                source_filename = os.path.join(output_dir, source_filename)
            with open(source_filename, "w") as f:
                model.model.df_source_infection().to_csv(f)

        save_source_nodes = False
        if save_source_nodes:
            source_nodes_filename = f"snodes{suffix}.csv"
            if output_dir is not None:
                source_nodes_filename = os.path.join(output_dir, source_nodes_filename)
            with open(source_nodes_filename, "w") as f:
                model.model.df_source_nodes().to_csv(f)


@click.command()
@click.option('--const-random-seed/--no-random-seed', ' /-r', default=True)
@click.option('--user_random_seed', '-R', default=None, help="User defined random seed number.")
@click.option('--print_interval',  default=1, help="How often print short info, defaults to daily.")
@click.option('--n_repeat',  default=1, help="Total number of simulations.")
@click.option('--log_level', default="CRITICAL", help="Logging level.")
@click.argument('filename', default="example.ini")
@click.argument('test_id', default="")
def test(const_random_seed, user_random_seed,  print_interval, n_repeat, log_level, filename, test_id):
    """ Run the simulation specified in FILENAME.

    \b
    FILENAME   name of the config file with the setup of the experiment
    TEST_ID    tag to append to output files names
    """

    if user_random_seed is not None:
        try:
            random_seed = int(user_random_seed)
        except ValueError:
            print(f"User defined random seed must be of type int. Provided: {user_random_seed}")
            print("Exiting.")
            exit()
    else:
        random_seed = 6321 if const_random_seed else random.randint(
            0, 429496729)

    logging.basicConfig(format='%(levelname)s:%(module)s:%(lineno)d: %(message)s',
                        level=log_level)

    print(f"ACTION LOG: random seed {random_seed}")
    def demo_fce(): return demo(filename, test_id,
                                model_random_seed=random_seed,  print_interval=print_interval, n_repeat=n_repeat)
    print(timeit.timeit(demo_fce, number=1))


if __name__ == "__main__":
    test()
