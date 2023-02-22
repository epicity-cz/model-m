import timeit
import time
import click
import random
import matplotlib.pyplot as plt
import numpy as np

from config_utils import ConfigFile
from graph_gen import GraphGenerator

from model_zoo import model_zoo


from model_m import ModelM, load_model_from_config


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


def demo(filename, test_id=None, model_random_seed=42, use_policy=None, print_interval=1, n_repeat=1):

    cf = ConfigFile()
    cf.load(filename)

    # create model
    model = load_model_from_config(cf, use_policy, model_random_seed)

    # run parameters
    ndays = cf.section_as_dict("TASK").get("duration_in_days", 60)
    print_interval = cf.section_as_dict("TASK").get("print_interval", 1)
    verbose = cf.section_as_dict("TASK").get("verbose", "Yes") == "Yes"

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
        cf.save(file_name)
        cfg_string = ""
        with open(file_name, "r") as f:
            cfg_string = "#" + "#".join(f.readlines())
        with open(file_name, "w") as f:
            f.write(cfg_string)
            f.write(f"# RANDOM_SEED = {model_random_seed}\n")
            model.save_history(f)

        with open(f"durations{suffix}.csv", "w") as f:
            model.model.save_durations(f)

        save_nodes = cf.section_as_dict("TASK").get(
            "save_node_states", "No") == "Yes"
        if save_nodes:
            model.save_node_states(f"node_states{suffix}.csv")


@click.command()
@click.option('--const-random-seed/--no-random-seed', ' /-r', default=True)
@click.option('--user_random_seed', '-R', default=None)
@click.option('--policy', '-p', default=None)
@click.option('--print_interval',  default=1)
@click.option('--n_repeat',  default=1)
@click.argument('filename', default="example.ini")
@click.argument('test_id', default="")
def test(const_random_seed, user_random_seed, policy, print_interval, n_repeat, filename, test_id):
    """ Run the demo test inside the timeit """

    if user_random_seed is not None:
        random_seed = int(user_random_seed)
    else:
        random_seed = 6321 if const_random_seed else random.randint(0, 429496729)

    print(f"ACTION LOG: random seed {random_seed}")
    def demo_fce(): return demo(filename, test_id,
                                model_random_seed=random_seed, use_policy=policy, print_interval=print_interval, n_repeat=n_repeat)
    print(timeit.timeit(demo_fce, number=1))


if __name__ == "__main__":
    test()
