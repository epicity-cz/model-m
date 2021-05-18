import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from matplotlib import animation
from typing import Dict, List


def plot_history(filename: str):
    history = _load_history(filename)
    history.plot(x="T", y="all_infectious")
    plt.show()

    

def plot_histories(*args, group_days: int = None, group_func: str = "max", **kwargs):
    histories = [_history_with_fname(
        filename, group_days=group_days, group_func=group_func) for filename in args]
    history_one_df = pd.concat(histories)
    _plot_lineplot(history_one_df, "day", "all_infectious", **kwargs)


def plot_mutliple_policies(policy_dict: Dict[str, List[str]],
                           group_days: int = None, group_func: str = "max", value="all_infectious", max_days=None, **kwargs):
    histories = []
    for policy_key, history_list in policy_dict.items():
        histories.extend([_history_with_fname(filename,
                                              group_days=group_days,
                                              group_func=group_func,
                                              policy_name=policy_key,
                                              max_days=max_days)
                          for filename in history_list])

    history_one_df = pd.concat(histories)
    _plot_lineplot(history_one_df, "day", value,
                   hue="policy_name", **kwargs)





def plot_mutliple_policies_everything(policy_dict: Dict[str, List[str]],
                                      group_days: int = None, group_func: str = "max",
                                      max_days=None, **kwargs):
    histories = []
    for policy_key, history_list in policy_dict.items():
        histories.extend([_history_with_fname(filename,
                                              group_days=group_days,
                                              group_func=group_func,
                                              policy_name=policy_key,
                                              max_days=max_days)
                          for filename in history_list])

    history_one_df = pd.concat(histories)

    if "variant" in kwargs and kwargs["variant"] == 2:
        plot_function = _plot_lineplot2 
        del kwargs["variant"] 
    else:
        plot_function = _plot_lineplot3

    plot_function(history_one_df, "day",
                  hue="policy_name", **kwargs)



def plot_state_histogram(filename: str, title: str = "Simulation", states: List[str] = None, save_path: str = None):
    def animate(i):
        fig.suptitle(f"{title} - day {day_labels.iloc[i]}")

        data_i = data.iloc[i]
        for d, b in zip(data_i, bars):
            b.set_height(math.ceil(d))

    fig, ax = plt.subplots()

    history = _history_with_fname(filename, group_days=1, keep_only_all=False)
    day_labels = history["day"]
    data = history.drop(["T", "day", "all_infectious", "filename"], axis=1)
    if states is not None:
        data = data[states]

    bars = plt.barplot(range(data.shape[1]),
                       data.values.max(), tick_label=data.columns)

    anim = animation.FuncAnimation(fig, animate, repeat=False, blit=False, frames=history.shape[0],
                                   interval=100)

    if save_path is not None:
        anim.save(save_path, writer=animation.FFMpegWriter(fps=10))
    plt.show()


def _plot_lineplot(history_df, x, y, hue=None, save_path=None,  **kwargs):
    if "title" in kwargs:
        title = kwargs["title"]
        del kwargs["title"]
    else:
        title = ""
    sns_plot = sns.lineplot(x=x, y=y, data=history_df,
                            hue=hue, estimator=np.median, ci='sd', **kwargs)
    # dirty hack (ro)
    if y == "mean_waiting":
        sns_plot.set(ylim=(0, 10))
    else:
        sns_plot.set(ylim=(0, 150))
    sns_plot.set_title(title)
    if save_path is not None:
        sns_plot.get_figure().savefig(save_path)

    plt.show()


def _plot_lineplot2(history_df, x,  hue=None, save_path=None,  plotall=True, **kwargs):

    title = kwargs["title"]
    del kwargs["title"]
    maxy = kwargs.get("maxy", None)
    if "maxy" in kwargs:
        del kwargs["maxy"]

    maxx = kwargs.get("maxx", None)
    if "maxx" in kwargs:
        del kwargs["maxx"]



    fig = plt.figure()
    axs = [None] * 2
    axs[0] = fig.add_subplot(121)
    axs[1] = fig.add_subplot(122)
#    axs[2] = fig.add_subplot(223)
#    axs[3] = fig.add_subplot(224)


    #dirty hack to get rid of stupid legend title
    kwargs["legend"] = False 
    sns_plot = sns.lineplot(x=x, y="I_d", data=history_df,
                            hue=hue, estimator=np.median, ci='sd', ax=axs[0], **kwargs)

    history_df_r = history_df[history_df["policy_name"] != "Czech Republic (scaled down)"]
#    maxy = 5000
    # dirty hack (ro)
    axs[0].set(ylim=(0, maxy))
    axs[0].set(xlim=(0, maxx))
    axs[0].set_ylabel("all detected states")
    axs[0].set_title("detected - active cases - median")
    axs[0].legend(history_df[hue].unique(), title=None, fancybox=True, )

    sns_plot2 = sns.lineplot(x=x, y="all_infectious", data=history_df_r,
                             hue=hue, estimator=np.median, ci='sd', ax=axs[1], **kwargs)
#    maxy = 25000
    # dirty hack (ro)
    axs[1].set(ylim=(0, maxy))
    axs[1].set(xlim=(0, maxx))
    axs[1].set_ylabel("all infected states")
    axs[1].set_title("all active cases - median")
    axs[1].legend(history_df_r[hue].unique(), title=None, fancybox=True, )

    """
    sns_plot = sns.lineplot(x=x, y="I_d", data=history_df,
                            hue=hue, estimator=np.mean, ci='sd', ax=axs[2], **kwargs)

    history_df_r = history_df[history_df["policy_name"] != "Czech Republic (scaled down)"]

    maxy = 5000
    # dirty hack (ro)
    axs[2].set(ylim=(0, maxy))
    axs[2].set(xlim=(0, maxx))
    axs[2].set_ylabel("all detected states")
    axs[2].set_title("detected - active cases - mean")
    axs[2].legend(history_df[hue].unique(), title=None, fancybox=True, )

    sns_plot2 = sns.lineplot(x=x, y="all_infectious", data=history_df_r,
                             hue=hue, estimator=np.mean, ci='sd', ax=axs[3], **kwargs)
    maxy = 25000
    # dirty hack (ro)
    axs[3].set(ylim=(0, maxy))
    axs[3].set(xlim=(0, maxx))
    axs[3].set_ylabel("all infected states")
    axs[3].set_title("all active cases - mean")
    axs[3].legend(history_df_r[hue].unique(), title=None, fancybox=True, )
    """

    fig.suptitle(title, fontsize=20)

    if save_path is not None:
        plt.savefig(save_path)


def _plot_lineplot3(history_df, x,  hue=None, save_path=None,  plotall=True, **kwargs):

    title = kwargs["title"]
    del kwargs["title"]
    maxy = kwargs.get("maxy", 300)
    if "maxy" in kwargs:
        del kwargs["maxy"]


    fig = plt.figure()
    axs = [None] * 6
    axs[0] = fig.add_subplot(131)
    axs[1] = fig.add_subplot(132)
    if plotall:
        axs[2] = fig.add_subplot(433)
        axs[4] = fig.add_subplot(436)
        axs[3] = fig.add_subplot(439)
        axs[5] = fig.add_subplot(4,3,12)
    else:        
        axs[3] = fig.add_subplot(133)

    # axs[2] = fig.add_subplot(433)
    # axs[3] = fig.add_subplot(436)
    # axs[4] = fig.add_subplot(439)
    # axs[5] = fig.add_subplot(4,3,12)

    #dirty hack to get rid of stupid legend title
    kwargs["legend"] = False 
    sns_plot = sns.lineplot(x=x, y="I_d", data=history_df,
                            hue=hue, estimator=np.median, ci='sd', ax=axs[0], **kwargs)

    history_df_r = history_df[history_df["policy_name"] != "Czech Republic (scaled down)"]

    # dirty hack (ro)
    axs[0].set(ylim=(0, 40))
#    axs[0].set(xlim=(1, 150))
    axs[0].set_ylabel("all detected states")
    axs[0].set_title("detected - active cases")
    axs[0].legend(history_df[hue].unique(), title=None, fancybox=True, )
    axs[0].axvline(x=5, color="gray")
    axs[0].axvline(x=36, color="gray")
    axs[0].axvline(x=66, color="gray")
    axs[0].axvline(x=97, color="gray")



    sns_plot2 = sns.lineplot(x=x, y="all_infectious", data=history_df_r,
                             hue=hue, estimator=np.median, ci='sd', ax=axs[1], **kwargs)
    # dirty hack (ro)
    axs[1].set(ylim=(0, 150))
 #   axs[1].set(xlim=(1, 150))
    axs[1].set_ylabel("all infected states")
    axs[1].set_title("all active cases")
    axs[1].legend(history_df_r[hue].unique(), title=None, fancybox=True, )

    
    if plotall:
        sns_plot3 = sns.lineplot(x=x, y="mean_waiting", data=history_df,
                                 hue=hue, estimator=np.median, ci='sd', ax=axs[2], **kwargs)
        axs[2].set(ylim=(0, 15))
        axs[2].set(xlim=(1, 120))
        axs[2].legend(history_df[hue].unique(), title=None, fancybox=True, )


#        axs[2].set_title("waiting times")
        
    sns.lineplot(x=x, y="detected_ratio", data=history_df_r,
                 hue=hue, estimator=np.median, ci=None, ax=axs[3], **kwargs)
    axs[3].set(ylim=(0,15))
    axs[3].set(xlim=(1,120))
    axs[3].legend(history_df_r[hue].unique(), title=None, fancybox=True, )

    #   axs[3].set_title("detected_ratio")
    axs[3].axvline(x=5)
    axs[3].axvline(x=36)
    axs[3].axvline(x=66)
    axs[3].axvline(x=97)
    axs[3].axhline(y=10)


    sns.lineplot(x=x, y="all_tests", data=history_df,
                 hue=hue, estimator=np.median, ci='sd', ax=axs[4],
                 **kwargs)
    axs[4].set(ylim=(0,52))
    axs[4].set(xlim=(1, 120))
    axs[4].legend(history_df[hue].unique(), title=None, fancybox=True, )

    
    sns.lineplot(x=x, y="mean_p_infection", data=history_df_r,
                hue=hue, estimator=np.median, ci='sd', ax=axs[5],
                **kwargs)
    axs[5].set(xlim=(1, 120))
    axs[5].legend(history_df_r[hue].unique(), title=None, fancybox=True, )

    
    # sns.lineplot(x=x, y="nodes_in_quarantine", data=history_df,
    #              hue=hue, estimator=np.mean, ci='sd', ax=axs[3], **kwargs)
    # sns.lineplot(x=x, y="contacts_collected", data=history_df,
    #              hue=hue, estimator=np.mean, ci='sd', ax=axs[4], **kwargs)
    # sns.lineplot(x=x, y="released_nodes", data=history_df,
    #              hue=hue, estimator=np.mean, ci='sd', ax=axs[5], **kwargs)
    # axs[3].set(ylim=(0, 200))
    # axs[3].set_title("nodes_in_quarantines")

    # axs[4].set(ylim=(0, 50))
    # axs[4].set_title("contacts_collected")

    # axs[5].set(ylim=(0, 50))
    # axs[5].set_title("released_nodes")

    
    #    sns.lineplot(x=x, y="", data=history_df,
    #                 hue=hue, estimator=np.median, ci='sd', ax=axs[3], **kwargs)
    #    axs[3].set(ylim=(0, 15))
    #    axs[3].set_title("waiting times")



    """
    sns.lineplot(x=x, y="tests_ratio", data=history_df,
                 hue=hue, estimator=np.median, ci='sd', ax=axs[3], **kwargs)
    sns.lineplot(x=x, y="tests_ratio_to_s", data=history_df,
                 hue=hue, estimator=np.median, ci='sd', ax=axs[3], **kwargs)
    axs[3].set_title("tests ratio to all infected, symptomatic infected")
    axs[3].set(ylim=(0, None))
    """

    fig.suptitle(title, fontsize=20)

    if save_path is not None:
        plt.savefig(save_path)

#    plt.show()



def _history_with_fname(filename, group_days: int = None, group_func: str = "max", policy_name: str = None,
                        keep_only_all: bool = False, max_days=None):
    history = _load_history(filename, max_days=max_days)
    if keep_only_all:
        history = history[["day", "all_infectious"]]

    if group_days is not None and group_days > 0:
        history["day"] = history["day"] // group_days * group_days
        history = history.groupby(
            "day", as_index=False).agg(func=group_func)

    history.insert(0, "filename", filename)

    if policy_name is not None:
        history["policy_name"] = policy_name
    return history


def _load_history(filename: str, max_days=None) -> pd.DataFrame:
    print(filename)
    history = pd.read_csv(filename, comment="#")
    if "E" in history.columns:
        history["all_infectious"] = history[[
            "I_n", "I_a", "I_s", "E",
            "I_dn", "I_da", "I_ds", "E_d", "J_ds", "J_dn",
            "J_n", "J_s"]].sum(axis=1)
        history["I_d"] = history[[
            "I_dn", "I_da", "I_ds", "E_d", "J_ds", "J_dn"]].sum(axis=1)
        history["all_tests"] = history[[
            "tests", "quarantine_tests"]].sum(axis=1)

        history["detected_ratio"] = history["all_infectious"] / history["I_d"]
        
        history["tests_ratio"] = history["tests"] / \
            history["all_infectious"]

        history["all_s"] = history[[
            "I_s", "I_ds", "J_s", "J_ds"]].sum(axis=1)
        history["tests_ratio_to_s"] = history["tests"] / history["all_s"]

        history["mean_waiting"] = history["sum_of_waiting"] / history["all_positive_tests"] 
        
        selected_cols = [ 
            col 
            for col in history.columns 
            if "nodes_in_quarantine" in col
        ]
        history["nodes_in_quarantine"] = history[selected_cols].sum(axis=1) 
        selected_cols = [ 
            col 
            for col in history.columns 
            if "released_nodes" in col
        ]
        history["released_nodes"] = history[selected_cols].sum(axis=1) 
        selected_cols = [ 
            col 
            for col in history.columns 
            if "contacts_collected" in col
        ]
        history["contacts_collected"] = history[selected_cols].sum(axis=1) 

    else:
        history["nodes_in_quarantine"] = 0
        history["released_nodes"] = 0 
        history["contacts_collected"] = 0
        history["mean_p_infection"] = 0

    if max_days is not None:
        history = history[:max_days]
    if "day" not in history.columns:
        history["day"] = range(len(history))
#    print(history)
    return history


if __name__ == "__main__":

    history = pd.read_csv(
        "../result_storage/tmp/history_seirsplus_quarantine_1.csv")
    plot_history(history)
