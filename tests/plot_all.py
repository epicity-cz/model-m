import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from zipfile import ZipFile
import sys

from matplotlib.ticker import FixedFormatter, FixedLocator

from zipfile import ZipFile

outputname = sys.argv[1]
expname = sys.argv[2] 
title = sys.argv[3]

zf = ZipFile(f"history_{expname}.zip")

dfs = [
    pd.read_csv(zf.open(textfile.filename), comment="#")
    for textfile in zf.infolist()
]

cr_scaled = pd.read_csv("~/covid/model-m/data/fit_me.csv")


history_df = pd.concat(dfs)

history_df["all_infectious"] = history_df[[
    "I_n", "I_a", "I_s", "E",
    "J_n", "J_s"]].sum(axis=1)

#history_df["I_d"] = history_df[[
#    "I_dn", "I_da", "I_ds", "E_d", "J_ds", "J_dn"]].sum(axis=1)

#history_df["all_tests"] = history_df[["tests", "quarantine_tests"]].sum(axis=1)
history_df["all_tests"] = 0

history_df["detected_ratio"] = history_df["all_infectious"] / history_df["I_d"]

#history_df["mean_waiting"] = history_df["sum_of_waiting"] / \
#    history_df["all_positive_tests"]
history_df["mean_waiting"] = 0


first_days = [5, 36, 66,
              97, 127, 158,
              189, 219, 250]
labels = ["March 1", "April 1", "May 1",
          "June 1", "July 1", "August 1",
          "September 1", "October 1", "November 1"]

fig = plt.figure(figsize=(28, 9))

axs = [None] * 5
axs[0] = fig.add_subplot(121)
axs[1] = fig.add_subplot(122)
"""
axs[2] = fig.add_subplot(333)
axs[3] = fig.add_subplot(336)
axs[4] = fig.add_subplot(339)
"""

"""
######
# Ratio of all active infected and active detected
######

sns.lineplot(x="T", y="detected_ratio",
             data=history_df,  estimator=np.median, ci='sd', ax=axs[3], label="Model")
axs[3].xaxis.set_minor_locator(FixedLocator(first_days))
axs[3].xaxis.set_minor_formatter(FixedFormatter(labels))
axs[3].grid(which="minor", axis="x", linestyle="--", linewidth=1)
plt.setp(axs[3].xaxis.get_minorticklabels(), rotation=70)

axs[3].set(xlim=(1, 150))
axs[3].set(ylim=(0, 15))
axs[3].set_ylabel("Ratio", fontsize=12)

axs[3].axhline(y=10)


######
# Number of all tests
######

cr_scaled["T"] = cr_scaled.index

sns.lineplot(x="T", y="all_tests", data=history_df,
             estimator=np.median,  ci='sd', ax=axs[3],
             label="Model")

sns.lineplot(x="T", y="all_tests", data=cr_scaled,
             estimator=np.median,  ax=axs[3], color="orange",
             label="ČR (scaled)")


axs[3].xaxis.set_minor_locator(FixedLocator(first_days))
axs[3].xaxis.set_minor_formatter(FixedFormatter(labels))
axs[3].grid(which="minor", axis="x", linestyle="--", linewidth=1)
plt.setp(axs[3].xaxis.get_minorticklabels(), rotation=70)

axs[3].set(xlim=(1, 120))
axs[3].set(ylim=(0, 50))
axs[3].set_ylabel("Number of tests", fontsize=12)



######
# Waiting times
######

sns.lineplot(x="T", y="mean_waiting", data=history_df,
             estimator=np.median,  ci='sd', ax=axs[2],
             label="Model")

sns.lineplot(x="T", y="mean_waiting", data=cr_scaled,
             estimator=np.median,  ax=axs[2], color="orange",
             label="ČR (scaled)")


axs[2].xaxis.set_minor_locator(FixedLocator(first_days))
axs[2].xaxis.set_minor_formatter(FixedFormatter(labels))
axs[2].grid(which="minor", axis="x", linestyle="--", linewidth=1)
plt.setp(axs[2].xaxis.get_minorticklabels(), rotation=70)

axs[2].set(xlim=(1, 120))
axs[2].set(ylim=(0, 10))
axs[2].set_ylabel("Mean waiting time", fontsize=12)
"""


######
# All active cases
######

def _clever_mean(a):
    sa = np.sort(a)
    if len(sa) < 100:
        return np.mean(sa)
    else:
        ommit = int(5*(len(sa)/100))  # ommit 10% 
        return np.mean(sa[ommit:-ommit])


sns.lineplot(x="T", y="all_infectious", data=history_df,
             estimator=np.median,  ci='sd', ax=axs[1],
             label="Model", color="blue")
sns.lineplot(x="T", y="all_infectious", data=history_df,
             estimator=np.mean,  ci='sd', ax=axs[1],
             label="Model", color="green")
sns.lineplot(x="T", y="all_infectious", data=history_df,
             estimator=_clever_mean,  ci='sd', ax=axs[1],
             label="Model", color="orange")


axs[1].xaxis.set_minor_locator(FixedLocator(first_days))
axs[1].xaxis.set_minor_formatter(FixedFormatter(labels))
axs[1].grid(which="minor", axis="x", linestyle="--", linewidth=1)
plt.setp(axs[1].xaxis.get_minorticklabels(), rotation=70)

axs[0].set(xlim=(1, 250))
axs[0].set(ylim=(0, 200))

axs[1].set_ylabel("Number of active cases (all)", fontsize=12)


######
# Detected active cases
######

sns.lineplot(x="T", y="I_d", data=history_df,
             estimator=np.median,  ci='sd', ax=axs[0],
             label="Model", color="blue")
sns.lineplot(x="T", y="I_d", data=history_df,
             estimator=np.mean,  ci='sd', ax=axs[0],
             label="Model", color="green")
sns.lineplot(x="T", y="I_d", data=history_df,
             estimator=_clever_mean,  ci='sd', ax=axs[0],
             label="Model", color="orange")

sns.lineplot(x="T", y="I_d", data=cr_scaled,
             estimator=np.median,  ax=axs[0], color="red",
             label="ČR (scaled)")


axs[0].xaxis.set_minor_locator(FixedLocator(first_days))
axs[0].xaxis.set_minor_formatter(FixedFormatter(labels))
axs[0].grid(which="minor", axis="x", linestyle="--", linewidth=1)
plt.setp(axs[0].xaxis.get_minorticklabels(), rotation=70)

axs[0].set(xlim=(1, 250))
axs[0].set(ylim=(0, 200))
axs[0].set_ylabel("Number of active cases (detected)", fontsize=12)

axs[0].axvline(237)

fig.suptitle(f"{title}")

plt.tight_layout()

axs[0].get_figure().savefig(f"{outputname}.png")

