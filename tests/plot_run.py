from zipfile import ZipFile
import sys
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt 
import numpy as np

OUTNAME = sys.argv[1]
LOGNAMES = sys.argv[2:] 

fig, ax = plt.subplots(figsize=(9,6))

colors = [
    "blue", "green", "orange",
    "red", "magenta", "purple",
    "pink"
] * 5

for i, LOGNAME in enumerate(LOGNAMES):
    print(f"Reading {LOGNAME} ...", end="") 
    zipfile = ZipFile(f"history_{LOGNAME}.zip")

    df_list = [
        pd.read_csv(zipfile.open(text_file.filename), comment="#")
        for text_file in zipfile.infolist()
    ]

    infected = ["E", "I_a", "I_n", "I_s", "J_n", "J_s"]

    for df in df_list:
        df["all_infected"] = df[infected].sum(axis=1) 
    
    df_plot = pd.concat(df_list)
    print(f" loaded.") 

    print(f"Plotting {LOGNAME} ...", end="") 
    sns.lineplot(x="T", y="all_infected", data=df_plot, estimator=np.mean, ci='sd',
                      label=f"{LOGNAME} (mean)", color=colors[i], ax=ax)
    sns.lineplot(x="T", y="all_infected", data=df_plot, estimator=np.median, ci=None, ls="--",
                 label=f"{LOGNAME} (median)", color=colors[i], ax=ax)
    
    print(f" done.") 

ax.set(ylim=(0, 180))    
ax.set(xlim=(35, 100))    

for i in range(0, 100, 7):
    ax.axvline(i, ls=":", color="gray")

plt.savefig(f"run_{OUTNAME}.png")
print("Figure saved.")
