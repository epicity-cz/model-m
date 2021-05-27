from zipfile import ZipFile
import sys
import pandas as pd
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt 

OUTNAME = sys.argv[1]
LOGNAMES = sys.argv[2:] 

NUM_NODES = 679

infected_df = pd.DataFrame()

fig, ax = plt.subplots(figsize=(18,12))

#betas = np.linspace(0.1, 2.0, 20
betas = [ "0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9", "1.0", "1.1"]
 #        "1.1", "1.2", "1.3", "1.4", "1.5", "1.6", "1.7"]

df_list = []

for LOGNAME in LOGNAMES:
    for beta in betas:
        print(LOGNAME, beta)
        zipfile = ZipFile(f"history_{beta}_{LOGNAME}.zip")

        infected_list = []
        
        for text_file in zipfile.infolist():
            df = pd.read_csv(zipfile.open(text_file.filename), comment="#")
            S = df.iloc[100]["S"] + df.iloc[100]["S_s"]
            infected_list.append(NUM_NODES - S)


        infected = pd.Series(infected_list)
        df = pd.DataFrame()
        df["num_infected"] = infected
        df["beta"] = float(beta)
        df["exp"] = LOGNAME

        df_list.append(df)
        
        
df = pd.concat(df_list)    



sns.catplot(data=df,
            x="beta",
            y="num_infected",
            hue="exp",
            kind="violin",
            height=5,
            aspect=3)


plt.savefig(f"cat_{OUTNAME}.png")
