from zipfile import ZipFile
import sys
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt 

OUTNAME = sys.argv[1]
LOGNAMES = sys.argv[2:] 

NUM_NODES = 679

infected_df = pd.DataFrame()

fig, ax = plt.subplots(figsize=(18,12))


for LOGNAME in LOGNAMES:
    zipfile = ZipFile(f"history_{LOGNAME}.zip")

    infected_list = []
    
    for text_file in zipfile.infolist():
        df = pd.read_csv(zipfile.open(text_file.filename), comment="#")
        S = df.iloc[100]["S"] + df.iloc[100]["S_s"]
        infected_list.append(NUM_NODES - S)


    infected = pd.Series(infected_list)
    print(infected.describe())

    infected_df[LOGNAME] = infected

    
infected_df = infected_df.stack().reset_index(level=1)
infected_df.columns=["exp", "overall_infected"]
print(infected_df)

sns.violinplot(y="exp", x="overall_infected", data=infected_df, cut=0, split=True, orient="h", ax = ax)
#sns.swarmplot(x="exp", y="overall_infected", data=infected_df,  ax = ax)
ax.set(xlim=(0, None))
ax.set_title("Kumulativní počet nemocných 100. den. (1000 běhů)") 
plt.savefig(f"violin_{OUTNAME}.png")
