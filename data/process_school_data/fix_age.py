import numpy as np
import pandas as pd
import sys

nodes_file =  sys.argv[1]

df = pd.read_csv(nodes_file, index_col=0)

print(df["age"].unique())

df["age"] = df["age"].replace({'20-29': '25', '30-39': '35', '40-49': '45', '50-59': '55', '60': '65',
                              }).fillna(0).astype(int)


print(df["age"].unique())

df_sel = df[df["age"]>20]
counts = df_sel.groupby("age")["age"].count()
counts = counts / counts.sum()
values = list(counts.index)
probs = list(counts.values)
N = len(df[df["age"]==0])

impute_values =  np.random.choice(values, size=N, p=probs)
df.loc[df.age == 0, "age"] = impute_values

print(df["age"])

df.to_csv("nodes_fixed_age.csv")
