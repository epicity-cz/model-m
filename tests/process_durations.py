import pandas as pd
import numpy as np
import sys 


exp_name = sys.argv[1] 

state_dict = {} 

for i in range(1000):
    with open(f"durations_{exp_name}_{i}.csv", "r")  as f:
        for line  in f:
            values = line.strip().split(",")
            state = values[0]
            times = [ 
                int(x) 
                for x in values[1:]
                if x != ""
            ]

            if state in state_dict:
                state_dict[state].extend(times)
            else:
                state_dict[state] = times

df = pd.DataFrame(index=["median", "mean", "std", "min", "max"])
for s in state_dict.keys():
    l = state_dict[s] 
    if l:
        df[s] = [np.median(l), np.mean(l), np.std(l), min(l), max(l)]


print(df.T)
