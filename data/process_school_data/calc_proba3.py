import pandas as pd
import numpy as np

BASEDIR = "skoly_graph"


df = pd.read_csv(f"{BASEDIR}/edges_with_score.csv")


def calc_prob(score):
    probs = np.zeros(len(score))
    int_coefs = np.ones(len(score))
    ONE_DAY = 1

    small = score <= ONE_DAY
    too_big = score > ONE_DAY

    over_one = score[too_big] / ONE_DAY

    probs[small] = score[small]
    probs[too_big] = ONE_DAY  # 1 + np.log(score[too_big])

    int_coefs[too_big] = 1 + np.log10(over_one)

    return probs, int_coefs


# magic constant for schools (elementary)
# MAGIC = 0.5933   # used in first version, before full graph in classes
#MAGIC = 0.675

print(df["score"].max())
idx = df["score"].idxmax()
print(idx)
print(df.iloc[idx])


probs, int_coefs = calc_prob(df["score"])
df["probability"] = probs
print(df)
print(df["probability"].describe())


result = df[["layer", "sublayer", "vertex1", "vertex2", "probability"]]
result["intensity"] = df["base_intensity"] * int_coefs
#result["int_coefs"] = int_coefs

print(result)
print(result["intensity"].describe())

result.to_csv("skoly_graph/edges_final.csv", index=False)
