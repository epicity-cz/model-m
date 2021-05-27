import pandas as pd
import numpy as np

BASEDIR = './skoly_graph'


df = pd.read_csv(f"{BASEDIR}/edges.csv")


def convert_to_int(names):
    cls_name, cls_name2 = names
    if cls_name != cls_name2:
        return np.nan
    if cls_name == np.nan:
        return np.nan
    return int(cls_name[0])


df["class"] = df[["vertex1_class", "vertex2_class"]].apply(
    convert_to_int, axis=1)
print(df)

# layers
layer_dict = {
    "break_class":  [1],
    "lunch":  [2],
    "club":  [3],
    #   "desk":  [4],
    "out_school":  [5, 6],
    "desk_sometime":  [7, 9],
    "break_others":  [8],
    "free_time":  list(range(41, 72)),
    "secretariat":  [10],
    "work":  [11],
    "out_work":  [12],
    "in_class": list(range(72, 98))
    #    "teacher_student":  list(range(13, 41))
}

teacher_student = list(range(13, 41))
desk = [4]
#in_class = list(range(72, 98))


def _sel(layer):
    sel = df.loc[df["layer"].isin(layer), :]
    print(sel["proba_text"].unique())
    return sel


def _create_values(sel, valdict):
    new_col = sel["proba_text"].replace(valdict)
    print(new_col)
    return new_col


replace_dict = {}

replace_dict["break_class"] = {
    "Jednou za den": 5/7,
    "Několikrát za den": 2*5/7,
    "Občas": 2/7,
    "Málo kdy": 0.5/7
}

replace_dict["lunch"] = {
    "Každý den": 5/7,
    "Občas": 2/7,
    "Málo kdy": 0.5/7
}


replace_dict["club"] = {
    "Každý den":  5*2/7,  # 2 hours in club
    "Občas": 3/7,
    "Málo kdy": 1/7
}

# replace_dict["desk"] = {"lavice":  5*5/7}

replace_dict["out_school"] = {
    "Jednou za den": 5/7,
    "Několikrát za den":  2.5*5/7,
    "Občas": 2/7,
    "Málo kdy": 0.5/7  # jednou za 14 dní
}

replace_dict["desk_sometime"] = {'lavice': 2/7}


replace_dict["break_others"] = {
    "Jednou za den": 5/7,
    "Několikrát za den": 2*5/7,
    "Občas": 2/7,
    "Málo kdy": 0.5/7
}

replace_dict["free_time"] = {np.nan: (3/15)/7}

replace_dict["secretariat"] = {
    "Každý den": 2*5/7,
    "Občas": 2/7,
    "Málo kdy": 0.5/7
}

replace_dict["work"] = {
    "Každý den": 2*5/7,
    "Občas":  2/7,
    "Málo kdy": 0.5/7
}

replace_dict["out_work"] = {
    "Každý den": 5/7,
    "Občas":     2/7,
    "Málo kdy":  0.5/7
}


for layer_name, layer_list in layer_dict.items():
    sel = _sel(layer_list)
    new_col = _create_values(sel, replace_dict[layer_name])
    df.loc[df["layer"].isin(layer_list), "score"] = new_col


sel = _sel(teacher_student)
hours = sel["hours_a_week"].replace(
    {"10 a více": 12}).astype(float).astype(int)

# if hours > 5:
#     hours_under_5 = 5
# else:
#     hours_under_5 = hours
# hours_under_5 = np.clip(hours, 0, 5)
df.loc[df["layer"].isin(teacher_student), "score"] = (
    hours/7) * (3.5/25)  # 25 pocet deti ve tride TODO


sel = _sel(desk)


def create_score(classnum):
    if classnum <= 5:
        return 25/7
    else:
        return 30/7


new_value = sel["class"].apply(create_score)
print(new_value)
df.loc[df["layer"].isin(desk), "score"] = new_value


df.rename(columns={'intensity': 'base_intensity'}, inplace=True)

df.to_csv(f"{BASEDIR}/edges_with_score.csv", index=False)
print("Saved.")
