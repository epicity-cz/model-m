import numpy as np
import pandas as pd
import re


def map_age(trida, age_diff=5):
    return int(trida[0]) + age_diff


def fix_missing_teachers(ucitel_nodes_df: pd.DataFrame, matrix_df: pd.DataFrame):
    ucitel_new = []

    for _, r in matrix_df.iterrows():
        t_id = r["kod_respondent"]
        data = r.iloc[1:]

        ucitel_data = ucitel_nodes_df[ucitel_nodes_df["id"] == t_id]
        if not len(ucitel_data):
            ucitel_dict = {
                "id": t_id,
                "sex": None,
                "age": None
            }

            ucitel_dict.update(data.to_dict())

            stupen_1 = [c for c in data.index if int(c[7]) < 6]
            stupen_2 = [c for c in data.index if int(c[7]) >= 6]
            ucitel_dict["1. stupeň"] = 1 if any([data[c] > 0 for c in stupen_1]) else 0
            ucitel_dict["2. stupeň"] = 1 if any([data[c] > 0 for c in stupen_2]) else 0

            ucitel_new.append(ucitel_dict)

    return pd.DataFrame(ucitel_new)


def fix_missing_students(zaci_contacts_df: pd.DataFrame, age_diff=5):
    new_zaci = []
    new_zaci_set = set()

    for _, row in zaci_contacts_df.iterrows():
        vertex2 = row["kod_kontakt"]
        trida_kontakt = row["trida_kontakt"]
        trida_kontakt = trida_kontakt if not pd.isna(trida_kontakt) else row["trida"]

        # add a node that is not the primary contact (the student chose not to answer) and hasn't been added before
        if (zaci_contacts_df["kod_respondent"] != vertex2).all() and vertex2 not in new_zaci_set:
            new_zaci.append({"id": vertex2,
                             "sex": None,
                             "class": trida_kontakt,
                             "age": map_age(trida_kontakt, age_diff=age_diff)})
            new_zaci_set.add(vertex2)

    return pd.DataFrame(new_zaci)


def fix_teacher_age(df: pd.DataFrame):
    df["age"] = df["age"].replace({
        '20-29': '25', '30-39': '35', '40-49': '45', '50-59': '55', '60': '65',
    }).fillna(0).astype(int)

    df_sel = df[df["age"] > 20]
    counts = df_sel.groupby("age")["age"].count()
    counts = counts / counts.sum()
    values = list(counts.index)
    probs = list(counts.values)
    N = len(df[df["age"] == 0])

    impute_values = np.random.choice(values, size=N, p=probs)
    df.loc[df.age == 0, "age"] = impute_values
    return df


def extract_students(df: pd.DataFrame, include_activities=True, age_diff=5):
    def extract_student_data(row):
        student_dict = {}
        student_id, student_class = row[0]

        student_dict["id"] = student_id
        student_dict["sex"] = None
        student_dict["age"] = map_age(student_class, age_diff=age_diff)  # age roughly corresponds to the class
        student_dict["class"] = student_class

        if include_activities:
            konicky = row[1]
            konicky.index = [c.replace(":Navštěvuješ nějakou z následujících aktivit ve škole?", "") for c in
                             konicky.index]
            student_dict.update(konicky.to_dict())

        return student_dict

    # drop fields which are relevant to layers only
    zaci_single = df.drop(
        columns=["otazka", "otazka_rec", "kod_kontakt", "intenzita_num", "intenzita", "trida_kontakt"])

    # construct a binary map of hobbies while grouping together fixed student features, then extract all data
    data = [extract_student_data(r)
            for r in zaci_single.groupby(["kod_respondent", "trida"], dropna=False).nunique().iterrows()]
    res_df = pd.DataFrame(data)
    return res_df


def extract_teachers(df: pd.DataFrame, zs=True):
    def extract_teacher_data(row, index):
        teacher_dict = {}

        teacher_vals = row[0]
        # "ZŠ" is splitted to two - one additional feature
        if zs:
            teacher_id, teacher_age, teacher_stupen = teacher_vals[0:3]
        else:
            teacher_id, teacher_age = teacher_vals[0:2]

        teacher_dict["id"] = teacher_id
        teacher_dict["sex"] = None
        teacher_dict["age"] = re.search('\d\d-\d\d|\d\d', teacher_age).group(0)  # extract age or age range

        # convert to one-hot encoding
        if zs and not pd.isna(teacher_stupen):
            teacher_dict["1. stupeň"] = 1 if "1." in teacher_stupen else 0
            teacher_dict["2. stupeň"] = 1 if "2." in teacher_stupen else 0

        # the rest is number of hours taught in a class
        teacher_dict.update(dict(zip(index[3:], teacher_vals[3:])))

        # fill in the binary map of subjects and other binary fields
        udaje = row[1]
        teacher_dict.update(udaje.to_dict())

        return teacher_dict

    # drop fields relevant for layers only
    teachers_single = df.drop(columns=["otazka", "kod_kontakt", "intenzita"])
    # fill invalid values
    if zs:
        teachers_single["stupen"].fillna("Neučí", inplace=True)
    for c in teachers_single.columns:
        if "pedagog" in c:
            teachers_single[c].fillna(0, inplace=True)

    # group together numeric/categorical features, and construct a binary map from the other (binary) features
    group_columns = ["kod_respondent", "vek", "stupen"] if zs else ["kod_respondent", "vek"]
    group_columns += [p for p in teachers_single.columns if p.startswith('pedagog')]

    grouped = teachers_single.groupby(group_columns, dropna=False).nunique()
    data = [extract_teacher_data(r, grouped.index.names) for r in grouped.iterrows()]

    res_df = pd.DataFrame(data)
    return res_df
