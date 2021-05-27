import itertools
import pandas as pd


def extract_students_krouzky_layers(zaci_nodes_df: pd.DataFrame):
    res_df = pd.DataFrame(columns=["id", "name", "weight"])

    names = zaci_nodes_df.columns[4:]
    res_df["name"] = names
    res_df["weight"] = 1.0

    return res_df


def extract_layers_class(student_node_df: pd.DataFrame):
    res_df = pd.DataFrame(columns=["id", "name", "weight"])

    classes = student_node_df["class"].unique()
    classes.sort()
    res_df["name"] = classes
    res_df["weight"] = 1.0

    return res_df


def extract_layers_contacts(contact_df: pd.DataFrame):
    res_df = pd.DataFrame(columns=["id", "name", "weight"])

    names = contact_df["otazka"].unique()
    res_df["name"] = names
    res_df["weight"] = 1.0

    return res_df


def extract_layers_teachers_students(teacher_df: pd.DataFrame):
    res_df = pd.DataFrame(columns=["id", "name", "weight"])

    names = [c for c in teacher_df.columns if c.startswith('pedagog') or c.startswith('asistent')]
    res_df["name"] = names
    res_df["weight"] = 1.0

    return res_df


def concat_layers(layer_list):
    first_row = pd.DataFrame([{"id": 0, "name": "no_layer", "weight": 0.0}])
    res_df = pd.concat(layer_list, ignore_index=True)
    res_df = pd.concat([first_row, res_df], ignore_index=True)

    res_df["id"] = res_df.index

    return res_df
