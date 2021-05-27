import itertools
import warnings
import pandas as pd


def students_krouzky_layers(zaci_nodes_df: pd.DataFrame):
    contacts = []

    for c in zaci_nodes_df.columns[4:]:
        # extract according to one-hot map
        zaci_krouzek = zaci_nodes_df[zaci_nodes_df[c] == 1]

        if not len(zaci_krouzek):
            continue

        # all students attending a leisure activity have a contact in pairs
        ids = itertools.product(zaci_krouzek["id"], zaci_krouzek["id"])
        for i_1, i_2 in ids:
            if i_1 == i_2:
                continue

            contacts.append({
                "layer": c,
                "sublayer": 0,
                "vertex1": i_1,
                "vertex2": i_2,
            })

    return pd.DataFrame(contacts)


def students_students_layers(students_contact_df: pd.DataFrame):
    # fix swapped layer names ("jine" is exclusive for contacts from a DIFFERENT class, and vice-versa)
    fix_contact_map = {
        'prestavka': 'prestavka_jine',
        'prestavka_jine': 'prestavka',
        'mimo_skolu': 'mimo_jine',
        'mimo_jine': 'mimo_skolu',
        'obed': 'obed_jine',
        'obed_jine': 'obed',
        'druzina': 'druzina',
        'lavice_jine': 'lavice_trida',
        'lavice_trida': 'lavice_jine',
        'lavice_hlavni': 'lavice_jine'
    }

    def process_row(row):
        layer = row["otazka"]
        vertex1 = row["kod_respondent"]
        vertex2 = row["kod_kontakt"]

        # answer to contact frequency, has to be converted do proba and intensity in our graph later
        probability_text = row["intenzita"] if not pd.isna(row["intenzita"]) else f"lavice"
        probability_text = probability_text.rstrip(' ').lstrip(' ')
        probability_value = row["intenzita_num"]

        if "lavice" in layer:
            assert pd.isna(probability_value)
            probability_value = 3 if "hlavni" in layer else 2

        trida = row["trida"]
        trida_kontakt = row["trida_kontakt"]
        trida_kontakt = trida_kontakt if not pd.isna(trida_kontakt) else row["trida"]

        # fix layer name according to class
        same_as_jine = trida == trida_kontakt and "jine" in layer
        jine_as_same = trida != trida_kontakt and "jine" not in layer
        if same_as_jine or jine_as_same:
            print(f"Layer change: {layer} -> {fix_contact_map[layer]}, classes: {trida}, {trida_kontakt}")
            layer = fix_contact_map[layer]

        return {
            "layer": layer,
            "sublayer": 0,
            "vertex1": vertex1,
            "vertex2": vertex2,
            "vertex1_class": trida,
            "vertex2_class": trida_kontakt,
            "proba_text": probability_text,
            "proba_value": probability_value
        }

    def intensity_valid(row):
        if row["intenzita_num"] > 0:
            return True

        return "lavice" in row["otazka"]

    # process contacts row by row, skip self-loops - that happens e.g. if a student does not sit next to anyone
    layers = pd.DataFrame([process_row(r) for _, r in students_contact_df.iterrows() if
                           r["kod_respondent"] != r["kod_kontakt"]
                           and not pd.isna(r["kod_kontakt"])
                           and intensity_valid(r)])
    return layers


def class_layers(student_node_df: pd.DataFrame):
    data = []

    for _, row in student_node_df.iterrows():
        layer = row["class"]

        classmates = student_node_df[student_node_df["class"] == layer]
        for _, c in classmates.iterrows():
            vertex1 = row["id"]
            vertex2 = c["id"]
            if vertex1 >= vertex2:
                continue

            proba_text = f"Třída {1 if int(layer[0]) < 6 else 2}. stupeň"

            data.append({
                "layer": layer,
                "sublayer": 0,
                "vertex1": vertex1,
                "vertex2": vertex2,
                "proba_text": proba_text
            })
    return pd.DataFrame(data)


def teacher_teacher_layers(teacher_contact_df: pd.DataFrame, zs=True):
    def process_row(row):
        layer = row["otazka"]
        vertex1 = row["kod_respondent"]
        vertex2 = row["kod_kontakt"]

        # "ZŠ" teacher dataframe has a different intensity mapping
        otazka_map = {
            1: 'Málo kdy',
            2: 'Občas',
            3: 'Každý den'
        }
        probability_value = row["intenzita"] if zs else row["weight"]
        probability_text = otazka_map[probability_value] if zs else row["intenzita"]
        probability_text = probability_text.rstrip(' ').lstrip(' ')

        return {
            "layer": layer,
            "sublayer": 0,
            "vertex1": vertex1,
            "vertex2": vertex2,
            "proba_text": probability_text,
            "proba_value": probability_value
        }

    layers = pd.DataFrame([process_row(r) for _, r in teacher_contact_df.iterrows() if
                           r["kod_respondent"] != r["kod_kontakt"] and not pd.isna(r["kod_kontakt"])])
    return layers


def students_teachers_layers(students_node_df: pd.DataFrame, teachers_node_df: pd.DataFrame):
    res_layers = []

    for pedagog in teachers_node_df.columns:
        if not pedagog.startswith('pedagog') and not pedagog.startswith('asistent'):
            continue

        class_name = pedagog[-2:]
        # hours per week in some columns have values like '10 a více', and some teachers have not answered - hence NaN
        class_teachers = teachers_node_df[
            (teachers_node_df[pedagog] != 0) & (teachers_node_df[pedagog] != '0') & ~pd.isna(teachers_node_df[pedagog])
            ]
        class_students = students_node_df[students_node_df["class"] == class_name]

        # print(class_name)
        # print(len(class_students))
        # print(len(class_teachers))
        # print()
        if pedagog == 'pedagog6A':
            print("now")

        # for now, when a teacher teaches a class, they have a contact with all students of the class
        for z_id, t_id in itertools.product(class_students.index, class_teachers.index):
            student = class_students.loc[z_id]
            teacher = class_teachers.loc[t_id]

            res_layers.append({
                "layer": pedagog,
                "sublayer": 0,
                "vertex1": student["id"],
                "vertex2": teacher["id"],
                "hours_a_week": teacher[pedagog]
            })

    return pd.DataFrame(res_layers)


def smooth_layers(layer_df: pd.DataFrame):
    res_df = []
    seen = set()
    switched = 0

    # smooth symmetries - keep only one undirected edge and only the maximum of contact intensities
    for layer in layer_df["layer"].unique():
        layers_df = layer_df[layer_df["layer"] == layer]

        for _, r in layers_df.iterrows():
            assert r["vertex1"] != r["vertex2"]

            other_row = layers_df[(layers_df["vertex2"] == r["vertex1"]) & (layers_df["vertex1"] == r["vertex2"])]

            if not len(other_row) < 2:
                print("Detected duplicated or broken rows:")
                print(other_row)
                if not all([len(other_row[c].unique()) == 1 for c in other_row.columns]):
                    warnings.warn(f"Invalid response of student {other_row.iloc[0]['vertex1']} "
                                  f"about their contact {other_row.iloc[0]['vertex2']}.")

            # only one contact answered
            if not len(other_row):
                res_df.append(r)
                continue

            other_row = other_row.iloc[0]

            if r["vertex1"] > r["vertex2"]:
                other_row, r = r, other_row

            if (r["vertex1"], r["vertex2"]) in seen:
                continue

            # note that "proba_value" is a preliminary value, and sometimes one value corresponds to multiple texts
            if other_row["proba_value"] > r["proba_value"]:
                switched += 1
                r["proba_value"] = other_row["proba_value"]
                r["proba_text"] = other_row["proba_text"]

            res_df.append(other_row)
            seen.add((r["vertex1"], r["vertex2"]))

    print(f"\nSwitched values of {switched} rows.")
    return pd.DataFrame(res_df)
