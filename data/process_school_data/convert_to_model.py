import os

import click
import node_data
import edge_data
import layer_data
import pandas as pd


def fix_invalid_ids(df, column_name):
    only_valid_ids = pd.to_numeric(df[column_name], errors='coerce').notnull()
    df = df[only_valid_ids]
    df = df[~pd.isna(df[column_name])]
    df[column_name] = df[column_name].astype(int)
    return df


@click.command()
@click.argument('student_file', default="./school_data/data_ZS_zaci_anonymni_fin.xlsx")
@click.argument('teacher_file', default="./school_data/data_ZS_ucitele_anonymni.xlsx")
@click.argument('teacher_class_matrix', default="./school_data/teacher_class_matrix.xlsx")
@click.option('--save_dir', default='./skoly_graph/')
@click.option('--exclude_konicky', is_flag=True, default=False)
@click.option('--zs/--ss', default=True)
@click.option('--age_diff', default=5, help="E.g. a first grader's age is 1+5 = 6 years, but for high school, a first "
                                            "grade student's age is 1 + 14 = 15 years. "
                                            "This value is used to deduce student age from they class, as both age and "
                                            "gender is anonymized by default.")
def convert(student_file, teacher_file, teacher_class_matrix, save_dir, exclude_konicky, zs, age_diff):
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    # --- nodes ---
    # students
    student_contact_df = pd.read_excel(student_file)
    student_contact_df = fix_invalid_ids(student_contact_df, "kod_respondent")
    student_contact_df = fix_invalid_ids(student_contact_df, "kod_kontakt")

    student_node_df = node_data.extract_students(student_contact_df,
                                                 include_activities=not exclude_konicky,
                                                 age_diff=age_diff)
    students_missing = node_data.fix_missing_students(student_contact_df,
                                                      age_diff=age_diff)

    student_node_df = pd.concat([student_node_df, students_missing], ignore_index=True)

    # teachers
    teacher_contact_df = pd.read_excel(teacher_file)
    teacher_matrix = pd.read_excel(teacher_class_matrix)

    teacher_node_df = node_data.extract_teachers(teacher_contact_df, zs=zs)
    teachers_missing = node_data.fix_missing_teachers(teacher_node_df, teacher_matrix)

    teacher_node_df = pd.concat([teacher_node_df, teachers_missing], ignore_index=True)
    teacher_node_df = node_data.fix_teacher_age(teacher_node_df)

    all_nodes = pd.concat([student_node_df, teacher_node_df], ignore_index=True)
    all_nodes.sort_values("id", inplace=True)

    # --- edges ---
    student_edges = edge_data.students_students_layers(student_contact_df)
    teacher_edges = edge_data.teacher_teacher_layers(teacher_contact_df, zs=zs)
    # smoothe symmetries
    student_edges = edge_data.smooth_layers(student_edges)
    teacher_edges = edge_data.smooth_layers(teacher_edges)

    student_teacher_edges = edge_data.students_teachers_layers(student_node_df, teacher_node_df)
    all_edges = [student_edges, teacher_edges, student_teacher_edges]

    if not exclude_konicky:
        konicky_edges = edge_data.students_krouzky_layers(student_node_df)
        all_edges.append(konicky_edges)

    class_edges = edge_data.class_layers(student_node_df)
    all_edges.append(class_edges)
    all_edges = pd.concat(all_edges, ignore_index=True)

    # --- layers ---
    student_layers = layer_data.extract_layers_contacts(student_contact_df)
    teacher_layers = layer_data.extract_layers_contacts(teacher_contact_df)
    # teacher_node_df has missing teachers as well, which is important for the student-teacher relationship
    student_teacher_layers = layer_data.extract_layers_teachers_students(teacher_node_df)
    all_layers = [student_layers, teacher_layers, student_teacher_layers]

    if not exclude_konicky:
        konicky_layers = layer_data.extract_students_krouzky_layers(student_node_df)
        all_layers.append(konicky_layers)

    # all students in the same class
    class_layers = layer_data.extract_layers_class(student_node_df)
    all_layers.append(class_layers)

    all_layers = layer_data.concat_layers(all_layers)

    # save model data
    edges_ids = dict(zip(all_layers["name"], all_layers["id"]))
    all_edges["layer"] = [edges_ids[r] for r in all_edges["layer"]]

    # edges with fixed probas and intensities
    edges_usable = all_edges[["layer", "sublayer", "vertex1", "vertex2"]]
    edges_usable["probability"] = 1.0
    edges_usable["intensity"] = 1.0

    all_nodes.to_csv(os.path.join(save_dir, 'nodes.csv'), index=False)
    all_edges.to_csv(os.path.join(save_dir, 'edges.csv'), index=False)
    edges_usable.to_csv(os.path.join(save_dir, 'edges_usable.csv'), index=False)
    all_layers.to_csv(os.path.join(save_dir, 'layers.csv'), index=False)


if __name__ == "__main__":
    convert()
