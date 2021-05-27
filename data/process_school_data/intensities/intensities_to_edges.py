import os

import click
import pandas as pd


def maybe_times(val: str, contact_map: dict):
    if '*' in val:
        val = val.split('*')
        return contact_map[val[0]] * float(val[1])
    return contact_map[val]


def parse_intensity(i_val: str, contact_map: dict):
    if i_val not in contact_map:
        i_val = i_val.replace("(", "")
        i_val = i_val.replace(" ", "")
        i_val = i_val.split(')/')
        left, div = i_val[0].split('+'), i_val[1]
        return sum(maybe_times(i, contact_map) for i in left) / float(div)

    return contact_map[i_val]


@click.command()
@click.argument('edge_file', default="../skoly_graph/edges.csv")
@click.argument('intensity_file', default="./zs.csv")
@click.argument('contact_map_file', default="./contacts.csv")
@click.option('--save_dir', default='../skoly_graph/')
def convert(edge_file, intensity_file, contact_map_file, save_dir):
    edge_df = pd.read_csv(edge_file)
    intensity_df = pd.read_csv(intensity_file)  # fill in intensities according to layer.csv, use layer id

    contact_intensity_map = {}
    with open(contact_map_file, 'r') as f:
        for line in f:
            line = line[:-1].split(',')
            contact_intensity_map[line[0]] = float(line[1])

    intensities = []
    for _, r in edge_df.iterrows():
        layer_row = intensity_df.iloc[r["layer"]]
        intensity = parse_intensity(layer_row["intensity"], contact_intensity_map)
        intensities.append(intensity)

    edge_df["intensity"] = intensities

    edge_df.to_csv(os.path.join(save_dir, "edges.csv"), index=False)


if __name__ == "__main__":
    convert()
