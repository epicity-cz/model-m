import pickle

import click
import pandas as pd


def load_scenario_dict(filename: str, sep=',', return_data='list'):
    df = pd.read_csv(filename, sep=sep)

    def get_output(series: pd.Series):
        l = series.to_list()

        if return_data == 'list':
            return l
        elif return_data == 'names':
            return {k: v for k, v in zip(df["name"], l)}
        elif return_data == 'ids':
            return {k: v for k, v in zip(df["name"], l)}
        else:
            raise ValueError(f"Unsupported format: {return_data}, valid options are 'list', 'names', 'ids'.")

    data_only = df.drop(columns=['id', 'name'])
    return {k: get_output(data_only[k]) for k in data_only.columns}


@click.command()
@click.argument('filename')
@click.option('--out_path', default=None)
@click.option('--sep', default=',')
@click.option('--return_data', default='list')
def run(filename, out_path, sep, return_data):
    out_dict = load_scenario_dict(filename, sep=sep, return_data=return_data)

    if out_path is not None:
        with open(out_path, 'wb') as of:
            pickle.dump(out_dict, of)

    print(out_dict)


if __name__ == "__main__":
    run()
