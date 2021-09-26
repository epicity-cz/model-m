import click
import os
import pandas as pd


@click.command()
@click.argument('out_path', default="okresy_aggregated.csv")
@click.option('--in_dir', default='./okresy/', help='Directory where the data per region are stored.')
@click.option('--exclude', default='Praha.csv,Karviná.csv,Frýdek-Místek.csv', help='Region files to exclude.')
@click.option('--aggregate', default='mean', help='Aggregation function - mean or median (default mean).')
def run(out_path, in_dir, exclude, aggregate):
    """ Aggregate several .csv files into one file. Supported methods are mean and median. Each .csv file
        represents epidemic statistic from one Czech region.

            \b
            OUT_PATH   path where to save the aggregated csv file
    """

    exclude = exclude.split(',')
    result_df = None

    for file in os.listdir(in_dir):
        if file in exclude:
            continue

        df = pd.read_csv(os.path.join(in_dir, file), sep=',', index_col=0)
        result_df = pd.concat([result_df, df], axis=0) if result_df is not None else df

    result_df = result_df.groupby(by='datum')

    if aggregate == 'mean':
        result_df = result_df.mean()
    elif aggregate == 'median':
        result_df = result_df.median()
    else:
        raise ValueError("Unknown aggregating function.")

    result_df = result_df.reset_index()
    result_df.to_csv(out_path)
    return result_df


if __name__ == "__main__":
    run()
