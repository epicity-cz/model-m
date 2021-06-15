import glob
import zipfile

import click
import os
import numpy as np
import pandas as pd
import seaborn as sns

import matplotlib.pyplot as plt


def _add_id_column(df, fname):
    fname = os.path.basename(fname).replace('.csv', '')
    df["id"] = fname


def process_zip(zip_path: str, save_feather=False):
    tmp_dir = zip_path + '.tmp'

    try:
        os.mkdir(tmp_dir)
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(tmp_dir)

        csv_files = [file for file in glob.glob(os.path.join(tmp_dir, '*.csv'))]

        dfs = {cf: pd.read_csv(cf, comment='#') for cf in csv_files}

        out_dfs = []
        for fname, df in dfs.items():
            _add_id_column(df, fname)
            out_dfs.append(df)

        res = pd.concat(out_dfs, ignore_index=True)
        if save_feather:
            res.to_feather(zip_path.replace('.zip', '.feather'))

    finally:
        # clean temp dir
        csv_files = [file for file in glob.glob(os.path.join(tmp_dir, '*.csv'))]
        for file in csv_files:
            os.remove(file)

        os.rmdir(tmp_dir)

    return res


def plot_dfs(dfs, column, figsize, out_path, xlabel, ylabel, labels=None, title=None, ymax=None, use_median=True,
             use_sd=False, fit_me=None, show_whole_fit=False, day_indices=None, day_labels=None):

    fig, ax = plt.subplots(figsize=figsize)

    estimator = np.median if use_median else np.mean
    ci = 'sd' if use_sd else None

    if title is not None:
        ax.set_title(title, fontsize=14)

    # list of all dfs including the fit data - to infer correct plot limits
    lim_dfs = dfs if fit_me is None else dfs + [fit_me]
    xlim_dfs = lim_dfs if show_whole_fit else dfs

    xmin = min([df['T'].min() for df in xlim_dfs])
    xmax = max([df['T'].max() for df in xlim_dfs])

    ymax = ymax if ymax is not None else max([df[column].max() for df in lim_dfs])
    ax.set_ylim(ymin=0.0, ymax=ymax)
    ax.set_xlim(xmin=xmin, xmax=xmax)

    for df in dfs:
        sns.lineplot(x='T', y=column, data=df, label=None,
                     estimator=estimator, ci=ci, ax=ax)

        if not use_sd:
            df_stats = df.groupby(["T"]).describe()
            q1 = df_stats[column]["25%"]
            q3 = df_stats[column]["75%"]

            ax.fill_between(df["T"].unique(), q1, q3, alpha=0.3)

    if fit_me is not None:
        ax.plot(fit_me['T'], fit_me[column])

    if labels is not None:
        ax.legend(labels)

    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)

    if day_indices is not None:
        ax.set_xticks([])

        ax.xaxis.set_minor_locator(plt.FixedLocator(day_indices))
        ax.xaxis.set_minor_formatter(plt.FixedFormatter(day_labels))
        ax.grid(which="minor", axis="x", linestyle="--", linewidth=1)
        plt.setp(ax.xaxis.get_minorticklabels(), rotation=70)

    fig.tight_layout()
    plt.savefig(out_path)


@click.command()
@click.argument('plot_files', nargs=-1)
@click.option('--label_names', default=None, help="Labels for each file to show in legend (separated by comma), "
                                                  "called as --label_names l_1,l_2, ... "
                                                  "If --fit_me is used, the last label in this argument is the label of"
                                                  " the fit data.")
@click.option('--out_file', default='./plot_out.png', help="Path where to save the plot.")
@click.option('--fit_me', default=None, help="Path to a .csv file with fit data.")
@click.option('--show_whole_fit/--show_partial_fit', default=False, help="If true, plot the whole fit data on the x"
                                                                         "axis (may be longer than other data).")
@click.option('--title', default=None, help="Title of the plot.")
@click.option('--column', default='I_d', help="Column to use for plotting.")
@click.option('--zip_to_feather/--no_zip_to_feather', default=False, help="If True, save processed .zips to .feather. "
                                                                          "The file name is the same except the "
                                                                          "extension.")
@click.option('--figsize', nargs=2, default=(6, 5), help="Size of the plot (specified without commas: --figsize 6 5).")
@click.option('--ymax', default=None, type=int, help="Y axis upper limit. By default the limit is inferred from source"
                                                     "data.")
@click.option('--xlabel', default='Day', help="X axis label.")
@click.option('--ylabel', default='Number of cases', help="Y axis label.")
@click.option('--use_median/--use_mean', default=True, help="Use median or mean in the plot (default median).")
@click.option('--use_sd/--use_iqr', default=False, help="Use sd or iqr for shades (default iqr).")
@click.option('--day_indices', default=None, help="Use dates on x axis - maps indices to labels (e.g. 5,36,66).")
@click.option('--day_labels', default=None, help="Use dates on x axis - string labels (e.g. "
                                                 "\"March 1,April 1,May 1\").")
def run(plot_files, label_names, out_file, fit_me, show_whole_fit, title, column, zip_to_feather, figsize, ymax, xlabel,
        ylabel, use_median, use_sd, day_indices, day_labels):
    """ Create plot using an arbitrary number of input files. Optionally, plot a fit curve specified by --fit_me

    \b
    PLOT_FILES   name of the input files - either .csv, .zip or .feather
    """

    if label_names is not None:
        label_names = label_names.split(',')

    if day_indices is not None or day_labels is not None:
        assert day_indices is not None and day_labels is not None, "Both --day_indices and --day_labels must be " \
                                                                   "passed to the script if string x axis labels are " \
                                                                   "used."
        day_labels = day_labels.split(',')

        # check if indices are valid
        try:
            day_indices = [int(i) for i in day_indices.split(',')]
        except ValueError as e:
            raise ValueError("Argument --day_indices must be a comma separated list of ints (e.g. 5,10,25)") from e

        assert len(day_labels) == len(day_indices), "Arguments --day_indices and --day_labels must have the same" \
                                                    "number of values."

    assert len(plot_files), "No input files were passed to the script."

    dfs = []

    for file in plot_files:
        print(f"Processing file {file}...")

        if file.endswith('.feather'):
            df = pd.read_feather(file)
        elif file.endswith('.zip'):
            df = process_zip(file, save_feather=zip_to_feather)
        elif file.endswith('.csv'):
            df = pd.read_csv(file,  comment='#')
            _add_id_column(df, file)
        else:
            raise ValueError(f"Unsupported file: {file}, supported extensions - .zip, .feather")

        dfs.append(df)

    if fit_me is not None:
        print(f"Reading fit file - {fit_me}...")
        fit_df = pd.read_csv(fit_me)
    else:
        fit_df = None

    print("All files processed.")

    plot_dfs(dfs, column, figsize, out_file, xlabel, ylabel, labels=label_names, title=title, ymax=ymax,
             use_median=use_median, use_sd=use_sd, day_indices=day_indices, day_labels=day_labels, fit_me=fit_df,
             show_whole_fit=show_whole_fit)


if __name__ == "__main__":
    run()
