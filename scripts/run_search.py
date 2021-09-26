from hyperparam_search.hyperparam_utils import run_hyperparam_search
from utils.config_utils import ConfigFile
import timeit
import random
import pandas as pd
import click
import datetime
import numpy as np
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))


def load_gold_data(csv_path, first_n_zeros=0, data_column=2, from_day=0, until_day=None, use_dates=False):
    df = pd.read_csv(csv_path)

    if use_dates:
        dates = pd.to_datetime(df["datum"])
        dates = dates - dates[0]
        dates = dates.apply(lambda t: t.days)
    else:
        dates = df["T"]

    result = pd.DataFrame({"day": range(dates.max() + 1), "infected": pd.NA})

    if isinstance(data_column, int):
        data_vals = df.iloc[:, data_column].to_list()
    else:
        data_vals = df[data_column].to_list()
    result["infected"] = data_vals

    result.fillna(method='ffill', inplace=True)

    if first_n_zeros > 0:
        result["day"] += first_n_zeros

        first_n = [{"day": i + 1, "infected": 0} for i in range(first_n_zeros)]
        result = pd.concat([pd.DataFrame(first_n), result], ignore_index=True)

    result = result.iloc[from_day:until_day]

    return result


@click.command()
@click.argument('filename', default="town0.ini")
@click.argument('hyperparam_filename', default="example_gridsearch.json")
@click.option('--set-random-seed/--no-random-seed', ' /-r', default=True,
              help="Random seed to set for the experiment. If `run_n_times` > 1, the seed is incremented by i for "
                   "the i-th run.")
@click.option('--n_jobs', default=0)
@click.option('--from_day', default=1, help="Lower bound for days in the gold data DataFrame (indexed from 0).")
@click.option('--until_day', default=None, type=int, help="Upper bound for days in the gold data DataFrame "
                                                          "(python-like indexing - elements up to 'until_day - 1' are "
                                                          "selected).")
@click.option('--use_dates/--use_T', default=False, help="If True, use column 'datum' for gold data indexing, else use "
                                                         "column 'T' (default).")
@click.option('--use_config_days/--use_args_days', default=False, help="If True, use 'start_day' and 'n_days' from "
                                                                       "the config file, otherwise using 'from_day' "
                                                                       "and 'until_day' command line arguments.")
@click.option('--run_n_times', default=1, help="Number of times to run th experiment with specific hyperparameter "
                                               "settings.")
@click.option('--first_n_zeros', default=0, help="Shifts gold data by this value - the first day is incremented"
                                                 " by `first_n_zeros`, and the data is padded with `first_n_zeros` days"
                                                 " with the gold values set to zero.")
@click.option('--fit_column', default='I_d', help="Data column to use for fit.")
@click.option('--fit_data', default='../data/fit_data/fit_me.csv',
              help="A DataFrame that has a column named 'datum' and contains the gold data in the column `data_column`.")
@click.option('--return_func', default='rmse', help="Loss function.")
@click.option('--log_csv_file/--no_log_csv', '-l/ ', default=False)
@click.option('--out_dir',  default=f'./search_{datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")}')
def run(filename, hyperparam_filename, set_random_seed, n_jobs, from_day, until_day, use_dates, use_config_days,
        run_n_times, first_n_zeros, fit_column, fit_data, return_func, log_csv_file, out_dir):

    random_seed = 42 if set_random_seed else random.randint(0, 10000)

    cf = ConfigFile()
    cf.load(filename)

    print(f"Output directory for results: {out_dir}")
    print(f"Running with n_jobs == {n_jobs}.")

    def search_func():
        try:
            data_col = int(fit_column)
        except ValueError:
            data_col = fit_column

        gold_data = load_gold_data(fit_data, first_n_zeros=first_n_zeros, data_column=data_col, use_dates=use_dates,
                                   from_day=from_day, until_day=until_day)

        # infer start day and experiment length from gold data
        start_day = None if use_config_days else int(gold_data.iloc[0]["day"])
        n_days = None if use_config_days else len(gold_data)

        gold_data = gold_data["infected"].to_numpy()

        if not os.path.exists(out_dir):
            os.mkdir(out_dir)

        results = run_hyperparam_search(
            filename,
            hyperparam_filename,
            model_random_seed=random_seed,
            n_jobs=n_jobs,
            start_day=start_day,
            n_days=n_days,
            return_func=return_func,
            return_func_kwargs={"y_true": gold_data, "fit_column": fit_column},
            run_n_times=run_n_times,
            output_file=out_dir + '/evo_log.csv' if log_csv_file else None
        )

        if not isinstance(results, list):
            results = [results]

        res_list = []
        for res in results:
            mean_val = np.mean(res['result'])

            res_row = {**res["hyperparams"], f"mean_{return_func}": mean_val}
            res_list.append(res_row)

        fit_name = os.path.split(fit_data)[1].split('.')[0]
        search_method = os.path.split(hyperparam_filename)[1].split('.')[0]

        df = pd.DataFrame(data=res_list)
        df.to_csv(os.path.join(out_dir, f'{search_method}_{return_func}_{fit_name}_seed={random_seed}.csv'))

    print(timeit.timeit(search_func, number=1))


if __name__ == "__main__":
    run()
