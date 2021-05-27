import click
import datetime
import numpy as np
import os
import pandas as pd
import random
import timeit

from config_utils import ConfigFile
from hyperparam_search.hyperparam_utils import run_hyperparam_search


def load_gold_data(csv_path, first_n_zeros=0, data_column=2):
    df = pd.read_csv(csv_path)
    dates = pd.to_datetime(df["datum"])
    dates = dates - dates[0]
    dates = dates.apply(lambda t: t.days + 1)

    result = pd.DataFrame({"day": range(1, dates.max() + 1), "infected": pd.NA})

    if isinstance(data_column, int):
        data_vals = df.iloc[:, data_column].to_list()
    else:
        data_vals = df[data_column].to_list()

    result.loc[result["day"].isin(dates), "infected"] = data_vals

    result.fillna(method='ffill', inplace=True)

    if first_n_zeros > 0:
        result["day"] += first_n_zeros

        first_n = [{"day": i + 1, "infected": 0} for i in range(first_n_zeros)]
        result = pd.concat([pd.DataFrame(first_n), result], ignore_index=True)

    return result


@click.command()
@click.option('--set-random-seed/--no-random-seed', ' /-r', default=True)
@click.option('--policy', '-p', default=None)
@click.option('--n_jobs', default=1)
@click.option('--run_n_times', default=1)
@click.option('--first_n_zeros', default=5)
@click.option('--data_column', default='2')
@click.option('--return_func', default='rmse')
@click.option('--fit_data', default='../data/litovel.csv')
@click.option('--log_csv_file/--no_log_csv','-l/ ', default=False)
@click.option('--out_dir',  default=f'./search_{datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")}')
@click.argument('filename', default="town0.ini")
@click.argument('hyperparam_filename', default="example_gridsearch.json")
def run(set_random_seed, policy, n_jobs, run_n_times, data_column, first_n_zeros, return_func, fit_data, out_dir,
        log_csv_file, filename, hyperparam_filename):

    random_seed = 42 if set_random_seed else random.randint(0, 10000)

    cf = ConfigFile()
    cf.load(filename)

    print(f"Output directory for results: {out_dir}")
    print(f"Running with n_jobs == {n_jobs}.")

    def search_func():
        try:
            data_col = int(data_column)
        except ValueError:
            data_col = data_column

        gold_data = load_gold_data(fit_data, first_n_zeros=first_n_zeros, data_column=data_col)["infected"].to_numpy()

        if not os.path.exists(out_dir):
            os.mkdir(out_dir)

        results = run_hyperparam_search(
            filename,
            hyperparam_filename,
            model_random_seed=random_seed,
            use_policy=policy,
            n_jobs=n_jobs,
            n_days=len(gold_data),
            return_func=return_func,
            return_func_kwargs={"y_true": gold_data},
            run_n_times=run_n_times,
            output_file=out_dir + '/evo_log.csv' if log_csv_file else None
        )

        # TODO better solution later
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
