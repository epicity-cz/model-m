import click
import numpy as np
import os
import pandas as pd


def fix_column_name(name_tuple):
    if not 'kumulativni' in name_tuple:
        return name_tuple

    name_tuple = eval(name_tuple)

    stats, okres = name_tuple
    okres = okres.replace(" ", "_")
    return '_'.join([stats, okres])


@click.command()
@click.argument('out_dir', default=".")
@click.option('--path',
              default='https://onemocneni-aktualne.mzcr.cz/api/v2/covid-19/kraj-okres-nakazeni-vyleceni-umrti.csv',
              help='Path to source data.')
#@click.option('--tests_path', default='https://onemocneni-aktualne.mzcr.cz/api/v2/covid-19/testy.csv')
@click.option('--round_func', default='none', help='Round scaled values using this function.'
                                                   ' Supported functions - [none, ceil, floor]')
@click.option('--scale', default=56102, help='Scale each region using this value.')
@click.option('--clip_negatives/--no_clip', default=True, help='Correct errors in source data - '
                                                               'replace negative I_d values with 0 (default True).')
def run(out_dir, path, round_func, scale, clip_negatives):
    """ Download source data (epidemic statistics in Czechia), preprocess them and save data for each region in
        a separate csv file. Each file starts with the date 2020-02-26.

        A new column is also computed for every region - pocet_I_d (English: I_d count), the column represents
        new detected cases per day. Computed from source columns: (infected - cured - dead), all three columns
        are cumulative. Possible negative values are clipped to zeros (turn off with --no_clip).

        Additionally, scale every data field according to the `scale` parameter and optionally round it:
        res = round_func( field / scale ). Every such column is saved with name 'ORIGINAL_NAME'_preprocitano .

               \b
               OUT_DIR   directory where to save the csv files
    """

    codes = pd.read_csv('kody.csv', delimiter=';')
    population = pd.read_csv('population.csv', delimiter=';')

    # Changed "Praha" in codes to "Hlavní město Praha" (manually)
    not_in_both = codes[~codes["ZKRTEXT"].isin(population["kraj/okres"])]["CHODNOTA"]

    if len(not_in_both):
        assert not_in_both[0] == "CZZZZZ"

    code_dict = {key: name for key, name in zip(codes["CHODNOTA"], codes["ZKRTEXT"])}

    # testy = pd.read_csv(tests_path)  #  tests are not per region yet
    covid_stats_raw = pd.read_csv(path)
    covid_stats_raw = covid_stats_raw[~covid_stats_raw["okres_lau_kod"].isna()]

    # Names from codes
    okresy_names = covid_stats_raw['okres_lau_kod'].apply(lambda code: code_dict[code])
    covid_stats = covid_stats_raw.drop(columns=['kraj_nuts_kod', 'okres_lau_kod'])
    covid_stats.insert(1, 'okres', okresy_names)

    # I_d
    infected = 'kumulativni_pocet_nakazenych'
    cured = 'kumulativni_pocet_vylecenych'
    dead = 'kumulativni_pocet_umrti'

    i_d = covid_stats[infected] - covid_stats[cured] - covid_stats[dead]
    if clip_negatives:
        i_d = i_d.clip(lower=0)

    covid_stats.insert(2, 'pocet_I_d', i_d)

    # Number of tests
    # covid_stats_t = covid_stats.merge(testy, how="left", left_on=["datum", "okres"], right_on=["datum", okres])
    # assert len(covid_stats) == len(covid_stats_t)
    # covid_stats = covid_stats_

    # Scale to our town size
    if round_func == 'ceil':
        round_func = np.ceil
    elif round_func == 'floor':
        round_func = np.floor
    elif round_func == 'none':
        round_func = lambda x: x 
    else:
        raise ValueError("Invalid round function string.")

    pop_dict = {key: int(val.replace(" ", "")) for key, val in
                zip(population["kraj/okres"], population["počet obyvatel"])}

    population_stats = covid_stats["okres"].apply(lambda val: pop_dict[val])

    columns_to_scale = ['pocet_I_d',
                        infected, cured, dead]
                        # 'kumulativni_pocet_testu']  # tests are not per region yet
    for column in columns_to_scale:
        covid_stats[f"{column}_prepocitano"] = round_func(covid_stats[column] / population_stats * scale).astype(float)

    # Move "okres" to columns
    pivot = covid_stats.pivot(index="datum", columns="okres",
                              values=[c for c in covid_stats.columns if c != "datum" and c != "okres"])

    result = pd.DataFrame(pivot.to_records())
    result.columns = [fix_column_name(c) for c in result.columns]

    okresy_out_dir = os.path.join(out_dir, 'okresy')
    if not os.path.exists(okresy_out_dir):
        os.mkdir(okresy_out_dir)

    okresy_dataframes = {}

    # fit dicts
    for okres in okresy_names.unique():
        print(okres)

        columns_prepocitano = [f'{c}_prepocitano' for c in columns_to_scale]
        okres_stats = {k: pivot[k][okres] for k in [*columns_to_scale, *columns_prepocitano]}
        okres_df = pd.DataFrame(okres_stats).reset_index()

        # add counts per day
        for kum_count in okres_df.columns:
            if not kum_count.startswith('kumulativni_'):
                continue

            shifted = pd.Series([0]).append(okres_df[kum_count][:-1])
            new_name = kum_count.replace('kumulativni_', '')
            okres_df[new_name] = okres_df[kum_count].to_numpy() - shifted.to_numpy()

        # start with 2020-02-26
        pad_df = pd.DataFrame(
            {'datum': pd.date_range('26-02-2020', okres_df.iloc[0, 0])[:-1].to_series()},
            columns=okres_df.columns
        )
        pad_df.reset_index(inplace=True, drop=True)
        pad_df.fillna(0.0, inplace=True)

        okres_df["datum"] = pd.to_datetime(okres_df["datum"])
        okres_df = pd.concat([pad_df, okres_df])

        okres_df.to_csv(os.path.join(okresy_out_dir, f'{okres}.csv'))

        okresy_dataframes[okres] = okres_df

    result.to_csv(os.path.join(out_dir, 'okres-nakazeni-vyleceni-umrti.csv'), index=False)


if __name__ == "__main__":
    run()
