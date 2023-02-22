import click
import numpy as np
import pandas as pd


def fix_column_name(name_tuple):
    if not 'kumulativni' in name_tuple:
        return name_tuple

    name_tuple = eval(name_tuple)

    stats, okres = name_tuple
    okres = okres.replace(" ", "_")
    return '_'.join([stats, okres])


@click.command()
@click.argument('out_path', default="okres-nakazeni-vyleceni-umrti.csv")
@click.option('--path', default='https://onemocneni-aktualne.mzcr.cz/api/v2/covid-19/kraj-okres-nakazeni-vyleceni-umrti.csv')
#@click.option('--tests_path', default='https://onemocneni-aktualne.mzcr.cz/api/v2/covid-19/testy.csv')
@click.option('--round_func', default='none')
@click.option('--scale', default=56102)
def run(out_path, path, round_func, scale):
    codes = pd.read_csv('kody.csv', delimiter=';')
    population = pd.read_csv('population.csv', delimiter=';')

    # Changed "Praha" in codes to "Hlavní město Praha" (manually)
    not_in_both = codes[~codes["ZKRTEXT"].isin(population["kraj/okres"])]["CHODNOTA"]

    if len(not_in_both):
        assert not_in_both[0] == "CZZZZZ"

    code_dict = {key: name for key, name in zip(codes["CHODNOTA"], codes["ZKRTEXT"])}

    # testy = pd.read_csv(tests_path)  # TODO testy zatim nejsou per okres
    covid_stats_raw = pd.read_csv(path)

    # Names from codes
    okresy_names = covid_stats_raw['okres_lau_kod'].apply(lambda code: code_dict[code])
    covid_stats = covid_stats_raw.drop(columns=['kraj_nuts_kod', 'okres_lau_kod'])
    covid_stats.insert(1, 'okres', okresy_names)

    # I_d
    infected = 'kumulativni_pocet_nakazenych'
    cured = 'kumulativni_pocet_vylecenych'
    dead = 'kumulativni_pocet_umrti'

    i_d = covid_stats[infected] - covid_stats[cured] - covid_stats[dead]
    covid_stats.insert(2, 'pocet_I_d', i_d)
    covid_stats.drop(columns=['kumulativni_pocet_nakazenych'], inplace=True)

    # Number of tests
    # covid_stats_t = covid_stats.merge(testy, how="left", left_on=["datum", "okres"], right_on=["datum", okres])
    # assert len(covid_stats) == len(covid_stats_t)
    # covid_stats = covid_stats_t

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

    columns_to_scale = ['kumulativni_pocet_I_d', 'kumulativni_pocet_vylecenych', 'kumulativni_pocet_umrti']
                        #'kumulativni_pocet_testu']  # TODO testy?
    for column in columns_to_scale:
        covid_stats[f"{column}_prepocitano"] = round_func(covid_stats[column] / population_stats * scale).astype(float)

    # Move "okres" to columns
    pivot = covid_stats.pivot(index="datum", columns="okres",
                              values=[c for c in covid_stats.columns if c != "datum" and c != "okres"])

    result = pd.DataFrame(pivot.to_records())
    result.columns = [fix_column_name(c) for c in result.columns]

    #    selected_columns = ["datum"] + [c for c in result.columns if "Rakovník" in c] 
    #    result = result[selected_columns] 

    result.to_csv(out_path, index=False)


if __name__ == "__main__":
    run()
