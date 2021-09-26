import click
import pandas as pd


@click.command()
@click.argument('out_path', default="fit_me.csv")
@click.argument('in_path', default="okresy_aggregated.csv")
def run(out_path, in_path):
    """ Extract data from IN_PATH for the purpose of model fitting.
        The data will be saved in a csv file saved in OUT_PATH.

        Source data must contain the following columns:
            ["datum", "pocet_I_d_prepocitano", "pocet_nakazenych_prepocitano",
             "kumulativni_pocet_nakazenych_prepocitano", "kumulativni_pocet_umrti_prepocitano",
             "pocet_umrti_prepocitano"]

            (column names are the same as in the original Czech source data;
             in English:
             date, I_d count, infected count, cumulative infected count, cumulative
             deaths, death count)

        \b
        OUT_PATH   path where to save the csv file
        IN_PATH    path to the source data
    """

    df = pd.read_csv(in_path, index_col=0)

    df = df[["datum", "pocet_I_d_prepocitano", "pocet_nakazenych_prepocitano",
             "kumulativni_pocet_nakazenych_prepocitano", "kumulativni_pocet_umrti_prepocitano",
             "pocet_umrti_prepocitano"]]

    df.rename(columns={
        "pocet_I_d_prepocitano": "I_d",
        "pocet_nakazenych_prepocitano": "inc_I_d",
        "kumulativni_pocet_nakazenych_prepocitano": "cum_I_d",
        "kumulativni_pocet_umrti_prepocitano": "D",
        "pocet_umrti_prepocitano": "inc_D"
    }, inplace=True)
    df["T"] = df.index

    df.to_csv(out_path, index=False)


if __name__ == "__main__":
    run()
