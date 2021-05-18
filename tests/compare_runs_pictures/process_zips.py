import click
import glob
import os
import pandas as pd
import zipfile


def process_zip(zip_path, tmp_dir, save_dir):
    if os.path.exists(save_dir):
        print(f"Skipping {save_dir}")
        return

    # temp dir
    if os.path.exists(tmp_dir):
        raise ValueError("Tmp dir for zip files already exists")

    os.mkdir(tmp_dir)
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(tmp_dir)

    csv_files = [file for file in glob.glob(os.path.join(tmp_dir, '*.csv'))]

    dfs = [process_csv(cf) for cf in csv_files]
    os.mkdir(save_dir)
    for df, csv_f in zip(dfs, csv_files):
        csv_f = os.path.basename(csv_f)
        csv_f = os.path.join(save_dir, csv_f)
        df.to_csv(csv_f, sep=',', index=False)

    # clean temp dir
    for file in csv_files:
        os.remove(file)
    os.rmdir(tmp_dir)


def process_csv(csv_file):
    df_raw = pd.read_csv(csv_file, comment='#')

    df = df_raw.copy()
    df["I_d"] = df[["I_dn", "I_da", "I_ds", "E_d", "J_ds", "J_dn"]].sum(axis=1)

    df["all_infected"] = df[["I_n", "I_a", "I_s", "E", "I_dn",
                             "I_da", "I_ds", "E_d", "J_ds", "J_dn", "J_n", "J_s"]].sum(axis=1)

    df = df[["T", "I_d", "all_infected"]]
    return df


@click.command()
@click.argument('zip_folder')
@click.option('--tmp_dir', default='./tmp/')
@click.option('--out_dir', default='./csv_out/')
def run(zip_folder, tmp_dir, out_dir):
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    for zip_file in glob.glob(os.path.join(zip_folder, '*.zip')):
        print(f"Processing zip file: {zip_file}")
        save_path = os.path.join(out_dir, os.path.basename(zip_file).replace('.zip', ''))
        process_zip(zip_file, tmp_dir, save_path)

    print("Done")


if __name__ == "__main__":
    run()
