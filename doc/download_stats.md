# Download gold data
In this section we describe how to download the 'COVID-19 epidemy in czech
regions' dataset. The downloaded file can be used for hyper-parameter fitting.

To download data and create a file for fitting in one step, use the following bash
script (OUT_DIR is the path where all data will be saved).

```console
bash get_all_data.sh OUT_DIR
```

The final file is saved in ``OUT_DIR/fit_me.csv``

The python scripts called in the bash script are described in the following sections.

## 1. Download source data
Download data from mzcr.cz, preprocess column names, separate into regions,
shift the data so that every file starts with ``2020-02-26``.

```
Usage: download_from_mzcr.py [OPTIONS] [OUT_DIR]

  Download source data (epidemic statistics in Czechia), preprocess them and
  save data for each region in a separate csv file. Each file starts with the
  date 2020-02-26.

  A new column is also computed for every region - pocet_I_d (English: I_d
  count), the column represents new detected cases per day. Computed from
  source columns: (infected - cured - dead), all three columns are cumulative.
  Possible negative values are clipped to zeros (turn off with --no_clip).

  Additionally, scale every data field according to the `scale` parameter and
  optionally round it: res = round_func( field / scale ). Every such column is
  saved with name 'ORIGINAL_NAME'_preprocitano .

                OUT_DIR   directory where to save the csv files

Options:
  --path TEXT        Path to source data.
  --round_func TEXT  Round scaled values using this function. Supported
                     functions - [none, ceil, floor]
  --scale INTEGER    Scale each region using this value.
  --clip_negatives / --no_clip  Correct errors in source data - replace
                                negative I_d values with 0 (default True).
  --help             Show this message and exit.
```

## 2. Aggregate regions into one file
Create file with mean/median data from all regions.

```
Usage: aggregate_stats.py [OPTIONS] [OUT_PATH]

  Aggregate several .csv files into one file. Supported methods are mean and
  median. Each .csv file represents epidemic statistic from one Czech region.

          OUT_PATH   path where to save the aggregated csv file

Options:
  --in_dir TEXT     Directory where the data per region are stored.
  --exclude TEXT    Region files to exclude.
  --aggregate TEXT  Aggregation function - mean or median (default mean).
  --help            Show this message and exit.
```

## 3. Create fit file
Process the aggregated file - add columns necessary for fitting.

```
Usage: create_fit_me.py [OPTIONS] [OUT_PATH] [IN_PATH]

  Extract data from IN_PATH for the purpose of model fitting. The data will be
  saved in a csv file saved in OUT_PATH.

  Source data must contain the following columns:
     ["datum", "pocet_I_d_prepocitano", "pocet_nakazenych_prepocitano",
      "kumulativni_pocet_nakazenych_prepocitano", "kumulativni_pocet_umrti_prepocitano",
      "pocet_umrti_prepocitano"]

     (column names are the same as in the original Czech source data;      in
      English:      date, I_d count, infected count, cumulative infected
      count, cumulative      deaths, death count)

  OUT_PATH   path where to save the csv file
  IN_PATH    path to the source data

Options:
  --help  Show this message and exit.
```
