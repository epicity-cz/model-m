# How to Run Simulations

### 1. [OPTIONAL] Graph Preloading

Loading a graph from CSV files takes minutes. Therefore it is loaded once, as you run the first simulation, and saved as
a pickle file. Another option is to prepare your pickle file in advance using `preload_graph.py`:

```console
python preload_graph.py ../config/hodoninsko.ini 
```

In the INI file, the path to CSV files of your graph and the name with path of the resulting pickle file are specified (
see [INI file](inifile.md#graph)).

```
Usage: preload_graph.py [OPTIONS] [FILENAME]

  Load the graph and pickle.

Options:
  --help  Show this message and exit.
```

### 2. Running Your Experiments

Run your experiment.

+ If you wish to run one run of the model only, use `run_experiment.py`:

```console
python run_experiment.py -r ../config/hodoninsko.ini 
``` 

or

```console
python run_experiment.py -r ../config/hodoninsko.ini my_experiment
``` 

After the run finishes, you should find the file `history.csv` or `history_my_experiment.csv`. The path to the output
directory can be specified in your [INI file](inifile.md#task) as `output_dir`. The default is the current
directory, `data/output/model` directory is intended for this purpose.

By default, the experiment is run always with the same random seed. To disable this, use otpion `-r` or provide your
random seed using `-R`.

```
Usage: run_experiment.py [OPTIONS] [FILENAME] [TEST_ID]

  Run the simulation specified in FILENAME.

  FILENAME   name of the config file with the setup of the experiment
  TEST_ID    tag to append to output files names

Options:
  --const-random-seed / -r, --no-random-seed
  -R, --user_random_seed TEXT     User defined random seed number.
  --print_interval INTEGER        How often print short info, defaults to
                                  daily.
  --n_repeat INTEGER              Total number of simulations.
  --log_level TEXT                Logging level.
  --help                          Show this message and exit.
```

+ For a proper experiment, you should evaluate the model more times. You can do it in parallel using:

```console
python run_multi_experiment.py -R ../config/random_seeds.txt --n_repeat=100 --n_jobs=4 ../config/hodoninsko.ini my_experiment
```

By default it produces a ZIP file with the resulting history files. You can change output_type to FEATHER and the result
will be stored as one data frame in the feather format.

In this case, you can provide the whole list of random seeds. Use option `-R random_seeds.txt`, where `random_seeds.txt`
is a file containing one random seed per line.

```
Usage: run_multi_experiment.py [OPTIONS] [FILENAME] [TEST_ID]

  Run the demo test inside the timeit

Options:
  --const-random-seed / -r, --no-random-seed
  -R, --user-random-seeds TEXT
  --print_interval INTEGER
  --n_repeat INTEGER
  --n_jobs INTEGER  
  --log_level TEXT                Logging level.
  --output_type TEXT              ZIP or FEATHER. ZIP produces `n_repeat`
                                  output csv files in one zip archive. FEATHER
                                  produces one big data frame in feather file.
  --help                          Show this message and exit.
``` 

### 3. Format of the Resulting Data Frame

The result of a simulation is one dataframe saved as a CSV file. If you use the `--output_type FEATHER`, the data frame
of all simulations are concatenated into one long dataframe with one extra column containing simulation id. The CSV file
starts with a comment section containing the whole config file and the random seed used (if you use `pd.read_csv` to
process it, use `comment="#"`).

DataFrame example:

|T|S|S_s|E|I_n|I_a|I_s|J_s|J_n|R|D|EXT|inc_S|inc_S_s|inc_E|inc_I_n|inc_I_a|inc_I_s|inc_J_s|inc_J_n|inc_R|inc_D|inc_EXT|day|all_tests|positive_tests|negative_tests|I_d|cum_I_d|day_policy|positive_enter_test|contacts_collected|
|--|--|--|--|--|--|--|--|--|--|--|--|--|--|--|--|--|--|--|--|--|--|--|--|--|--|--|--|--|--|--|--|
|...|...|...|...|...|...|...|...|...|...|...|...|...|...|...|...|...|...|...|...|...|...|...|...|...|...|...|...|...|...|...|...|
|10|4547|0|195|10|45|19|22|9|215|3|1|0|0|60|2|21|7|11|5|0|0|0|10|4|3|1|12|12|10|0|15|
|11|4486|0|241|11|46|24|28|9|217|3|1|0|0|61|2|13|12|7|1|2|0|0|11|18|6|12|18|18|11|2|25|
|12|4440|0|264|12|54|25|38|10|219|3|1|0|0|46|3|20|12|11|2|2|0|0|12|11|6|5|24|24|12|2|29|
|13|4392|0|275|17|68|35|44|10|221|3|1|0|0|48|6|31|17|7|1|2|0|0|13|14|10|4|34|34|13|5|56|
|14|4354|0|277|20|85|38|52|12|224|3|1|0|0|38|5|31|14|11|2|3|0|0|14|32|16|16|50|50|14|11|74|
|15|4308|0|296|19|88|47|60|15|229|3|1|0|0|46|2|25|22|13|3|5|0|0|15|30|15|15|65|65|15|8|76|
|16|4255|0|310|30|82|61|71|14|239|3|1|0|0|53|12|27|33|19|1|10|0|0|16|47|20|27|83|84|16|13|57|
|17|4203|0|324|33|91|66|79|14|252|3|1|0|0|52|6|32|23|18|3|13|0|0|17|81|26|55|108|110|17|16|122|
|18|4157|0|326|31|103|71|98|20|256|3|1|0|0|46|5|39|27|22|7|4|0|0|18|81|24|57|130|132|18|13|84|
|19|4105|0|333|36|103|86|113|24|262|3|1|0|0|52|9|36|36|21|4|6|0|0|19|76|28|48|152|158|19|17|71|
|20|4049|0|356|40|97|88|135|24|273|3|1|0|0|56|6|27|33|31|2|11|0|0|20|117|34|83|178|188|20|14|77|

Column **T** contains the number of the day and the corresponding row contains statistics for this day. Columns **S**
to **EXT** contain the number of nodes in the given state, columns **inc_S** to **inc_EXT** contain increments, i.e.
nodes that were moved to the given state this date. Column **day** is the same as **T** and is left for backward
compatibility. The rest of columns come from a policy, in this case CRLikePolicy. It contains the number of all tests
done, positive and negative tests, **I_d** number of detected individuals (active cases), **cum_I_d** cummulative number
of detected individuals, number of enter tests performed (test when a node is traced), number of contacts collected, and
day of policy run (should again be the same as **T**).

### 4. INI File

Each experiment is defined by an INI file. See [INI file](inifile.md) for the list of possible parameters and their
meaning and [hodoninsko.ini](../config/hodoninsko.ini) for example of such a file.

### 5. Visualisation

Plot the results.

```console
python plot_experiments.py ../data/output/model/history_my_experiment.zip
```

The script `plot_experiment.py` enables you to plot a selected column of the resulting dataframe. It has a wide
functionality, see the examples below.

```
Usage: plot_experiments.py [OPTIONS] [PLOT_FILES]...

  Create plot using an arbitrary number of input files. Optionally, plot a fit
  curve specified by --fit_me.

  PLOT_FILES   name of the input files - either .csv, .zip or .feather

Options:
  --label_names TEXT              Labels for each file to show in legend
                                  (separated by comma), called as
                                  --label_names l_1,l_2, ... If --fit_me is
                                  used, the last label in this argument is the
                                  label of the fit data.
  --out_file TEXT                 Path where to save the plot.
  --fit_me TEXT                   Path to a .csv file with fit data.
  --show_whole_fit / --show_partial_fit
                                  If true, plot the whole fit data on the
                                  xaxis (may be longer than other data).
  --title TEXT                    Title of the plot.
  --column TEXT                   Column to use for plotting.
  --zip_to_feather / --no_zip_to_feather
                                  If True, save processed .zips to .feather.
                                  The file name is the same except the
                                  extension.
  --figsize INTEGER...            Size of the plot (specified without commas:
                                  --figsize 6 5).
  --ymax INTEGER                  Y axis upper limit. By default the limit is
                                  inferred from sourcedata.
  --xlabel TEXT                   X axis label.
  --ylabel TEXT                   Y axis label.
  --use_median / --use_mean       Use median or mean in the plot (default median).
  --use_sd / --use_iqr            Use sd or iqr for shades (default iqr).
  --day_indices TEXT              Use dates on x axis - maps indices to labels
                                  (e.g. 5,36,66).
  --day_labels TEXT               Use dates on x axis - string labels (e.g.
                                  "March 1,April 1,May 1").
  --help                          Show this message and exit.
```

Examples:

- plot 3 files with shades representing the bootstrapped standard deviation and the lines means

```console
python plot_experiments.py data_1.feather data_2.csv data_3.zip --use_sd --use_mean
```

- plot 1 file and the fit data, shades are by default interquantile range and the lines are medians. Plot a legend with
  labels.

```console
python plot_experiments.py data_1.feather --column I_d --fit_me fit_data.csv \
     --label_names 'Model output,Fit data'
```

- plot 1 file, modify the plot design, use string labels for the x axis, save to a custom location

```console
python plot_experiments.py spring.feather --out_file oo.png --title 'Epidemic data in spring' \
    --day_indices 5,36,66 --day_labels 'March 1,April 1,May 1' --xlabel Date \
    --ylabel 'Detected cases' --ymax 10000 --figsize 10 20 --column I_d
```

### 6. Fitting Your Model

We enable model hyper-parameter search in our implementation. You can for example use the data openly accessible from
Czech Ministry of Health [mzcr.cz](https://onemocneni-aktualne.mzcr.cz/api/v2/covid-19). We provide a script that
downloads all necessary data for convenience, [see here](download_stats.md).

To find the values of model parameters, use the following script:

```
Usage: run_search.py [OPTIONS] [FILENAME] [HYPERPARAM_FILENAME]

  Runs hyperparameter search.

  FILENAME File with the model config.
  HYPERPARAM_FILENAME Config file of the hyperparameter search. 

Options:
  --set-random-seed / -r, --no-random-seed
                                  Random seed to set for the experiment. If
                                  `run_n_times` > 1, the seed is incremented
                                  by i for the i-th run.

  --n_jobs INTEGER
  --from_day INTEGER              Lower bound for days in the gold data
                                  DataFrame (indexed from 0).

  --until_day INTEGER             Upper bound for days in the gold data
                                  DataFrame (python-like indexing - elements
                                  up to 'until_day - 1' are selected).

  --use_dates / --use_T           If True, use column 'datum' for gold data
                                  indexing, else use column 'T' (default).

  --use_config_days / --use_args_days
                                  If True, use 'start_day' and 'n_days' from
                                  the config file, otherwise using 'from_day'
                                  and 'until_day' command line arguments.

  --run_n_times INTEGER           Number of times to run th experiment with
                                  specific hyperparameter settings.

  --first_n_zeros INTEGER         Shifts gold data by this value - the first
                                  day is incremented by `first_n_zeros`, and
                                  the data is padded with `first_n_zeros` days
                                  with the gold values set to zero.

  --fit_column TEXT               Data column to use for fit.
  --fit_data TEXT                 A DataFrame that has a column named 'datum'
                                  and contains the gold data in the column
                                  `data_column`.

  --return_func TEXT              Loss function - supported are rmse, mae and r2.
  -l, --log_csv_file / --no_log_csv
  --out_dir TEXT
  --help                          Show this message and exit.
```

Examples of the hyperparam config file are in the folder
`../config/hyperparam_search/*.json`. Supported search methods are grid-search and CMA-ES.

For example, the following command fits the mean `I_d` of 10 model runs to the gold data using CMA-ES and *R<sup>
2</sup>* loss.

```console
python run_search.py --n_jobs 4 --run_n_times 10 --return_func r2 --fit_column I_d \
    --out_dir fit_cmaes_test --log_csv_file --fit_data ../data/fit_data/fit_me.csv \
    ../config/papertown.ini ../config/hyperparam_search/cmaes.json
```

If you want to fit to a smaller interval rather than to the whole gold sequence, pass the arguments `--from_day` and/or
`--until_day` to the script:

```console
python run_search.py --n_jobs 4 --run_n_times 10 --return_func r2 --fit_column I_d \
    --from_day 184 --until_day 195 \
    --out_dir fit_cmaes_test --log_csv_file --fit_data ../data/fit_data/fit_me.csv \
    ../config/papertown.ini ../config/hyperparam_search/cmaes.json
```

We use python-like indexing, so the first row of the gold data has index 0, and the last selected element has
index `until_day - 1`.

