# Model M
Covid-19 epidemic model on realistic social network

### Authors:
**Luděk Berec**, *Centre for Modelling of Biological and Social Processes, Centre for Mathematical Biology, Institute of Mathematics, Faculty of Science, University of South Bohemia and  Czech  Academy  of  Sciences,  Biology  Centre,  Institute  of  Entomology* <br>
**Tomáš Diviák**, *Centre for Modelling of Biological and Social Processes and Department of Criminology and Mitchell Centre for Social Network Analysis, School of Social Sciences, University of Manchester* <br> 
**Aleš Kuběna**, *Centre for Modelling of Biological and Social Processes and The Czech Academy of Sciences, Institute of Information Theory and Automation* <br>
**René Levinský**, *Centre for Modelling of Biological and Social Processes and CERGE-EI* <br>
**Roman Neruda**, *Centre for Modelling of Biological and Social Processes and The Czech Academy of Sciences, Institute of Computer Science* <br>
**Gabriela Suchopárová**, *Centre for Modelling of Biological and Social Processes and The Czech Academy of Sciences, Institute of Computer Science* <br>
**Josef Šlerka**, *Centre for Modelling of Biological and Social Processes and New Media Studies, Faculty of Arts, Charles University* <br>
**Martin Šmíd**, *Centre for Modelling of Biological and Social Processes and The Czech Academy of Sciences, Institute of Information Theory and Automation* <br> 
**Jan Trnka**, *Centre for Modelling of Biological and Social Processes and Department of Biochemistry, Cell and Molecular Biology, Third Faculty of Medicine, Charles University* <br> 
**Vít Tuček**, *Centre for Modelling of Biological and Social Processes and Department of Mathematics, University of Zagreb* <br> 
**Petra Vidnerová**, *Centre for Modelling of Biological and Social Processes and The Czech Academy of Sciences, Institute of Computer Science* <br> 
**Karel Vrbenský**, *Centre for Modelling of Biological and Social Processes and The Czech Academy of Sciences, Institute of Information Theory and Automation* <br> 
**Milan Zajíček**, *Centre for Modelling of Biological and Social Processes and The Czech Academy of Sciences, Institute of Information Theory and Automation* <br>
**František Zapletal**, *The Czech Academy of Sciences, Institute of Information Theory and Automation* <br> 

### Acknowledgement: 
The work has been supported by the "City for People, Not for Virus" project No. TL04000282 of the Technology Agency of the Czech Republic.

### How to cite:

If you would like to refer to this software in a publication, please cite the following paper preprint on medrXiv: 

Model-M: An agent-based epidemic model of a middle-sized municipality <br>
Ludek Berec, Tomas Diviak, Ales Kubena, Rene Levinsky, Roman Neruda, Gabriela Suchoparova, Josef Slerka, 
Martin Smid, Jan Trnka, Vit Tucek, Petra Vidnerova, Milan Zajicek, Frantisek Zapletal <br>
medRxiv 2021.05.13.21257139; doi: https://doi.org/10.1101/2021.05.13.21257139

```
@article {Berec2021.05.13.21257139,
	author = {Berec, Ludek and Diviak, Tomas and Kubena, Ales and Levinsky, Rene and Neruda, Roman 
  and Suchoparova, Gabriela and Slerka, Josef and Smid,  Martin and Trnka, Jan and Tucek, Vit 
  and Vidnerova, Petra and Zajicek, Milan and Zapletal, Frantisek},
	title = {Model-M: An agent-based epidemic model of a middle-sized municipality},
	elocation-id = {2021.05.13.21257139},
	year = {2021},
	doi = {10.1101/2021.05.13.21257139},
	publisher = {Cold Spring Harbor Laboratory Press},
	URL = {https://www.medrxiv.org/content/10.1101/2021.05.13.21257139v1},
	eprint = {https://www.medrxiv.org/content/10.1101/2021.05.13.21257139v1.full.pdf},
	journal = {medRxiv}
}

```

## Requirements 

Install them:

```console
$ python3 -m venv myenv
$ source myenv/bin/activate
$ pip install -r requirements.txt
```

Find our packages:

```console
$ cd scripts
$ export PYTHONPATH=../src/
```

## Usage 

1. Loading a graph from CSV files takes minutes. Load the graph once and save it to a pickle file.  
```
Usage: preload_graph.py [OPTIONS] [FILENAME] [OUTPUTNAME]

  Load the graph and pickle.

$ python preload_graph.py ../config/load.ini papertown.pickle
```

2. Run your experiment. 
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

$ python run_experiment.py -r ../config/example.ini 
``` 
After the run finishes, you should find the file history.csv or history_test_id.csv, if test_id was provided.

If you run your simulation on a cluster in parallel, use:
```
Usage: run_all.py [OPTIONS] [FILENAME] [TEST_ID]

  Run the simulations in parallel.

  FILENAME   name of the config file with the setup of the experiments
  TEST_ID    tag to append to output file names

Options:
  --const-random-seed / -r, --no-random-seed
  -R, --user-random-seeds TEXT    File with user defined random seeds.
  --print_interval INTEGER        How often print short info, defaults to
                                  daily.
  --n_repeat INTEGER              Total number of simulations.
  --n_jobs INTEGER                Number of workers.
  --log_level TEXT                Logging level.
  --help                          Show this message and exit.
  
$ python run_all.py -R ../config/random_seeds.txt --n_repeat=100 --n_jobs=20 ../config/example.ini
$ zip history.zip history__*.csv
```


3. Example of INI file:
(**warning:** this is an example only, the model parameters are not final)
```
[TASK]
duration_in_days = 300
print_interval = 1
verbose = Yes
model = SimulationDrivenModel
save_node_states = No

[GRAPH]
type = pickle
file = papertown.pickle

[POLICY]
filename = customised_policy 
name = CustomPolicy

[POLICY_SETUP]
layer_changes_filename = ../data/policy_params/wasabi.csv
#policy_calendar_filename = ../data/policy_params/sim_cr.json
beta_factor_filename = ../data/policy_params/beta_factor.csv
face_masks_filename = ../data/policy_params/masks.csv
#theta_filename = ../data/policy_params/tety.csv
#test_rate_filename = ../data/policy_params/test_new2.csv
#init_filename = ../data/policy_params/init_october.json
reduction_coef1=0.25
reduction_coef2=0.75
new_beta=Yes
sub_policies = self_isolation
self_isolation_filename = wee_cold_sim
self_isolation_name = WeeColdPolicy

[MODEL]
start_day = 185
durations_file=../config/duration_probs.json
prob_death_file=../data/prob_death.csv
beta=1.0
beta_reduction=1.0
beta_in_family=1.0
beta_A=0.50000000000000000000
beta_A_in_family=0.50000000000000000000
mu=0.85
p=0.00
theta_E=0.0
theta_Ia=0.0
#theta_Is=0.76                                                                                                                           
theta_Is=0.2
theta_In=0.0
test_rate=0.65
psi_E=1
psi_Ia=1
psi_Is=1
psi_In=1
q=0
false_symptoms_rate=0.0002999550045
false_symptoms_recovery_rate=0.166247081924819
asymptomatic_rate=0.179
init_E=80
init_I_n = 0
init_I_a = 0
init_I_s = 0
init_R = 215
init_D = 2
```

4. Plot the results

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

$ python plot_experiments.py history.csv
$ python plot_experiments.py history.zip
```
Examples:
- plot 3 files with shades representing the bootstrapped standard deviation and
  the lines means
```
$ python plot_experiments.py data_1.feather data_2.csv data_3.zip --use_sd --use_mean
```
- plot 1 file and the fit data, shades are by default interquantile range and the
  lines are medians. Plot a legend with labels.
```
$ python plot_experiments.py data_1.feather --fit_me fit_data.csv \
     --label_names 'Model output,Fit data'
```

- plot 1 file, modify the plot design, use string labels for the x axis, save to a custom location
```
$ python plot_experiments.py spring.feather --out_file oo.png --title 'Epidemic data in spring' \
    --day_indices 5,36,66 --day_labels 'March 1,April 1,May 1' --xlabel Date \
    --ylabel 'Detected cases' --ymax 10000 --figsize 10 20 --column I_d
```

5. Run hyper-parameter search:
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
  --run_n_times INTEGER           Number of times to run the experiment with
                                  specific hyperparameter settings.
  --first_n_zeros INTEGER         Shifts gold data by this value - the first
                                  day is incremented by `first_n_zeros`, and
                                  the data is padded with `first_n_zeros` days
                                  with the gold values set to zero.
  --data_column TEXT              The index of the column to use from the gold
                                  data DataFrame - can be both int or string
                                  column name.
  --return_func TEXT              Loss function.
  --fit_data TEXT                 A DataFrame that has a column named 'datum'
                                  and contains the gold data in the column
                                  `data_column`.
  -l, --log_csv_file / --no_log_csv
  --out_dir TEXT
  --help                          Show this message and exit.
```
Examples of the hyperparam config file are in the folder 
`../config/hyperparam_search/*.json`. Supported search methods are grid-search and
CMA-ES.
