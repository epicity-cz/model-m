# Model M
Covid-19 epidemic model on realistic social network

### Authors:
**Luděk Berec**, *Centre for Modelling of Biological and Social Processes, Centre for Mathematical Biology, Institute of Mathematics, Faculty of Science, University of South Bohemia and  Czech  Academy  of  Sciences,  Biology  Centre,  Institute  of  Entomology* <br>
**Cyril Brom\***, *Faculty of Mathematics and Physics,  Charles University* <br>
**Tomáš Diviák\***, *Centre for Modelling of Biological and Social Processes and Department of Criminology and Mitchell Centre for Social Network Analysis, School of Social Sciences, University of Manchester* <br>
**Jakub Drbohlav\***, *Centre for Modelling of Biological and Social Processes* <br>
**Václav Korbel\***, *CERGE-EI* <br>
**Aleš Kuběna**, *Centre for Modelling of Biological and Social Processes and The Czech Academy of Sciences, Institute of Information Theory and Automation* <br>
**René Levinský\***, *Centre for Modelling of Biological and Social Processes and CERGE-EI* <br>
**Roman Neruda\***, *Centre for Modelling of Biological and Social Processes and The Czech Academy of Sciences, Institute of Computer Science* <br>
**Gabriela Suchopárová\***, *Centre for Modelling of Biological and Social Processes and The Czech Academy of Sciences, Institute of Computer Science* <br>
**Josef Šlerka\***, *Centre for Modelling of Biological and Social Processes and New Media Studies, Faculty of Arts, Charles University* <br>
**Martin Šmíd\***, *Centre for Modelling of Biological and Social Processes and The Czech Academy of Sciences, Institute of Information Theory and Automation* <br> 
**Jan Trnka\***, *Centre for Modelling of Biological and Social Processes and Department of Biochemistry, Cell and Molecular Biology, Third Faculty of Medicine, Charles University* <br> 
**Vít Tuček**, *Centre for Modelling of Biological and Social Processes and Department of Mathematics, University of Zagreb* <br> 
**Petra Vidnerová\***, *Centre for Modelling of Biological and Social Processes and The Czech Academy of Sciences, Institute of Computer Science* <br> 
**Karel Vrbenský**, *Centre for Modelling of Biological and Social Processes and The Czech Academy of Sciences, Institute of Information Theory and Automation* <br> 
**Milan Zajíček**, *Centre for Modelling of Biological and Social Processes and The Czech Academy of Sciences, Institute of Information Theory and Automation* <br>
**František Zapletal**, *The Czech Academy of Sciences, Institute of Information Theory and Automation* <br> 

\* authors that contributed to the schools release and paper.

### Acknowledgement: 
This work was supported by the European Regional Development Fund-Project "Creativity and Adaptability as Conditions of the Success of Europe in an Interrelated World" (No. CZ.02.1.01/0.0/0.0/16019/0000734) and by the  "City for People, Not for Virus" project No. TL04000282 of the Technology Agency of the Czech Republic. We would like to thank Eva Blechová for insightful suggestions. We highly appreciate the help of Helena Patáková with data collection and to Jan Žák with data collection and anonymization.



## Requirements 

Install them:

```console
$ python3 -m venv myenv
$ source myenv/bin/activate
$ pip install -r requirements.txt
```

Find our packages:

```console
$ cd tests
$ export PYTHONPATH=../models:../model_m:../graphs:../policies:../utils:../
```

## Usage 

1. Loading a graph from CSV files takes minutes. Load the graph once and save it to a pickle file.  
```
Usage: preload_graph.py [OPTIONS] [FILENAME] [OUTPUTNAME]

  Load the graph and pickle.

$ python preload_graph.py load.ini papertown.pickle
```

2. Run your experiment. 
```
Usage: run_experiment.py [OPTIONS] [FILENAME] [TEST_ID]

  Run the experiment given by the config file FILENAME.

  FILENAME  config file with the experiment definition  
  TEST_ID   ID is appended to  names of all output files

Options:
  --const-random-seed / -r, --no-random-seed
                                  Use -r to run with different random seed
                                  each time.

  -R, --user_random_seed TEXT     User defined random seed.
  --print_interval INTEGER        How often to print the info message.
  --n_repeat INTEGER              Number of repetition of the experiment.
  --help                          Show this message and exit.

$ python run_experiment.py -r example.ini 
``` 
After the run finishes, you should find the file history.csv or history_test_id.csv, if test_id was provided.

If you run your simulation on a cluster in parallel, use:
```
Usage: run_all.py [OPTIONS] [FILENAME] [TEST_ID]

  Run the experiment given by the config file FILENAME.
  FILENAME  config file with the experiment definition
  TEST_ID   ID is appended to  names of all output files

Options:
  --const-random-seed / -r, --no-random-seed
                                  Use -r to run with different random seed
                                  each time.

  -R, --user-random-seeds TEXT    File with random seeds.
  --print_interval INTEGER        How often to print the info message.
  --n_repeat INTEGER              Number repetition of the experiment.
  --n_jobs INTEGER                Number of processes.
  --help                          Show this message and exit.

$ python run_all.py -R random_seeds.txt --n_repeat=100 --n_jobs=20 example.ini
```


3. Example of INI file:
(**warning:** this is an example only, the model parameters are not final)
```
[TASK]
duration_in_days = 200
print_interval = 1
verbose = Yes
model = SimulationDrivenModel
save_node_states = No

[GRAPH]
name = pickle
file = papertown.pickle

[POLICY]
filename = customised_policy 
name = CustomPolicy

[POLICY_SETUP]
layer_changes_filename = ../data/policy_params/whisky.csv
policy_calendar_filename = ../data/policy_params/sim_cr.json
beta_factor_filename = ../data/policy_params/beta_factor.csv
face_masks_filename = ../data/policy_params/masks.csv
theta_filename = ../data/policy_params/tety.csv
test_rate_filename = ../data/policy_params/test_new2.csv
init_filename = ../data/policy_params/init_new.json
reduction_coef1=0.75
reduction_coef2=0.1
new_beta=Yes


[MODEL]
durations_file=./duration_probs.json
beta=0.8
beta_reduction=1.0
beta_in_family=0.8
beta_A=0.4
beta_A_in_family=0.4
mu=0.00131326350277332
p=0.00
theta_E=0.0
theta_Ia=0.0
#theta_Is=0.76                                                                                                                              
theta_Is=1.0
theta_In=0.0
test_rate=0.8
psi_E=1
psi_Ia=1
psi_Is=1
psi_In=1
q=0
false_symptoms_rate=0.0002999550045
false_symptoms_recovery_rate=0.166247081924819
asymptomatic_rate=0.179
init_E=2
init_I_n = 0
init_I_a = 0
init_I_s = 0
init_I_d = 0

```

4. Run hyper-parameter search:
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

### School experiments: 

+ preload school graph using `load_schools.ini`
```console
$ python preload_graph.py load_schools.ini zs_new2.pickle 
```

+ select template INI file starting with `zs_`, fill in values for beta and
import.

+ run with this INI file 
```console 
$ python run_experiment.py zs_baseline.ini
```
