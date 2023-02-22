# model-m :mask:
Model of an imaginary town 

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
$ export PYTHONPATH=../models:../model_m:../graphs:../policies:../utils
```

## Usage 

1. Loading a graph from CSV files takes minutes. Load the graph once and save it to a pickle file.  
```
Usage: preload_graph.py [OPTIONS] [FILENAME] [OUTPUTNAME]

  Load the graph and pickle.

$ python preload_graph.py load.ini town.pickle
```

2. Run your experiment. 
```
Usage: run_experiment.py [OPTIONS] [FILENAME] [TEST_ID]

  Run the demo test inside the timeit

Options:
  --set-random-seed / -r, --no-random-seed
  -p, --policy TEXT
  --print_interval INTEGER
  --n_repeat INTEGER
  --help                          Show this message and exit.

$ python run_experiment.py -r baseline.ini
``` 

3. Example of INI file:
(**warning:** this is an example only, the model parameters are not final)
```
[TASK]
duration_in_days = 150
print_interval = 1
verbose = Yes
model = TGMNetworkModel
save_node_states = No

[GRAPH]
name = pickle
file = papertown.pickle

[POLICY]
filename = customised_policy 
name = CustomPolicy

[POLICY_SETUP]
layer_changes_filename = ../data/policy_params/bourbon.csv
policy_calendar_filename = ../data/policy_params/policy_calendar.json
param_changes_filename = ../data/policy_params/param_changes31.5.json

[MODEL]
beta=0.5
beta_reduction=0.20
beta_in_family=0.5
beta_A=0.25
beta_A_in_family=0.25
sigma=0.174947033019464
gamma_In=0.0689372202959773
mu=0.000399920010665578
p=0.00
beta_D=0
gamma_Is=0.0465030451665233
gamma_Id=0.0465030451665233
theta_E=0.0
theta_Ia=0.0
theta_Is=0.0
theta_In=0.0
psi_E=1
psi_Ia=1
psi_Is=1
psi_In=1
q=0
false_symptoms_rate=0.005
false_symptoms_recovery_rate=0.4
asymptomatic_rate=0.3
symptoms_manifest_rate=0.393469340287367
init_E = 6
init_I_n = 0
init_I_a = 0
init_I_s = 0
init_I_d = 0
```

## References:

:bookmark: [Model antiCOVID-19 pro ČR při IDEA CERGE-EI](https://idea.cerge-ei.cz/anti-covid-19/iniciativa-model-anticovid-19-pro-cr)

:bookmark: [Změmy chování české populace v době COVID-19 a jejich reflexe v epidemiologických modelech](https://idea.cerge-ei.cz/anti-covid-19/iniciativa-model-anticovid-19-pro-cr) (CERGE-EI, 19. 5. 2020)

:bookmark: [konference NZIS a ISIN Open 2020](https://nzis-open.uzis.cz/nzis-open-05-2020.html) (ÚZIS, 27.5. 2020)


<hr>

:construction:
<h1> TODO: rewrite README completely !!!! </h1> 

**old stuff (to be updated soon):** 

The model started as an extension of [seirsplus
model](https://github.com/ryansmcgee/seirsplus) adding more states. Later it
started to diverge by modifying the simulation cycle. Currently, the main
difference is the state-update only once a day and possibility of plugin callback
policy function that may modify the given graph (i.e. delete or weaken adges
for nodes in quarantine).  See [doc/model.pdf](doc/model.pdf) for more details




## Usage 
```
Usage: run_experiment.py [OPTIONS] [FILENAME] [TEST_ID]

  Run the demo test inside the timeit

Options:
  --set-random-seed / -r, --no-random-seed
  -p, --policy TEXT
  --print_interval INTEGER
  --help                          Show this message and exit.


```

See example .ini files romeo_and_juliet.ini (toy example) and seirsplus_example.ini (uses same
graph as examples in the seirsplus project)

```
(covid_env) (initial_experiments) petra@totoro:~/covid/model-m/tests$ python run_experiment.py -r romeo_and_juliet.ini

(covid_env) (initial_experiments) petra@totoro:~/covid/model-m/tests$ python
run_experiment.py -r -p strong_policy town0.ini
```

Output looks like:
(prints states overiew after the first event every day)
```
(covid_env) (initial_experiments) petra@totoro:~/covid/model-m/tests$ python run_experiment.py romeo_and_juliet.ini 

N =  37
t = 0.03
	 S = 25
	 S_s = 5
	 E = 0
	 I_n = 2
	 I_a = 1
	 I_s = 4
	 I_d = 0
	 R_d = 0
	 R_u = 0
	 D_d = 0
	 D_u = 0

t = 1.11
	 S = 24
	 S_s = 6
	 E = 0
	 I_n = 2
	 I_a = 1
	 I_s = 3
	 I_d = 0
	 R_d = 0
	 R_u = 1
	 D_d = 0
	 D_u = 0

	.... 
	
t = 59.04
	 S = 13
	 S_s = 2
	 E = 0
	 I_n = 0
	 I_a = 0
	 I_s = 1
	 I_d = 0
	 R_d = 14
	 R_u = 7
	 D_d = 0
	 D_u = 0

t = 60.04
	 S = 12.0
	 S_s = 3.0
	 E = 0.0
	 I_n = 0.0
	 I_a = 0.0
	 I_s = 1.0
	 I_d = 0.0
	 R_d = 14.0
	 R_u = 7.0
	 D_d = 0.0
	 D_u = 0.0

Avg. number of events per day:  7.7
```

## Config File Format
town0.ini
```
[TASK]
duration_in_days = 100

# print_interval  -  0:  do not print state counts during simulation
#                 -  N:  print every N-th day
print_interval = 1
verbose = Yes
model = ExtendedDailyNetworkModel

[GRAPH]
# name: romeo_and_juliet   - verona 
#       seirsplus_example  - graph from seirsplus example
#       csv                - from .csv file
name = csv
nodes = ../graphinput/hodonin.raw/20-04-12/nodes.csv
edges = ../graphinput/hodonin.raw/20-04-12/edges.csv
layers = ../graphinput/hodonin.raw/20-04-12/etypes.csv

[POLICY]
filename = test_policy
name = strong_policy, weighted_policy

[MODEL]
beta = 0.155
sigma = 0.1923076923076923
gamma = 0.099
mu_I = 0.0004
p = 0.2
beta_D = 0.155
gamma_D = 0.1
mu_D = 0.0004
theta_E = 0.1
theta_Ia = 0.1
theta_Is = 0.1
phi_E = 0
phi_Ia = 0
phi_Is = 0
psi_E = 1.0
psi_Ia = 1.0
psi_Is = 1.0
q = 0.1
false_symptoms_rate = 0.05
false_symptoms_recovery_rate = 0.4
asymptomatic_rate = 0.3
symptoms_manifest_rate = 0.9
init_E = 0
init_I_n =  15
init_I_a = 15
init_I_s = 70
init_I_d = 0



```


## Saved history
You may wish to generate a novel from a model history. Output example: 

> Once upon a time ...<br>
> A gentleman Page stopped to have flue symptoms and started to be healthy.<br>
> A gentleman Watchmen 1 stopped to be infectious without symptoms and started to be healthy again.<br>
> A gentleman Friar John stopped to be healthy and started to have flue symptoms.<br>
> A gentleman Prince Escalus stopped to manifest symptoms and started to push up daisies.<br>
> A gentleman Paris stopped to have flue symptoms and started to be healthy.<br>
> A lady Lady Capulet stopped to have flue symptoms and started to be healthy.<br>
> A gentleman Friar Lawrence stopped to be healthy and started to have flue symptoms.<br>
> A gentleman Balthasar stopped to be symptomatic and infectious with no  manifest of symptoms and started to manifest symptoms.<br>
> A gentleman Lord Capulet stopped to have flue symptoms and started to be healthy.<br>
> A gentleman Old Capulet stopped to have flue symptoms and started to be healthy.<br>
> A gentleman Chorus stopped to be healthy and started to have flue symptoms.<br>
> A gentleman Chorus stopped to have flue symptoms and started to be healthy.<br>
> A gentleman Friar Lawrence stopped to have flue symptoms and started to be healthy.<br>
> A lady Rosaline stopped to be healthy and started to have flue symptoms.<br>
> A gentleman Lord Capulet stopped to be healthy and started to have flue symptoms.<br>
> A gentleman Gregory stopped to have flue symptoms and started to be healthy.<br>
> A gentleman Balthasar stopped to manifest symptoms and started to push up daisies.<br>
> A lady Rosaline stopped to have flue symptoms and started to be healthy.<br>
> A lady Queen Mab stopped to manifest symptoms and started to push up daisies.<br>
> A gentleman Benvolio stopped to manifest symptoms and started to be as famous as a taxidriver.<br>
> A gentleman Mercutio stopped to manifest symptoms and started to be healthy again.<br>
> A gentleman Lord Capulet stopped to have flue symptoms and started to be healthy.<br>
> A gentleman Lord Capulet stopped to be healthy and started to have flue symptoms.<br>
> A gentleman Lord Capulet stopped to have flue symptoms and started to be healthy.<br>
> A gentleman Friar John stopped to have flue symptoms and started to be healthy.<br>
> A gentleman Benvolio stopped to be as famous as a taxidriver and started to pine for the fjords.<br>
> A gentleman Servant 2 stopped to be healthy and started to have flue symptoms.<br>
> A lady Nurse stopped to be healthy and started to have flue symptoms.<br>
> A gentleman Servant 2 stopped to have flue symptoms and started to be healthy.<br>
> A gentleman Servant 2 stopped to be healthy and started to have flue symptoms.<br>
> A gentleman Servant 2 stopped to have flue symptoms and started to be healthy.<br>
> A lady Nurse stopped to have flue symptoms and started to be healthy.<br>
> A gentleman Valentine stopped to be healthy and started to have flue symptoms.<br>
> A gentleman Valentine stopped to have flue symptoms and started to be healthy.<br>
> A lady Nurse stopped to be healthy and started to have flue symptoms.<br>
> A lady Nurse stopped to have flue symptoms and started to be healthy.<br>
> A lady Lady Capulet stopped to be healthy and started to have flue symptoms.<br>
> A gentleman Friar John stopped to be healthy and started to have flue symptoms.<br>
> A lady Lady Capulet stopped to have flue symptoms and started to be healthy.<br>
> A gentleman Friar John stopped to have flue symptoms and started to be healthy.<br>
> A gentleman Lord Capulet stopped to be healthy and started to have flue symptoms.<br>
> A gentleman Lord Capulet stopped to have flue symptoms and started to be healthy.<br>
> A gentleman Apothacary stopped to be healthy and started to have flue symptoms.<br>
> A gentleman Apothacary stopped to have flue symptoms and started to be healthy.<br>
> A gentleman Apothacary stopped to be healthy and started to have flue symptoms.<br>
> A gentleman Apothacary stopped to have flue symptoms and started to be healthy.<br>
> A gentleman Sampson stopped to be healthy and started to have flue symptoms.<br>
> A gentleman Lord Capulet stopped to be healthy and started to have flue symptoms.<br>
> A gentleman Abram stopped to be healthy and started to have flue symptoms.<br>
> A gentleman Lord Capulet stopped to have flue symptoms and started to be healthy.<br>
> A gentleman Sampson stopped to have flue symptoms and started to be healthy.<br>
> A lady Lady Capulet stopped to be healthy and started to have flue symptoms.<br>
> A lady Lady Capulet stopped to have flue symptoms and started to be healthy.<br>
> A gentleman Tybalt stopped to be healthy and started to have flue symptoms.<br>
> A gentleman Tybalt stopped to have flue symptoms and started to be healthy.<br>
> A gentleman Watchmen 2 stopped to be healthy and started to be exposed.<br>
> A gentleman Abram stopped to have flue symptoms and started to be healthy.<br>
> A gentleman Paris stopped to be healthy and started to have flue symptoms.<br>
> A gentleman Watchmen 3 stopped to be infectious without symptoms and started to be healthy again.<br>
> A gentleman Paris stopped to have flue symptoms and started to be healthy.<br>
> A gentleman Page stopped to be healthy and started to have flue symptoms.<br>
> A gentleman Watchmen 2 stopped to be exposed and started to be symptomatic and infectious with no  manifest of symptoms.<br>
> A gentleman Page stopped to have flue symptoms and started to be healthy.<br>
> A gentleman Watchmen 2 stopped to be symptomatic and infectious with no  manifest of symptoms and started to manifest symptoms.<br>
> A gentleman Watchmen 2 stopped to manifest symptoms and started to be as famous as a taxidriver.<br>
> A gentleman Peter stopped to be healthy and started to have flue symptoms.<br>
> A gentleman Watchmen 2 stopped to be as famous as a taxidriver and started to pine for the fjords.<br>
> Well! I never wanted to do this in the first place. I wanted to be... an epidemiologist!<br>


## Implementing custom network models

You can derive your customized network model.

```
import matplotlib.pyplot as plt
import numpy as np
from run_experiment import tell_the_story
from model import create_custom_model
from romeo_juliet_graph_gen import RomeoAndJuliet as Verona
from run_experiment import magic_formula
```

1. **Define whatever you need**
```
model_definition = {
    # model definition comes here
    "states": ["sleeping", "alert", "tired", "dead"],
    "transitions": [
        ("sleeping", "alert"),
        ("alert", "tired"),
        ("tired", "sleeping"),
        ("tired", "dead")
    ],
    # optionally:
    "final_states": ["dead"],
    "invisible_states": ["dead"],

    "model_parameters": {
        "wake_up_rate": (0.2, "wake up prob"),
        "tiredability": (0.3, "getting tired rate"),
        "mu": (0.1, "death rate"),
        "sleepiness": (0.7, "rate of falling in sleep")
    }
}


def calc_propensities(model):
    # define your calculations here
    # you may use various model utilities, as
    #       model.num_contacts(state or list of states),
    #       model.current_state_count(state), model.current_N(),
    #       etc.; access list of states, transitions, parameters.

    propensities = {}

    propensities[("sleeping", "alert")] = model.wake_up_rate * \
        (model.X == "sleeping")
    propensities[("alert",  "tired")] = (model.tiredability
                                         * (model.num_contacts(["alert", "tired"]) / model.current_N())
                                         * (model.X == "alert")
                                         )
    tired = model.X == "tired"
    propensities[("tired", "sleeping")] = model.sleepiness * tired
    propensities[("tired", "dead")] = model.mu * tired

    # TODO move this part to model.py
    propensities_list = [
        propensities[t] for t in model.transitions
    ]
    stacked_propensities = np.hstack(propensities_list)

    return stacked_propensities, model.transitions
```

2. **Create custom class**

```
CustomModel = create_custom_model("CustomModel", **model_definition,
                                  calc_propensities=calc_propensities)
```

3. **Load your graph**
```
g = Verona()
A = magic_formula(g.as_dict_of_graphs(), g.get_layers_info())
```

4. **Create model**
```
tiredability = 0.01 * np.array(g.get_attr_list("age"))
model = CustomModel(A,  wake_up_rate=0.8, init_alert=10, tiredability=tiredability,
                    init_tired=10, random_seed=35)
```

5. **Run**
```
ndays = 60
model.run(T=ndays, verbose=True, print_interval=5)
print("Avg. number of events per day: ", model.tidx/ndays)
```

6. **Inspect results**
```
x = model.tseries
population = model.N
alert = model.state_counts["alert"]
plt.plot(x, population, label="population")
plt.plot(x, alert, label="alert population")
plt.legend()
plt.savefig("alert_pop.png")
# etc
```
7. **Procrastinate**
```
# text = tell_the_story(model.history, g)
# print()
```
