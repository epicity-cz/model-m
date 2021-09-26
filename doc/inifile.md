# INI FILE

All the user adjustable configuration is defined by an INI file. The INI file is parsed by python`s
module [configparser](https://docs.python.org/3/library/configparser.html) and follows the common INI file syntax.

This document lists all the sections and their keys.

# TASK

|key|type|default|meaning|
|---|---|---|---|
|duration_in_days|number|60|duration of the simulation in days|
|print_interval|number|1|period of printing the information of numbers of nodes in individual states, in days, -1 = never print| 
|verbose|string|Yes|Use no for no output during simulation| 
|model|string|SimulationDrivenModel|type of model for backward compatibility; older models are not supported, change at own risk|
|save_node_states|string|No|if "Yes", states of all nodes are saved| 
|monitor_node|number|None|If you provide a number of a node, this node is tracked and information about state changes, testing, quarantine, etc. are printed. Hint: Gabriela is 29681|
|output_dir|string|.|path to directory which is used for saving output files|

# GRAPH

|key|type|default|meaning|
|---|---|---|---|
|type|string|light|only "light" is supported by SimulationDrivenModel; for backward compatibility with older models| 
|nodes|string|p.csv|filename for nodes|
|edges|string|edges.csv|filename for edges|
|layers|string|etypes.csv|filename for etypes| 
|externals|string|e.csv|filename for export nodes|
|nums|string|nums.csv|not used; for backward compatibility| 
|objects|string|objects.csv|filename for objects| 
|quarantine|string|--|filename for quarantine specification, optional, needed only by graphs with custom layers|
|layer_groups|string|--|layer groups defition for contact tracing, optional, needed only by graphs with custom layers|
|file|string|town.pickle|filename to save the pickled graph|

# GRAPH_GEN

### GRAPH_GEN

Some default values for bigger arrays or matrix configuarions are not copied in full here. See source code
of [base_config.py](../src/gen_graph/config/base_config.py) for actual values.

This section contains all the configuration used for graph generation. All the defaults are set to some "sane" and
working values.

|key|type|default|meaning|
|---|---|---|---|
|base_dir|string|..|path used as root for file locations settings|
|file_nodes|string|p.csv|filename for nodes
|file_export|string|e.csv|filename for exports
|file_edges|string|edges.csv|filename for edges
|file_layers|string|etypes.csv|filename for layers
|file_nums|string|nums.csv|filename for counts of objects
|file_objects|string|objects.csv|filename for objects listing
|towns_ob|comma separated list|_empty_|list of _ZUJ_ to include
|towns_orp|comma separated list|_empty_|list of _ORP_ to include
|name|string|unknown|descriptor of generated graph. Used by some other default config values
|load_persons|boolean|No|Should persons be loaded (from file `<BASE_DIR>/config/gen_graph/<NAME>_osoby.csv` ) or automatically generated
|class_size|number|20|number of students in one class
|additional_teachers|number|2|nuber of additional teacher generated for every class
|mutual_limit|number|25|maximum size of group to generate every mutual contact. For larger groups only some contacts would be generated
|max_all_edges|number|20|maximum size of group of students to generate every teacher contact. For larger groups only some contacts would be generated
|school_tc_num_rc|number|8|number of contact generated for group of students larger then <MAX_ALL_EDGES>
|first_in_friends|number|10|minimum age of person to be included in friends network
|opp_sex_accept|number|0.2|degradation of friends probability for opposite sex
|ten_yrs_accept|number|0.3|base of power degradation of friends probability defined by age difference. The degradation is TEN_YRS_ACCEPT ** ABS((person1.age - person2.age)/10)
|friend_dist_pref|number|0.65|parameter of friends probability defined by distance
|friends_kappa|number|4|kappa parameter of friends network
|friends_lambda|number|6|lambda parameter of friends network
|friends_min_age_pub|number|18|minimal age allowing friends to meet in pub
|friends_pub_dist_pref|number|0.5|weight of choosing preferred pub by distance
|friends_sample|number|5000|sample size for calculating local cluster coef
|friends_accept_edge|number|0.99|ratio for accepting an edge in friends network
|family_age_same|number|10|max age distance to be treated as "the same age"
|family_factor_limit|number|60|maximum age for use of delta parameter
|family_senior_age|number|50|minimum age for treating person as potential senior to be visited
|family_child_age_difference|number|20|minimum age difference of junior and senior (parent and child)
|first_nursary|number|3|minimal age for attending nursary school
|last_nursary|number|5|maximal age for attending nursary school
|first_elem|number|6|minimal age for attending elementary school
|last_first_elem|number|10|maximal age for attending first level of elementary school
|last_elem|number|14|maximal age for attending elementary school
|first_highschool|number|15|minimal age for attending secondary school
|last_highschool|number|18|maximal age for attending secondary school
|last_pot_student|number|24|maximal age a student of a university
|first_retired|number|55|minimal age for retired person
|prob_of_univ|array of numbers|[0.7, 0.6]| probability of attending a university based on sex (male/female)
|work_ratio_adjust|number|0.91|adjustment of slace factor for workers
|other_age_adjust|number|8|adjusting parameter for "other" contacts by age
|other_same_town_adjust|number|0.1|adjusting parameter for "other" contacts if both in same town
|other_probability|number|0.21|base probability for "other" contacts
|other_is_export|number|0.01|probability that contact in "other" would be handled by an export node
|max_age|number|100|maximum age to work with
|eating_rate|number|2|lambda parameter for pub visits
|single_export|boolean|Yes|generate single export node. If No file `<BASE_DIR>/config/gen_graph/<NAME>_exportplaces.csv` must be present
|child_is_export|number|0.001|probability that a child in family network would be handled by an export node
|client_is_export|number|0.01|probability that a client in working network is handled by export node
|intensities|array|see [base_config.py](../src/gen_graph/config/base_config.py)|default intensities for contacts by type <br>`CLOSE_LONG_TERM`<br>`PHYSICAL_LONG_TERM`<br>`PHYSICAL_SHORT_TERM`<br>`DININING`<br>`DISTANT_LONG_TERM`<br>`CLOSE_RANDOM`<br>`SERVICE_CLIENT`<br>`CLOSE_OPEN_AIR`<br>
|number_of_children|array|see [base_config.py](../src/gen_graph/config/base_config.py)|probabilities of nuber of childs (0-5)
|work_matrix|matrix|see [base_config.py](../src/gen_graph/config/base_config.py)|work contacts based on work type. rows and columns represent work type by <br>`AGRICULTURE` <br>`INDUSTRY` <br>`CONSTRUCTION` <br>`TRADE_AND_MOTO` <br>`TRANSPORTATION` <br>`EATING_HOUSING` <br>`IT` <br>`BANKS` <br>`ADMIN_ETC` <br>`PUBLIC_SECTOR` <br>`EDUCATION` <br>`HEALTH`
|client_mult|number|10|multiplication factor for client in working network
|work_2_client|array|see [base_config.py](../src/gen_graph/config/base_config.py)|Clients counts for different work type
|work_export_rate|number|0.4 * 6|Rate for export node in work network
|csu_max_hh_size|number|6| maximal household size in CSU data
|travel_info|string|TravelInfoDelaunaySparse| `TravelInfoTree` <br> `TravelInfoDelaunay` <br>`TravelInfoDelaunaySparse`<br> `TravelInfoFile`<br> see [travel info](travel-info.md)
|calibrate_eps|number|1e-4| `epsilon` parameter for calibration
|calibrate_gtol|number|1e-2| `gtol` parameter fro calibration
|commuting_export|array of numbers|see [base_config.py](../src/gen_graph/config/base_config.py)|probabilities that given commuting time is handled by export node

# MODEL

|key|type|default|meaning|
|---|---|---|---|
|start_day|number|0|day to start the simulation|
|durations_file|string|../config/duration_probs.json|file with probs for calculation durations of RNA positivity, infectious time and incubation perios|
|prob_death_file|string|../data/prob_death.csv|file with probabilities of death by age|
|mu|number|0|multiplies the probability of death|
|ext_epi|number|0|probability of beeing infectious for external nodes|
|beta|number|0|rate of transmission (upon exposure) (note that transmission depands also on intensity of edge of contact)|
|beta_reduction|number|0|multiplication coefficient for beta of asymptomatic nodes|
|theta_Is|number|0|prob of being tested after decision of going to the test is made|      
|test_rate|number|0|prob of deciding to go for test if symptomatic|
|asymptomatic_rate|number|0|rate of asymtomatic flow after being exposed|
|init_X|number|0|initial number of nodes in state X; the rest of nodes is asigned to S| 

# POLICY

|key|type|default|meaning|
|---|---|---|---|
|filename|string|None|filename for policy code|
|name|string|None|name of the policy object| 

# POLICY SETUP

This depends on your policy. The following parameters are used for `customised_policy`. This policy enables you to
control various parameters of the model and also to run other policies.

|key|type|default|meaning|
|---|---|---|---|
|layer_changes_filename|string|None|layer weights calendar, csv file|
|policy_calendar_filename|string|None|json file with policy calendar|
|beta_factor_filename|string|None|csv file, values between 0.0 and 1.0, compliance with protective meassures|
|face_masks_filename|string|None|csv file, values between 0.0 and 1.0, compliance with wearing masks|
|theta_filename|string|None|csv file, multiplication of theta_Is calendar| 
|test_rate_filename|string|None|csv file, multiplication of test rate calender|
|init_filename|string|None|json, additional seeding with E nodes|
|reduction_coef1|number|1.0|controls reduction by wearing masks|
|reduction_coef2|number|1.0|controls reduction by protective meassures|
|new_beta|string|None|must be 'Yes', for backward compability| 
|sub_policies|comma separated|None|list of aliases for other policies to run|
|<POLICY_ALIAS>_filename|string|None|name of file with policy code|
|<POLICY_ALIAS>_name|string|None|name of policy object|
|<POLICY_ALIAS>_config|string|None|file with policy config, see [POLICY](policy.md)|

