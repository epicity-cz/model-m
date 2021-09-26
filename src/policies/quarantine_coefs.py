import numpy as  np 

"""
Quarantine coeficients are prefered to be provided with graph.
These are default cofficitients. 
"""

QUARANTINE_COEFS = {
    0:  1.0,  # family_inside
    1:  1.0,  # family_in_house
    2:  0,  # family_visitsors_to_visited
    3:  0,  # nursary_children_inclass
    4:  0,  # nursary_teachers_to_children
    5:  0,  # lower_elementary_children_inclass
    6:  0,  # lower_elementary_teachers_to_children
    7:  0,  # higher_elementary_children_inclass
    8:  0,  # higher_elementary_teachers_to_children
    9:  0,  # highschool_children_inclass
    10: 0,  # highschool_teachers_to_children
    11: 0,  # nursary_children_coridors
    12: 0,  # elementary_children_coridors
    13: 0,  # highschool_children_coridors
    14: 0,  # nursary_teachers
    15: 0,  # elementary_teachers
    16: 0,  # highschool_teachers
    17: 0,  # leasure_outdoor
    18: 0,  # leasure_visit
    19: 0,  # leasure_pub
    20: 0,  # work_contacts
    21: 0,  # work_workers_to_clients_distant
    22: 0,  # work_workers_to_clients_plysical_short
    23: 0,  # work_workers_to_clients_physical_long
    24: 0,  # public_transport
    25: 0,  # shops_customers
    26: 0,  # shops_workers_to_clients
    27: 0,  # pubs_customers
    28: 0,  # pubs_workers_to_clients
    29: 0,
    30: 0,  # superspreader
    31: 0,  # superspreader
    32: 0,  # superspreader
    33: 0, # superspreader
    34: 0  # superspreader
}   


def dict2risk(rdict):
    ret = np.empty(36, dtype=float)
    for key, value in rdict.items():
        ret[key] = value 
    return ret 

#RISK_FOR_LAYERS = None
RISK_FOR_LAYERS = dict2risk({
    0: 1,  # family_inside
    1: 1,  # family_in_house
    2: 1,  # family_visitsors_to_visited
    3: 0.8,  # nursary_children_inclass
    4: 0.8,  # nursary_teachers_to_children
    5: 0.8,  # lower_elementary_children_inclass
    6: 0.8,  # lower_elementary_teachers_to_children
    7: 0.8,  # higher_elementary_children_inclass
    8: 0.8,  # higher_elementary_teachers_to_children
    9:  0.8,  # highschool_children_inclass
    10: 0.8,  # highschool_teachers_to_children
    11: 0.8,  # nursary_children_coridors
    12: 0.8,  # elementary_children_coridors
    13: 0.8,  # highschool_children_coridors
    14: 0.8,  # nursary_teachers
    15: 0.8,  # elementary_teachers
    16: 0.8,  # highschool_teachers
    17: 0.4,  # leasure_outdoor
    18: 0.5,  # leasure_visit
    19: 0.3,  # leasure_pub
    20: 0.8,  # work_contacts
    21: 0.8,  # work_workers_to_clients_distant
    22: 0.8,  # work_workers_to_clients_plysical_short
    23: 0.8,  # work_workers_to_clients_physical_long
    24: 0.0,  # public_transport
    25: 0.0,  # shops_customers
    26: 0,  # shops_workers_to_clients
    27: 0,  # pubs_customers
    28: 0,  # pubs_workers_to_clients
    29: 0,  # superspreader
    30: 0,
    31: 0,  # superspreader
    32: 0,  # superspreader
    33: 0, # superspreader
    34: 0  # superspreader
})

#RISK_FOR_LAYERS_60 = None 
RISK_FOR_LAYERS_60 = dict2risk({
    0:  1,  # family_inside
    1:  1,  # family_in_house
    2:  1,  # family_visitsors_to_visited
    3:  0.6,  # nursary_children_inclass
    4:  0.6,  # nursary_teachers_to_children
    5:  0.6,  # lower_elementary_children_inclass
    6:  0.6,  # lower_elementary_teachers_to_children
    7:  0.6,  # higher_elementary_children_inclass
    8:  0.6,  # higher_elementary_teachers_to_children
    9:  0.6,  # highschool_children_inclass
    10: 0.6,  # highschool_teachers_to_children
    11: 0.6,  # nursary_children_coridors
    12: 0.6,  # elementary_children_coridors
    13: 0.6,  # highschool_children_coridors
    14: 0.6,  # nursary_teachers
    15: 0.6,  # elementary_teachers
    16: 0.6,  # highschool_teachers
    17: 0.3,  # leasure_outdoor
    18: 0.3,  # leasure_visit
    19: 0.3,  # leasure_pub
    20: 0.6,  # work_contacts
    21: 0.6,  # work_workers_to_clients_distant
    22: 0.6,  # work_workers_to_clients_plysical_short
    23: 0.6,  # work_workers_to_clients_physical_long
    24: 0.0,  # public_transport
    25: 0.0,  # shops_customers
    26: 0,  # shops_workers_to_clients
    27: 0,  # pubs_customers
    28: 0,  # pubs_workers_to_clients
    29: 0,  # superspreader
    30: 0,
    31: 0,  # superspreader
    32: 0,  # superspreader
    33: 0, # superspreader
    34: 0  # superspreader
})

#RISK_FOR_LAYERS_10 = None 
RISK_FOR_LAYERS_10 = dict2risk({
    0:  1,  # family_inside
    1:    1,  # family_in_house
    2:    1,  # family_visitsors_to_visited
    3:  0.1,  # nursary_children_inclass
    4:  0.1,  # nursary_teachers_to_children
    5:  0.1,  # lower_elementary_children_inclass
    6:  0.1,  # lower_elementary_teachers_to_children
    7:  0.1,  # higher_elementary_children_inclass
    8:  0.1,  # higher_elementary_teachers_to_children
    9:  0.1,  # highschool_children_inclass
    10: 0.1,  # highschool_teachers_to_children
    11: 0.1,  # nursary_children_coridors
    12: 0.1,  # elementary_children_coridors
    13: 0.1,  # highschool_children_coridors
    14: 0.1,  # nursary_teachers
    15: 0.1,  # elementary_teachers
    16: 0.1,  # highschool_teachers
    17: 0.05,  # leasure_outdoor
    18: 0.05,  # leasure_visit
    19: 0.05,  # leasure_pub
    20: 0.1,  # work_contacts
    21: 0.1,  # work_workers_to_clients_distant
    22: 0.1,  # work_workers_to_clients_plysical_short
    23: 0.1,  # work_workers_to_clients_physical_long
    24: 0.0,  # public_transport
    25: 0.0,  # shops_customers
    26: 0,  # shops_workers_to_clients
    27: 0,  # pubs_customers
    28: 0,  # pubs_workers_to_clients
    29: 0,  # superspreader
    30: 0,
    31: 0,  # superspreader
    32: 0,  # superspreader
    33: 0, # superspreader
    34: 0  # superspreader
})

#RISK_FOR_LAYERS_30 = None
RISK_FOR_LAYERS_30 = dict2risk({
    0:  1,  # family_inside
    1:  1,  # family_in_house
    2:  1,  # family_visitsors_to_visited
    3:  0.3,  # nursary_children_inclass
    4:  0.3,  # nursary_teachers_to_children
    5:  0.3,  # lower_elementary_children_inclass
    6:  0.3,  # lower_elementary_teachers_to_children
    7:  0.3,  # higher_elementary_children_inclass
    8:  0.3,  # higher_elementary_teachers_to_children
    9:  0.3,  # highschool_children_inclass
    10: 0.3,  # highschool_teachers_to_children
    11: 0.3,  # nursary_children_coridors
    12: 0.3,  # elementary_children_coridors
    13: 0.3,  # highschool_children_coridors
    14: 0.3,  # nursary_teachers
    15: 0.3,  # elementary_teachers
    16: 0.3,  # highschool_teachers
    17: 0.15,  # leasure_outdoor
    18: 0.15,  # leasure_visit
    19: 0.15,  # leasure_pub
    20: 0.3,  # work_contacts
    21: 0.3,  # work_workers_to_clients_distant
    22: 0.3,  # work_workers_to_clients_plysical_short
    23: 0.3,  # work_workers_to_clients_physical_long
    24: 0.0,  # public_transport
    25: 0.0,  # shops_customers
    26: 0,  # shops_workers_to_clients
    27: 0,  # pubs_customers
    28: 0,  # pubs_workers_to_clients
    29: 0,  # superspreader
    30: 0,
    31: 0,  # superspreader
    32: 0,  # superspreader
    33: 0, # superspreader
    34: 0  # superspreader
})

#RISK_FOR_LAYERS_MINI = None
RISK_FOR_LAYERS_MINI = dict2risk({
    0:  1,  # family_inside
    1:  1,  # family_in_house
    2:  0,  # family_visitsors_to_visited
    3:  0.0,  # nursary_children_inclass
    4:  0.0,  # nursary_teachers_to_children
    5:  0.0,  # lower_elementary_children_inclass
    6:  0.0,  # lower_elementary_teachers_to_children
    7:  0.0,  # higher_elementary_children_inclass
    8:  0.0,  # higher_elementary_teachers_to_children
    9:  0.0,  # highschool_children_inclass
    10: 0.0,  # highschool_teachers_to_children
    11: 0.0,  # nursary_children_coridors
    12: 0.0,  # elementary_children_coridors
    13: 0.0,  # highschool_children_coridors
    14: 0.0,  # nursary_teachers
    15: 0.0,  # elementary_teachers
    16: 0.0,  # highschool_teachers
    17: 0.0,  # leasure_outdoor
    18: 0.0,  # leasure_visit
    19: 0.0,  # leasure_pub
    20: 0.0,  # work_contacts
    21: 0.0,  # work_workers_to_clients_distant
    22: 0.0,  # work_workers_to_clients_plysical_short
    23: 0.0,  # work_workers_to_clients_physical_long
    24: 0.0,  # public_transport
    25: 0.0,  # shops_customers
    26: 0,  # shops_workers_to_clients
    27: 0,  # pubs_customers
    28: 0,  # pubs_workers_to_clients
    29: 0,  # superspreader
    30: 0,
    31: 0,  # superspreader
    32: 0,  # superspreader
    33: 0, # superspreader
    34: 0  # superspreader
})  

#RISK_FOR_LAYERS_MAX = None
RISK_FOR_LAYERS_MAX = dict2risk({
    0:  1.0,  # family_inside
    1:  1.0,  # family_in_house
    2:  1.0,  # family_visitsors_to_visited
    3:  1.0,  # nursary_children_inclass
    4:  1.0,  # nursary_teachers_to_children
    5:  1.0,  # lower_elementary_children_inclass
    6:  1.0,  # lower_elementary_teachers_to_children
    7:  1.0,  # higher_elementary_children_inclass
    8:  1.0,  # higher_elementary_teachers_to_children
    9:  1.0,  # highschool_children_inclass
    10: 1.0,  # highschool_teachers_to_children
    11: 1.0,  # nursary_children_coridors
    12: 1.0,  # elementary_children_coridors
    13: 1.0,  # highschool_children_coridors
    14: 1.0,  # nursary_teachers
    15: 1.0,  # elementary_teachers
    16: 1.0,  # highschool_teachers
    17: 1.0,  # leasure_outdoor
    18: 1.0,  # leasure_visit
    19: 1.0,  # leasure_pub
    20: 1.0,  # work_contacts
    21: 1.0,  # work_workers_to_clients_distant
    22: 1.0,  # work_workers_to_clients_plysical_short
    23: 1.0,  # work_workers_to_clients_physical_long
    24: 1.0,  # public_transport
    25: 1.0,  # shops_customers
    26: 1.0,  # shops_workers_to_clients
    27: 1.0,  # pubs_customers
    28: 1.0,  # pubs_workers_to_clients
    29: 1.0,  
    30: 1.0,  # superspreader
    31: 1.0,  # superspreader
    32: 1.0,  # superspreader
    33: 1.0, # superspreader
    34: 1.0  # superspreader
})  

LAYER_GROUPS = {
    "family": [0, 1, 2],
    "school_work": list(range(3, 17)) + list(range(20,24)),
    "leasure": [17, 18, 19],
    "rest": list(range(24, 35))
}

def get_riskiness(graph, family, work, leasure, rest=0.0):
    ret = np.zeros(len(graph.layer_name), dtype=float)
    if graph.LAYER_GROUPS is not None:
        lg = graph.LAYER_GROUPS
    else:
        lg = LAYER_GROUPS
        
    ret[lg["family"]] = family 
    ret[lg["school_work"]] = work
    ret[lg["leasure"]] = leasure 
    ret[lg["rest"]] = rest 

    # ret = { 0: 0.0 }
    # ret.update({
    #     key: family
    #     for key in [1, 2, 3]
    # })
    # ret.update({
    #     key: work
    #     for key in range(4, 18)
    # })
    # ret.update({
    #     key: work
    #     for key in range(21, 25)
    # })
    # ret.update({
    #     key: leasure
    #     for key in [18,19,20]
    # })
    # ret.update({
    #     key: rest
    #     for key in range(25,36)
    # })
    return ret 
