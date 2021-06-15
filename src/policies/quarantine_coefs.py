import numpy as  np 

QUARANTINE_COEFS = {
    0: 0,
    1: 1.0,  # family_inside
    2: 1.0,  # family_in_house
    3: 0,  # family_visitsors_to_visited
    4: 0,  # nursary_children_inclass
    5: 0,  # nursary_teachers_to_children
    6: 0,  # lower_elementary_children_inclass
    7: 0,  # lower_elementary_teachers_to_children
    8: 0,  # higher_elementary_children_inclass
    9: 0,  # higher_elementary_teachers_to_children
    10: 0,  # highschool_children_inclass
    11: 0,  # highschool_teachers_to_children
    12: 0,  # nursary_children_coridors
    13: 0,  # elementary_children_coridors
    14: 0,  # highschool_children_coridors
    15: 0,  # nursary_teachers
    16: 0,  # elementary_teachers
    17: 0,  # highschool_teachers
    18: 0,  # leasure_outdoor
    19: 0,  # leasure_visit
    20: 0,  # leasure_pub
    21: 0,  # work_contacts
    22: 0,  # work_workers_to_clients_distant
    23: 0,  # work_workers_to_clients_plysical_short
    24: 0,  # work_workers_to_clients_physical_long
    25: 0,  # public_transport
    26: 0,  # shops_customers
    27: 0,  # shops_workers_to_clients
    28: 0,  # pubs_customers
    29: 0,  # pubs_workers_to_clients
    30: 0,
    31: 0,  # superspreader
    32: 0,  # superspreader
    33: 0,  # superspreader
    34: 0, # superspreader
    35: 0  # superspreader
}

WEE_COLD_COEFS = {
    0: 0,
    1: 1.0,  # family_inside
    2: 1.0,  # family_in_house
    3: 0.9,  # family_visitsors_to_visited
    4: 0,  # nursary_children_inclass
    5: 0,  # nursary_teachers_to_children
    6: 0,  # lower_elementary_children_inclass
    7: 0,  # lower_elementary_teachers_to_children
    8: 0,  # higher_elementary_children_inclass
    9: 0,  # higher_elementary_teachers_to_children
    10: 0,  # highschool_children_inclass
    11: 0,  # highschool_teachers_to_children
    12: 0,  # nursary_children_coridors
    13: 0,  # elementary_children_coridors
    14: 0,  # highschool_children_coridors
    15: 0,  # nursary_teachers
    16: 0,  # elementary_teachers
    17: 0,  # highschool_teachers
    18: 0,  # leasure_outdoor
    19: 0,  # leasure_visit
    20: 0,  # leasure_pub
    21: 0,  # work_contacts
    22: 0,  # work_workers_to_clients_distant
    23: 0,  # work_workers_to_clients_plysical_short
    24: 0,  # work_workers_to_clients_physical_long
    25: 0,  # public_transport
    26: 0,  # shops_customers
    27: 0,  # shops_workers_to_clients
    28: 0,  # pubs_customers
    29: 0,  # pubs_workers_to_clients
    30: 0,  # superspreader
    31: 0,
    32: 0,  # superspreader
    33: 0,  # superspreader
    34: 0, # superspreader
    35: 0  # superspreader

}

def dict2risk(rdict):
    ret = np.empty(36, dtype=float)
    for key, value in rdict.items():
        ret[key] = value 
    return ret 

RISK_FOR_LAYERS = dict2risk({
    0: 0,
    1: 1,  # family_inside
    2: 1,  # family_in_house
    3: 1,  # family_visitsors_to_visited
    4: 0.8,  # nursary_children_inclass
    5: 0.8,  # nursary_teachers_to_children
    6: 0.8,  # lower_elementary_children_inclass
    7: 0.8,  # lower_elementary_teachers_to_children
    8: 0.8,  # higher_elementary_children_inclass
    9: 0.8,  # higher_elementary_teachers_to_children
    10: 0.8,  # highschool_children_inclass
    11: 0.8,  # highschool_teachers_to_children
    12: 0.8,  # nursary_children_coridors
    13: 0.8,  # elementary_children_coridors
    14: 0.8,  # highschool_children_coridors
    15: 0.8,  # nursary_teachers
    16: 0.8,  # elementary_teachers
    17: 0.8,  # highschool_teachers
    18: 0.4,  # leasure_outdoor
    19: 0.5,  # leasure_visit
    20: 0.3,  # leasure_pub
    21: 0.8,  # work_contacts
    22: 0.8,  # work_workers_to_clients_distant
    23: 0.8,  # work_workers_to_clients_plysical_short
    24: 0.8,  # work_workers_to_clients_physical_long
    25: 0.0,  # public_transport
    26: 0.0,  # shops_customers
    27: 0,  # shops_workers_to_clients
    28: 0,  # pubs_customers
    29: 0,  # pubs_workers_to_clients
    30: 0,  # superspreader
    31: 0,
    32: 0,  # superspreader
    33: 0,  # superspreader
    34: 0, # superspreader
    35: 0  # superspreader

})

RISK_FOR_LAYERS_60 = dict2risk({
    0: 0,
    1: 1,  # family_inside
    2: 1,  # family_in_house
    3: 1,  # family_visitsors_to_visited
    4: 0.6,  # nursary_children_inclass
    5: 0.6,  # nursary_teachers_to_children
    6: 0.6,  # lower_elementary_children_inclass
    7: 0.6,  # lower_elementary_teachers_to_children
    8: 0.6,  # higher_elementary_children_inclass
    9: 0.6,  # higher_elementary_teachers_to_children
    10: 0.6,  # highschool_children_inclass
    11: 0.6,  # highschool_teachers_to_children
    12: 0.6,  # nursary_children_coridors
    13: 0.6,  # elementary_children_coridors
    14: 0.6,  # highschool_children_coridors
    15: 0.6,  # nursary_teachers
    16: 0.6,  # elementary_teachers
    17: 0.6,  # highschool_teachers
    18: 0.3,  # leasure_outdoor
    19: 0.3,  # leasure_visit
    20: 0.3,  # leasure_pub
    21: 0.6,  # work_contacts
    22: 0.6,  # work_workers_to_clients_distant
    23: 0.6,  # work_workers_to_clients_plysical_short
    24: 0.6,  # work_workers_to_clients_physical_long
    25: 0.0,  # public_transport
    26: 0.0,  # shops_customers
    27: 0,  # shops_workers_to_clients
    28: 0,  # pubs_customers
    29: 0,  # pubs_workers_to_clients
    30: 0,  # superspreader
    31: 0,
    32: 0,  # superspreader
    33: 0,  # superspreader
    34: 0, # superspreader
    35: 0  # superspreader
})

RISK_FOR_LAYERS_10 = dict2risk({
    0: 0,
    1: 1,  # family_inside
    2: 1,  # family_in_house
    3: 1,  # family_visitsors_to_visited
    4: 0.1,  # nursary_children_inclass
    5: 0.1,  # nursary_teachers_to_children
    6: 0.1,  # lower_elementary_children_inclass
    7: 0.1,  # lower_elementary_teachers_to_children
    8: 0.1,  # higher_elementary_children_inclass
    9: 0.1,  # higher_elementary_teachers_to_children
    10: 0.1,  # highschool_children_inclass
    11: 0.1,  # highschool_teachers_to_children
    12: 0.1,  # nursary_children_coridors
    13: 0.1,  # elementary_children_coridors
    14: 0.1,  # highschool_children_coridors
    15: 0.1,  # nursary_teachers
    16: 0.1,  # elementary_teachers
    17: 0.1,  # highschool_teachers
    18: 0.05,  # leasure_outdoor
    19: 0.05,  # leasure_visit
    20: 0.05,  # leasure_pub
    21: 0.1,  # work_contacts
    22: 0.1,  # work_workers_to_clients_distant
    23: 0.1,  # work_workers_to_clients_plysical_short
    24: 0.1,  # work_workers_to_clients_physical_long
    25: 0.0,  # public_transport
    26: 0.0,  # shops_customers
    27: 0,  # shops_workers_to_clients
    28: 0,  # pubs_customers
    29: 0,  # pubs_workers_to_clients
    30: 0,  # superspreader
    31: 0,
    32: 0,  # superspreader
    33: 0,  # superspreader
    34: 0, # superspreader
    35: 0  # superspreader
})

RISK_FOR_LAYERS_30 = dict2risk({
    0: 0,
    1: 1,  # family_inside
    2: 1,  # family_in_house
    3: 1,  # family_visitsors_to_visited
    4: 0.3,  # nursary_children_inclass
    5: 0.3,  # nursary_teachers_to_children
    6: 0.3,  # lower_elementary_children_inclass
    7: 0.3,  # lower_elementary_teachers_to_children
    8: 0.3,  # higher_elementary_children_inclass
    9: 0.3,  # higher_elementary_teachers_to_children
    10: 0.3,  # highschool_children_inclass
    11: 0.3,  # highschool_teachers_to_children
    12: 0.3,  # nursary_children_coridors
    13: 0.3,  # elementary_children_coridors
    14: 0.3,  # highschool_children_coridors
    15: 0.3,  # nursary_teachers
    16: 0.3,  # elementary_teachers
    17: 0.3,  # highschool_teachers
    18: 0.15,  # leasure_outdoor
    19: 0.15,  # leasure_visit
    20: 0.15,  # leasure_pub
    21: 0.3,  # work_contacts
    22: 0.3,  # work_workers_to_clients_distant
    23: 0.3,  # work_workers_to_clients_plysical_short
    24: 0.3,  # work_workers_to_clients_physical_long
    25: 0.0,  # public_transport
    26: 0.0,  # shops_customers
    27: 0,  # shops_workers_to_clients
    28: 0,  # pubs_customers
    29: 0,  # pubs_workers_to_clients
    30: 0,  # superspreader
    31: 0,
    32: 0,  # superspreader
    33: 0,  # superspreader
    34: 0, # superspreader
    35: 0  # superspreader
})



RISK_FOR_LAYERS_MINI = dict2risk({
    0: 0,
    1: 1,  # family_inside
    2: 1,  # family_in_house
    3: 0,  # family_visitsors_to_visited
    4: 0.0,  # nursary_children_inclass
    5: 0.0,  # nursary_teachers_to_children
    6: 0.0,  # lower_elementary_children_inclass
    7: 0.0,  # lower_elementary_teachers_to_children
    8: 0.0,  # higher_elementary_children_inclass
    9: 0.0,  # higher_elementary_teachers_to_children
    10: 0.0,  # highschool_children_inclass
    11: 0.0,  # highschool_teachers_to_children
    12: 0.0,  # nursary_children_coridors
    13: 0.0,  # elementary_children_coridors
    14: 0.0,  # highschool_children_coridors
    15: 0.0,  # nursary_teachers
    16: 0.0,  # elementary_teachers
    17: 0.0,  # highschool_teachers
    18: 0.0,  # leasure_outdoor
    19: 0.0,  # leasure_visit
    20: 0.0,  # leasure_pub
    21: 0.0,  # work_contacts
    22: 0.0,  # work_workers_to_clients_distant
    23: 0.0,  # work_workers_to_clients_plysical_short
    24: 0.0,  # work_workers_to_clients_physical_long
    25: 0.0,  # public_transport
    26: 0.0,  # shops_customers
    27: 0,  # shops_workers_to_clients
    28: 0,  # pubs_customers
    29: 0,  # pubs_workers_to_clients
    30: 0,  # superspreader
    31: 0,
    32: 0,  # superspreader
    33: 0,  # superspreader
    34: 0, # superspreader
    35: 0  # superspreader
})


RISK_FOR_LAYERS_MAX = dict2risk({
    0: 0,
    1: 1.0,  # family_inside
    2: 1.0,  # family_in_house
    3: 1.0,  # family_visitsors_to_visited
    4: 1.0,  # nursary_children_inclass
    5: 1.0,  # nursary_teachers_to_children
    6: 1.0,  # lower_elementary_children_inclass
    7: 1.0,  # lower_elementary_teachers_to_children
    8: 1.0,  # higher_elementary_children_inclass
    9: 1.0,  # higher_elementary_teachers_to_children
    10: 1.0,  # highschool_children_inclass
    11: 1.0,  # highschool_teachers_to_children
    12: 1.0,  # nursary_children_coridors
    13: 1.0,  # elementary_children_coridors
    14: 1.0,  # highschool_children_coridors
    15: 1.0,  # nursary_teachers
    16: 1.0,  # elementary_teachers
    17: 1.0,  # highschool_teachers
    18: 1.0,  # leasure_outdoor
    19: 1.0,  # leasure_visit
    20: 1.0,  # leasure_pub
    21: 1.0,  # work_contacts
    22: 1.0,  # work_workers_to_clients_distant
    23: 1.0,  # work_workers_to_clients_plysical_short
    24: 1.0,  # work_workers_to_clients_physical_long
    25: 1.0,  # public_transport
    26: 1.0,  # shops_customers
    27: 1.0,  # shops_workers_to_clients
    28: 1.0,  # pubs_customers
    29: 1.0,  # pubs_workers_to_clients
    30: 1.0,  
    31: 1.0,  # superspreader
    32: 1.0,  # superspreader
    33: 1.0,  # superspreader
    34: 1.0, # superspreader
    35: 1.0  # superspreader
})

def get_riskiness(family, work, leasure, rest=0):
    ret = np.empty(36, dtype=float) 
    ret[0] = 0 
    ret[[1, 2, 3]] = family 
    ret[range(4, 18)] = work 
    ret[range(21, 25)] = work 
    ret[[18, 19, 20]] = leasure
    ret[range(25, 36)] = rest 
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
