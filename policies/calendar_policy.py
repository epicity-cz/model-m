
def calendar_policy(graph, policy_coefs, history, tseries, time, contact_history=None):

    if int(time) == 11:
        # close everything
        close = [
            "nursary_children_inclass",
            "nursary_teachers_to_children",
            "lower_elementary_children_inclass",
            "lower_elementary_teachers_to_children",
            "higher_elementary_children_inclass",
            "higher_elementary_teachers_to_children",
            "highschool_children_inclass",
            "highschool_teachers_to_children",
            "nursary_children_coridors",
            "lower_elementary_children_coridors",
            "higher_elementary_children_coridors",
            "highschool_children_coridors",
            "nursary_teachers",
            "lower_elementary_teachers",
            "higher_elementary_teachers",
            "highschool_teachers"
        ]
        weaken = [
            "leasure_outdoor",
            "leasure_visit",
            "leasure_pub",
            "work_contacts",
            "work_workers_to_clients_distant",
            "work_workers_to_clients_plysical_short",
            "work_workers_to_clients_physical_long",
            "public_transport",
            "shops_customers",
            "shops_workers_to_clients",
            "pubs_customers",
            "pubs_workers_to_clients",
        ]
        coefs = [0.5, 0.1, 0.0, 
                 0.5, 0.5, 0.5, 0.5, 
                 0.2, 0.1, 0.5, 0, 0]
        graph.close_layers(close)
        graph.close_layers(weaken, coefs)
        return {"graph": None}

    if int(time) == 25:
        # open little bit
        stronger = [
            "leasure_outdoor",
            "leasure_visit",
            "leasure_pub",
            "work_contacts",
            "work_workers_to_clients_distant",
            "work_workers_to_clients_plysical_short",
            "work_workers_to_clients_physical_long",
            "public_transport",
            "shops_customers",
            "shops_workers_to_clients",
        ]
        coefs = [0.9, 0.5, 0.0, 
                 0.75, 0.75, 0.75, 0.75,
                 0.3, 0.4, 0.75]
        graph.close_layers(stronger, coefs)
        return {"graph": None}

    return {}
