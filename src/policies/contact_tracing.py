import numpy as np
import pandas as pd
import logging
from itertools import chain

from policies.testing_policy import TestingPolicy
from utils.history_utils import TimeSeries
from models.agent_based_network_model import STATES


from policies.depo import Depo

from policies.quarantine_coefs import QUARANTINE_COEFS, RISK_FOR_LAYERS
from policies.quarantine_coefs import RISK_FOR_LAYERS_MAX, RISK_FOR_LAYERS_MINI, RISK_FOR_LAYERS_60, RISK_FOR_LAYERS_10, RISK_FOR_LAYERS_30
from policies.quarantine_coefs import get_riskiness

from utils.global_configs import monitor
import utils.global_configs as cfgs
from utils.config_utils import ConfigFile


class ContactTracingPolicy(TestingPolicy):

    """
    Testing + ContactTracing 

    policy responsible for testing and contact tracing
    - individuals with symptoms go for a test with a certain probability 
    - those positively tested undergo contact tracing
    """

    def __init__(self, graph, model, config_file=None):
        super().__init__(graph, model)

        if graph.QUARANTINE_COEFS is not None:
            self.coefs = graph.QUARANTINE_COEFS
        else:
            self.coefs = QUARANTINE_COEFS

        self.riskiness = get_riskiness(graph, 1.0, 0.8, 0.4)
        self.duration = 10
        self.duration_quara = 14
        self.days_back = 7
        self.phone_call_delay = 2
        self.enter_test_delay = 5

        # if node already passed enter test and it was negative
        self.negative_enter_test = np.zeros(self.model.num_nodes, dtype=bool)
        # the day of the last contact with infectious node, -1 for undefined
        self.last_contact = np.full(
            self.model.num_nodes, fill_value=-1, dtype=int)

        self.auto_recover = False  # no final condititon
        self.enter_test = True    # do enter test

        if config_file is not None:
            self.load_config(config_file)

        logging.info(f"CONTACT TRACING POLICY: riskiness is {self.riskiness}")

        # nodes marked as contacts and waiting for being informed
        self.waiting_room_phone_call = Depo(graph.number_of_nodes)

        # nodes waiting for enter-test
        self.waiting_room_enter_test = Depo(graph.number_of_nodes)

        # nodes waiting for enter-test result
        self.waiting_room_result_enter_test = Depo(graph.number_of_nodes)

        # nodes waiting for second test
        self.waiting_room_second_test = Depo(graph.number_of_nodes)

        # statistics
        self.stat_positive_enter_tests = TimeSeries(300, dtype=int)
        self.stat_contacts_collected = TimeSeries(300, dtype=int)

    def load_config(self, config_file):
        cf = ConfigFile()
        cf.load(config_file)

        isolation = cf.section_as_dict("ISOLATION")
        self.duration = isolation.get("duration", self.duration)

        quarantine = cf.section_as_dict("QUARANTINE")
        self.duration_quara = quarantine.get("DURATION", self.duration_quara)

        tracing = cf.section_as_dict("CONTACT_TRACING")
        cfg_riskiness = tracing.get("riskiness", None)
        if cfg_riskiness is not None:
            cfg_riskiness = list(map(float, cfg_riskiness))
            self.riskiness = get_riskiness(self.graph, *cfg_riskiness)
        self.days_back = tracing.get("days_back", self.days_back)
        self.phone_call_delay = tracing.get(
            "phone_call_delay", self.phone_call_delay)
        self.enter_test_delay = tracing.get(
            "enter_test_delay", self.enter_test_delay)
        self.auto_recover = tracing.get(
            "auto_recover", "Yes" if self.auto_recover else "No") == "Yes"
        self.enter_test = tracing.get(
            "enter_test", "Yes" if self.enter_test else "No") == "Yes"

    def first_day_setup(self):

        # fill the days before start by zeros
        self.stat_positive_enter_tests[0:self.model.t] = 0
        self.stat_contacts_collected[0:self.model.t] = 0

        super().first_day_setup()

    def select_contacts(self, detected_nodes):
        """
        Realizes contact tracing of detected nodes,  returns 
        the set of contacts. 
        """
        contacts = set()

        if len(detected_nodes) == 0:
            return contacts

        # for each node trace back 2 days before first symtoms but
        # at most (or if there are no symptoms) self.days_back days
        # (no symptoms -> first_symptoms == -1)
        days_back_array = np.clip(
            (self.model.t - self.first_symptoms[detected_nodes]) + 2,
            0,
            self.days_back
        )

        if cfgs.MONITOR_NODE is not None and cfgs.MONITOR_NODE in detected_nodes:
            monitor(self.model.t,
                    f"is beeing asked for contacts for {days_back_array[np.where(detected_nodes == cfgs.MONITOR_NODE)[0][0]]} days back.")

        for i, node in enumerate(detected_nodes):

            days_back = days_back_array[i]
            for t, day_list in enumerate(self.model.contact_history[-days_back:]):
                if day_list is None:  # first iterations
                    continue
                if len(day_list[0]) == 0:
                    continue
                # daylist is tuple (source_nodes, dest_nodes, types) including both directions
                my_edges = day_list[0] == node
                selected_contacts = day_list[1][my_edges]
                types = day_list[2][my_edges]

                selected_contacts = self.filter(selected_contacts, types)
                selected_contacts = self.filter_out_ext(selected_contacts)
                selected_contacts, _ = self.filter_dead(selected_contacts)

                self.last_contact[selected_contacts] = np.maximum(
                    self.last_contact[selected_contacts], self.model.t - days_back + t + 1)

                contacts.update(selected_contacts)

        return contacts

    def filter_out_ext(self, contacts):
        is_not_ext = self.model.memberships[STATES.EXT, contacts, 0] == 0
        return contacts[is_not_ext]

    def filter(self, contacts, types):
        """
        Randomly filters contacts. Returns those that are `recalled`. 
        Probability of recall is given by a riskiness of the type of the contact. 
        """
        risk = self.riskiness[types]
        r = np.random.rand(len(risk))
        return contacts[r < risk]

    def stop(self):
        """ just finish necessary, but do not qurantine new nodes """
        self.stopped = True

    def select_test_candidates(self):
        """ 
        Selects nodes that are going to be tested, exclude those alredy in isolation/quarantine.
        """
        test_candidates = super().select_test_candidates()
        # exclude those that are already in quarantine
        return self.depo.filter_locked(test_candidates)

    def process_detected_nodes(self, target_nodes):
        """
        Put detected nodes to quarantine, finish quaratines, do contact tracing.
        """

        released = self.tick()
        self.leaving_procedure(released)

        self.quarantine_nodes(target_nodes)
        if cfgs.MONITOR_NODE is not None and cfgs.MONITOR_NODE in target_nodes:
            monitor(self.model.t,
                    f"is sent to isolation.")

        # nodes asked for contacts
        contacts = self.select_contacts(target_nodes)

        # phone call
        contacted = self.waiting_room_phone_call.tick_and_get_released()
        if len(contacted) > 0:
            contacted = contacted[self.model.node_detected[contacted] == False]
        if cfgs.MONITOR_NODE is not None and cfgs.MONITOR_NODE in contacted:
            monitor(self.model.t,
                    f"recieves a phone call and is sent to quarantine.")
        self.quarantine_nodes(contacted, last_contacts=True,
                              duration=self.duration_quara)

        # enter test
        positive_contacts = self.enter_test_procedure(contacted)
        if cfgs.MONITOR_NODE is not None and cfgs.MONITOR_NODE in positive_contacts:
            monitor(self.model.t,
                    f"has positive enter test.")

        # another contacts
        another_contacts = self.select_contacts(positive_contacts)

        # contacts contacted
        #all_contacts = np.union1d(contacts, another_contacts)
        #        all_contacts = np.array(list(set(chain(contacts, another_contacts))))
        all_contacts = np.array(list(contacts.union(another_contacts)))
        # if len(all_contacts) > 0:
        #     # filter out already detected
        #     all_contacts = all_contacts[self.model.node_detected[all_contacts] == False]
        if cfgs.MONITOR_NODE is not None and cfgs.MONITOR_NODE in all_contacts:
            monitor(self.model.t,
                    f"is marked as contact and waits for phone call.")
        logging.debug(f"all contacts: {len(all_contacts)}")

        #all_contacts = np.array(list(set(contacts + another_contacts)))
        # print(contacts)
        # print(another_contacts)
        #all_contacts = np.union1d(contacts, another_contacts).astype(int)
        if len(all_contacts) > 0:
            all_contacts = self.depo.filter_locked(all_contacts)
            if len(all_contacts) > 0:
                # print(all_contacts)
                all_contacts = self.waiting_room_phone_call.filter_locked(
                    all_contacts)
                if len(all_contacts) > 0:
                    self.waiting_room_phone_call.lock_up(
                        all_contacts, self.phone_call_delay)
                    self.stat_contacts_collected[self.model.t] += len(
                        all_contacts)

    def enter_test_procedure(self, contacts):
        """
        Process enter tests. 
        `contacts` will be registred for test and wait for it, those
        who waited enough are tested.
        Returns nodes detected during the enter test (as np.array).
        """
        if not self.enter_test:
            return np.array([])

        to_be_tested = self.waiting_room_enter_test.tick_and_get_released()
        to_be_tested, _ = self.filter_dead(to_be_tested)
        if cfgs.MONITOR_NODE is not None and cfgs.MONITOR_NODE in to_be_tested:
            monitor(self.model.t,
                    f"goes for enter test.")

        contacts = self.waiting_room_enter_test.filter_locked(contacts)
        enter_test_delay = np.clip(
            (self.last_contact[contacts] - self.model.t) +
            self.enter_test_delay,
            1,
            self.enter_test_delay
        )

        self.waiting_room_enter_test.lock_up(contacts, enter_test_delay)
        if cfgs.MONITOR_NODE is not None and cfgs.MONITOR_NODE in contacts:
            monitor(self.model.t,
                    f"is going to wait for enter test for {enter_test_delay[np.where(contacts == cfgs.MONITOR_NODE)[0][0]]}.")

        healthy, ill = self.perform_test(to_be_tested)
        if len(healthy) > 0:
            self.negative_enter_test[healthy] = True

        if len(ill) > 0:
            self.stat_positive_enter_tests[self.model.t] += len(ill)

        return ill

    def leaving_procedure(self, nodes):
        """
        Process nodes that should be released from isolation/quarantine.
        If not auto recover enabled, nodes are tested and those tested possitively 
        are sent to quarantine/isolation for another two days.
        """

        if not self.auto_recover:
            if cfgs.MONITOR_NODE is not None and cfgs.MONITOR_NODE in nodes:
                monitor(self.model.t,
                        f"goes for leaving test.")

            nodes, dead = self.filter_dead(nodes)
            recovered, still_ill = self.perform_test(nodes)

            if len(still_ill) > 0:
                self.negative_enter_test[still_ill] = False
                self.depo.lock_up(still_ill, 2)

            if cfgs.MONITOR_NODE is not None and cfgs.MONITOR_NODE in still_ill:
                monitor(self.model.t,
                        f"will wait in quarantine/isolation another 2 days.")

            if len(recovered) > 0:
                to_release = recovered[self.negative_enter_test[recovered]]
                to_retest = recovered[self.negative_enter_test[recovered] == False]
            else:
                to_release = []
                to_retest = []

            if cfgs.MONITOR_NODE is not None and cfgs.MONITOR_NODE in to_release:
                monitor(self.model.t,
                        f"leaves qurantine/isolation with negative enter test and negative leaving test.")
            if cfgs.MONITOR_NODE is not None and cfgs.MONITOR_NODE in to_retest:
                monitor(self.model.t,
                        f"waits for second leaving test.")

            released = self.waiting_room_second_test.tick_and_get_released()
            if cfgs.MONITOR_NODE is not None and cfgs.MONITOR_NODE in released:
                monitor(self.model.t,
                        f"leaves quarantine/isolation with two negative leaving tests.")

            if len(to_retest) > 0:
                self.waiting_room_second_test.lock_up(to_retest, 2)

            if len(to_release) > 0:
                released = np.union1d(released, to_release)

        else:
            # auto recover
            released = nodes
            dead = None
            self.node_active_case[released] = False

        if len(released) > 0:
            logging.info(f"releasing {len(released)} nodes from isolation")
            self.release_nodes(released)
            self.negative_enter_test[released] = False  # for next time

        if dead is not None and len(dead) > 0:
            logging.info(f"releasing {len(released)} dead nodes from isolation")
            self.release_nodes(dead)
            self.node_active_case[dead] = False

    def run(self):
        self.stat_positive_enter_tests[self.model.t] = 0
        self.stat_contacts_collected[self.model.t] = 0
        super().run()

    def to_df(self):
        df = super().to_df()

        df["positive_enter_test"] = self.stat_positive_enter_tests[:self.model.t+1]
        df["contacts_collected"] = self.stat_contacts_collected[:self.model.t+1]

        return df


# variants - should be done using config

class CRLikePolicy(ContactTracingPolicy):

    def __init__(self, graph, model, config_file=None):
        super().__init__(graph, model, config_file=None)  # run it without loading config
        self.riskiness = get_riskiness(graph, 1, 0.1, 0.01)
        self.enter_test = True

        self.auto_recover = False
        # reconfig
        for date in 10, 66, 125, 188, 300, 311:
            if self.model.T >= date:
                self.reconfig(date)

        # if config file provided, override the setup
        # do not forget about reconfig
        if config_file is not None:
            self.load_config(config_file)

    def reconfig(self, day):

        if day == 66:
            self.riskiness = get_riskiness(self.graph, 1.0, 0.8, 0.4)
            self.enter_test = True

        if day == 10:
            self.riskiness = get_riskiness(self.graph, 1.0, 0.6, 0.2)

        if day == 125:
            self.riskiness = get_riskiness(self.graph, 0.8, 0.6, 0.2)
            self.auto_recover = True

        if day == 188:
            #            self.riskiness = get_riskiness(0.6, 0.4, 0.1)
            self.riskiness = get_riskiness(self.graph, 0.5, 0.1, 0.05)
           # self.riskiness = get_riskiness(0, 0, 0)
            self.auto_recover = True

        # if day == 300:
        #     self.riskiness = get_riskiness(0.0, 0.0, 0.0)
        #     self.auto_recover = True

        # if day == 311:
        #     self.riskiness = get_riskiness(0.0, 0.0, 0.0)
        #     self.auto_recover = True

    def run(self):
        # if self.model.T == 218:
        #     self.riskiness = get_riskiness(0.8, 0.4, 0.2)
        self.reconfig(self.model.T)
        super().run()


# do not use these classes (abandoned), use config file
class StrongEvaQuarantinePolicy(ContactTracingPolicy):

    def __init__(self, graph, model):
        super().__init__(graph, model)
        self.riskiness = RISK_FOR_LAYERS_MAX


class NoEvaQuarantinePolicy(ContactTracingPolicy):

    def __init__(self, graph, model):
        super().__init__(graph, model)
        self.riskiness = get_riskiness(self.graph, 0.0, 0.0, 0.0, 0.0)


class MiniEvaQuarantinePolicy(ContactTracingPolicy):

    def __init__(self, graph, model):
        super().__init__(graph, model)
        self.riskiness = RISK_FOR_LAYERS_MINI


class Exp2AQuarantinePolicy(ContactTracingPolicy):

    def __init__(self, graph, model):
        super().__init__(graph, model)
        self.riskiness = get_riskiness(self.graph, 1.0, 1.0, 1.0, 0.0)


class Exp2BQuarantinePolicy(ContactTracingPolicy):

    def __init__(self, graph, model):
        super().__init__(graph, model)
        self.riskiness = get_riskiness(self.graph, 1.0, 1.0, 0.0, 0.0)


class Exp2CQuarantinePolicy(ContactTracingPolicy):

    def __init__(self, graph, model):
        super().__init__(graph, model)
        self.riskiness = get_riskiness(self.graph, 1.0, 0.0, 0.0, 0.0)


class W10QuarantinePolicy(ContactTracingPolicy):

    def __init__(self, graph, model):
        super().__init__(graph, model)
        self.riskiness = get_riskiness(self.graph, 0.1, 0.1, 0.1)


class W20QuarantinePolicy(ContactTracingPolicy):

    def __init__(self, graph, model):
        super().__init__(graph, model)
        self.riskiness = get_riskiness(self.graph, 0.2, 0.2, 0.2)


class W30QuarantinePolicy(ContactTracingPolicy):

    def __init__(self, graph, model):
        super().__init__(graph, model)
        self.riskiness = get_riskiness(self.graph, 0.3, 0.3, 0.3)


class W40QuarantinePolicy(ContactTracingPolicy):

    def __init__(self, graph, model):
        super().__init__(graph, model)
        self.riskiness = get_riskiness(self.graph, 0.4, 0.4, 0.4)


class W60QuarantinePolicy(ContactTracingPolicy):

    def __init__(self, graph, model):
        super().__init__(graph, model)
        self.riskiness = get_riskiness(self.graph, 0.6, 0.6, 0.6)


class W80QuarantinePolicy(ContactTracingPolicy):

    def __init__(self, graph, model):
        super().__init__(graph, model)
        self.riskiness = get_riskiness(self.graph, 0.8, 0.8, 0.8)


def _riskiness(contact, graph, riskiness):
    #    print(f"DBG riskiness {graph.e_types[contact]}:{riskiness[graph.get_layer_for_edge(contact)]}")
    return riskiness[graph.get_layer_for_edge(contact)]
