import numpy as np
import pandas as pd
from testing_policy import TestingPolicy
from history_utils import TimeSeries
import logging
from pprint import pprint 

from agent_based_network_model import STATES


from depo import Depo
from quarantine_coefs import QUARANTINE_COEFS, RISK_FOR_LAYERS, RISK_FOR_LAYERS_MAX, RISK_FOR_LAYERS_MINI, RISK_FOR_LAYERS_60, RISK_FOR_LAYERS_10, RISK_FOR_LAYERS_30
from quarantine_coefs import get_riskiness

from global_configs import monitor
import global_configs as cfgs


logging.basicConfig(level=logging.DEBUG)


def _select_by_bitmap(x_list, flags):
    return [
        x
        for i, x in enumerate(x_list)
        if flags[i] == 1
        ]

class ContactTracingPolicy(TestingPolicy):

    """
    Testing + ContactTracing 
    """

    def __init__(self, graph, model):
        super().__init__(graph, model)

        self.coefs = QUARANTINE_COEFS
        self.riskiness = get_riskiness(1.0, 0.8, 0.4)
        self.duration = 14
        self.days_back = 7
        self.phone_call_delay = 2
        self.enter_test_range = [3, 4, 5]
        self.negative_enter_test = np.zeros(self.model.num_nodes, dtype=bool)


        self.enter_test = True

        logging.info(f"I'm CONTACT TRACING POLICY. My riskiness is {self.riskiness}")

        # nodes marked as contacts and waiting for being informed
        self.waiting_room_phone_call = Depo(graph.number_of_nodes)

        # nodes waiting for enter-test
        self.waiting_room_enter_test = Depo(graph.number_of_nodes)

        # nodes waiting for enter-test result
        self.waiting_room_result_enter_test = Depo(graph.number_of_nodes)

        # nodes waiting for second test
        self.waiting_room_second_test = Depo(graph.number_of_nodes)

    def filter_contact_history(self, detected_nodes):

        
        #days_back = self.days_back
        

        days_back_array = np.clip(
            (self.model.t - self.first_symptoms[detected_nodes]) + 2,
            0,
            self.days_back
            )

        if cfgs.MONITOR_NODE is not None and cfgs.MONITOR_NODE in detected_nodes:
            index = list(detected_nodes).index(cfgs.MONITOR_NODE)
            monitor(self.model.t,
                    f" being asked for contacts, days back = {days_back_array[index]}.") 


        relevant_contacts = [] 
        for i, d_node in enumerate(detected_nodes):
            days_back = days_back_array[i] 

            nodes_contacts = [
                (contact[1], 
                 _riskiness(contact[2], self.graph, self.riskiness), 
                 (self.model.t - days_back +i +1)
                 )
                # five day back
                for i, contact_list in enumerate(self.model.contact_history[-days_back:])
                for contact in contact_list
                if contact[0] == d_node 
                ]
            relevant_contacts.extend(nodes_contacts)

        if cfgs.MONITOR_NODE is not None and cfgs.MONITOR_NODE in [x for x, _, _ in relevant_contacts]:
            dates = [
                d
                for x, _, d in relevant_contacts
                if x == cfgs.MONITOR_NODE
                ]
            monitor(self.model.t,
                    f" is a contact of detected node (contact date: {dates}).") 

        # TODO check time from start of E, if lower than five, do lottery

        logging.info(f"DBG QE {type(self).__name__}: all contacts {len(relevant_contacts)}")

        if not relevant_contacts:
            return [], []

        r = np.random.rand(len(relevant_contacts))
        selected_contacts = [
            (contact, d)
            for (contact, threashold, d), r_number in zip(relevant_contacts, r)
            if r_number < threashold
        ]
        
        ret = {}
        for x, d in selected_contacts:
            if x in ret: 
                ret[x].append(d) 
            else:
                ret[x] = [d] 

        ret_nodes = list(ret.keys())
        ret_dates = [
            max(ret[x]) 
            for x in ret_nodes
            ]

        logging.info(f"DBG QE {type(self).__name__}: selected contacts {len(ret.keys())}")
        return ret_nodes, ret_dates 

    def select_contacts(self, detected_nodes):
        return self.filter_contact_history(detected_nodes)

    def to_df(self):
        # todo
        return super().to_df()

    def stop(self):
        """ just finish necessary, but do not qurantine new nodes """
        self.stopped = True

    def do_contact_tracing(self, detected_nodes):


        if len(detected_nodes) > 0:

            self.last_contact[detected_nodes] = self.model.t # these are self-tests, full isolation

            detected_nodes = self.nodes[detected_nodes]

            detected_nodes = list(self.depo.filter_locked(detected_nodes))
            
            #print(detected_nodes) 
                        
            new_contacts, dates = self.select_contacts(detected_nodes) # returns list of tuples 
                        
            if cfgs.MONITOR_NODE is not None and cfgs.MONITOR_NODE in new_contacts:
                date = dates[new_contacts.index(cfgs.MONITOR_NODE)]
                monitor(self.model.t, 
                        f" is a contact and will be contacted by a phone call. date:{date}")

        else:
            new_contacts, dates = [], []
        
        

        # get rid of days
        logging.info(f"{self.model.t} DBG QE: Qurantined nodes: {len(detected_nodes)}")
        logging.info(f"{self.model.t} DBG QE: Found contacts: {len(new_contacts)}")

        # get contats to be quarantined (except those who are already in quarantine)
        contacts_ready_for_quarantine =  self.waiting_room_phone_call.tick_and_get_released()
        if cfgs.MONITOR_NODE is not None and cfgs.MONITOR_NODE in contacts_ready_for_quarantine:
            monitor(self.model.t, 
                    f" recieves a phone call.")
        contacts_ready_for_quarantine = self.depo.filter_locked(contacts_ready_for_quarantine)
        if cfgs.MONITOR_NODE is not None and cfgs.MONITOR_NODE in contacts_ready_for_quarantine:
            monitor(self.model.t, 
                    f" goes to quarantine.")
        
        # filter out those who are already waiting 
        new_contacts_bits = self.waiting_room_phone_call.filter_locked_bitmap(new_contacts)
        new_contacts = _select_by_bitmap(new_contacts, new_contacts_bits)
        dates = _select_by_bitmap(dates, new_contacts_bits)
        # TODO: sole if waiting with older contact 
        self.last_contact[new_contacts] = dates
        self.waiting_room_phone_call.lock_up(
            new_contacts, self.phone_call_delay)

        logging.info(
            f"{self.model.t} DBG QE: Quaratinted contacts: {len(contacts_ready_for_quarantine)}")

        # contacts wait for 5th day test and collect those who should be tested today
        if self.enter_test:
            nodes_to_be_tested = self.waiting_room_enter_test.tick_and_get_released()
            
            if cfgs.MONITOR_NODE is not None and cfgs.MONITOR_NODE in nodes_to_be_tested:
                monitor(self.model.t, 
                        f" goes to enter-quarantine test.")
                
            waiting_time = np.random.choice(self.enter_test_range,
                                            size=len(
                    contacts_ready_for_quarantine))
            # waiting tie -  days already past 
            # time is at least 1 
            
            #if len(contacts_ready_for_quarantine) > 0:
            #    x = self.model.t - self.last_contact[contacts_ready_for_quarantine] 
            #    print(contacts_ready_for_quarantine) 
            #    print(self.model.t) 
            #    print(self.last_contact[contacts_ready_for_quarantine])
            #    print(x)
            #    exit()

            waiting_time = np.clip(
                ( waiting_time 
                  - (self.model.t - self.last_contact[contacts_ready_for_quarantine])    
                  ), 
                1, None
                )

            if cfgs.MONITOR_NODE is not None and cfgs.MONITOR_NODE in contacts_ready_for_quarantine:
                index = list(contacts_ready_for_quarantine).index(cfgs.MONITOR_NODE)
                monitor(self.model.t, 
                        f"waits for enter-test for {waiting_time[index]} days")
                
            self.waiting_room_enter_test.lock_up(contacts_ready_for_quarantine,
                                                 waiting_time
                                                 )
            # waiting for test do not go for test itself
            self.testable[contacts_ready_for_quarantine, 0] = False
        

            # do testing
            healthy, ill = self.test_check(nodes_to_be_tested)
            if len(healthy) > 0: # healthy numpy array, treated as list 
                self.negative_enter_test[healthy] = True 

            if cfgs.MONITOR_NODE is not None and cfgs.MONITOR_NODE in healthy:
                monitor(self.model.t, 
                        f" had negative test.")
            if cfgs.MONITOR_NODE is not None and cfgs.MONITOR_NODE in ill:
                monitor(self.model.t, 
                        f" had positive test.")

            # we do not care about healthy, ill waits for test results
            contacts_positively_tested = self.waiting_room_result_enter_test.tick_and_get_released()
            self.waiting_room_result_enter_test.lock_up(ill, 2)
            if cfgs.MONITOR_NODE is not None and cfgs.MONITOR_NODE in contacts_positively_tested:
                monitor(self.model.t, 
                        f" recieves the positive test result.")
            
            # select contacts of positively tested
            other_contacts, dates = self.select_contacts(
                contacts_positively_tested)
            if cfgs.MONITOR_NODE is not None and cfgs.MONITOR_NODE in other_contacts:
                date = dates[other_contacts.index(cfgs.MONITOR_NODE)]
                monitor(self.model.t, 
                        f" is a contact and will be contacted by a phone call. date{date}.")
            if other_contacts:
                # filter out those who are already waiting 
                other_contacts_bits = self.waiting_room_phone_call.filter_locked_bitmap(other_contacts)
                other_contacts = _select_by_bitmap(other_contacts, other_contacts_bits)
                dates = _select_by_bitmap(dates, other_contacts_bits)
                # TODO: sole if waiting with older contact 
                self.last_contact[other_contacts] =  dates

                self.waiting_room_phone_call.lock_up(
                    other_contacts, self.phone_call_delay, check_duplicate=True)

        # quarantine opens doors - those who spent the 14 days are released,
        # newly detected + contacted contacts are locked up
        released = self.tick()
        self.quarantine_nodes(
            list(detected_nodes)+list(contacts_ready_for_quarantine),
            )

        # do final testing
        # two positive tests are needed for leaving (tests are always correct!)
        if cfgs.MONITOR_NODE is not None and cfgs.MONITOR_NODE in released:
                monitor(self.model.t, 
                        f" goes for final test.")
        second_test_candidates, still_ill = self.test_check(released)
        if cfgs.MONITOR_NODE is not None and cfgs.MONITOR_NODE in second_test_candidates:
                monitor(self.model.t, 
                        f" was negative and waits for the second test.")
        if cfgs.MONITOR_NODE is not None and cfgs.MONITOR_NODE in still_ill:
                monitor(self.model.t, 
                        f" is still positive.")

        if len(still_ill) > 0:
            self.node_detected[still_ill] = True
            self.node_active_case[still_ill] = True
        #        self.node_active_case[negative_nodes] = False #for now second test candinate

        # still ill nodes go back to quarantine
        if len(still_ill) > 0:
            self.depo.lock_up(still_ill, 2)
            # enter test is not valid anymore
            self.negative_enter_test[still_ill] = False 
        # healthy wait for the second test, those tested second time released
        ready_to_leave = self.waiting_room_second_test.tick_and_get_released()
        # filter those who already passed negative test 
        #second_test_candidates, numpy array treated as list 
        if len(second_test_candidates)>0:
            with_negative_test_mask = self.negative_enter_test[second_test_candidates]
            ready_to_leave_too = second_test_candidates[with_negative_test_mask] 
            still_second_test_candidates = second_test_candidates[with_negative_test_mask == False] 
            second_test_candidates = still_second_test_candidates
        # nebylo by rychlejsi udelat listy? a extend
            ready_to_leave = np.union1d(ready_to_leave, ready_to_leave_too)

        if len(second_test_candidates) > 0:
            self.waiting_room_second_test.lock_up(second_test_candidates, 2)
            #       self.model.num_qtests[self.model.t] += len(ready_to_leave)

        if cfgs.MONITOR_NODE is not None and cfgs.MONITOR_NODE in ready_to_leave:
                monitor(self.model.t, 
                        f" passes second test and leaves isolation/quarantine.")


        #self.node_active_case[ready_to_leave] = False #for now second test candinate

        self.release_nodes(ready_to_leave)
        self.node_active_case[ready_to_leave] = False 
        # leaves the system, reset
        self.negative_enter_test[ready_to_leave] = False

    def run(self):


        if self.first_day:
            self.first_day_setup() 

        logging.info(
            f"Hello world! This is the CONTACT TRACING POLICY function speaking.  {'(STOPPED)' if self.stopped else ''}")

        # reset those who stopped be positive 
        target_nodes = np.logical_and(
            self.model.memberships[STATES.S],
            self.testable != -1
            )
        self.testable[target_nodes] = -1
        
        # TODO: move this
        # get all symptomatic nodes
        target_nodes = np.logical_or(
            self.model.memberships[STATES.S_s],
            self.model.memberships[STATES.I_s]
        )
        n = target_nodes.sum()

        if cfgs.MONITOR_NODE is not None and target_nodes[cfgs.MONITOR_NODE,0] == True:
            monitor(self.model.t, 
                    f"is target node, testable {self.testable[cfgs.MONITOR_NODE]}")


        new_symptomatic = np.logical_and(
            target_nodes,
            self.testable == -1
        )
        if cfgs.MONITOR_NODE is not None and new_symptomatic[cfgs.MONITOR_NODE,0] == True:
            monitor(self.model.t, 
                    f"becomes symptomatic")

        if new_symptomatic.sum() > 0:
            r = np.random.rand(new_symptomatic.sum())
            self.testable[new_symptomatic] = r < self.model.test_rate[new_symptomatic]
            if cfgs.MONITOR_NODE is not None and new_symptomatic[cfgs.MONITOR_NODE,0] == True :
                if self.testable[cfgs.MONITOR_NODE]:
                    monitor(self.model.t,
                            f"plans to go for test")
                else:
                    monitor(self.model.t,
                            f"does not plan to go for test")
                    

        possitive_nodes = [] 
        if n == 0:
            logging.info(
                f"TESTING: no symptomatic nodes")
            self.stat_positive_tests[self.model.t] = 0
            self.stat_negative_tests[self.model.t] = 0

        else:
            tested, possitive_nodes, negative_nodes = self.do_testing(
                target_nodes, n)
            if cfgs.MONITOR_NODE:
                if cfgs.MONITOR_NODE in possitive_nodes:
                    monitor(self.model.t,
                            f"has positive test result.")
                if cfgs.MONITOR_NODE in negative_nodes:                    
                    monitor(self.model.t,
                            f"has negative test result.")

        self.stat_Id[self.model.t] = self.node_active_case.sum()
        self.stat_cum_Id[self.model.t] = self.node_detected.sum()

        print(f"t = {self.model.t} ACTIVE CASES: {self.stat_Id[self.model.t]}")


        self.do_contact_tracing(possitive_nodes)


# variants (TODO: do this by config)

class CRLikePolicy(ContactTracingPolicy):

    def __init__(self, graph, model):
        super().__init__(graph, model)
        self.riskiness = get_riskiness(1, 0.1, 0.01)
        self.enter_test = True

    def run(self): 
        if self.model.t == 66:
            self.riskiness = get_riskiness(1.0, 0.8, 0.4)
            self.enter_test = True
        if self.model.t == 10:
            self.riskiness = get_riskiness(1.0, 0.6, 0.2)
        super().run()


class StrongEvaQuarantinePolicy(ContactTracingPolicy):

    def __init__(self, graph, model):
        super().__init__(graph, model)
        self.riskiness = RISK_FOR_LAYERS_MAX


class NoEvaQuarantinePolicy(ContactTracingPolicy):

    def __init__(self, graph, model):
        super().__init__(graph, model)
        self.riskiness = get_riskiness(0.0,0.0,0.0,0.0)



class MiniEvaQuarantinePolicy(ContactTracingPolicy):

    def __init__(self, graph, model):
        super().__init__(graph, model)
        self.riskiness = RISK_FOR_LAYERS_MINI


class Exp2AQuarantinePolicy(ContactTracingPolicy):

    def __init__(self, graph, model):
        super().__init__(graph, model)
        self.riskiness = get_riskiness(1.0, 1.0, 1.0, 0.0)

class Exp2BQuarantinePolicy(ContactTracingPolicy):

    def __init__(self, graph, model):
        super().__init__(graph, model)
        self.riskiness = get_riskiness(1.0, 1.0, 0.0, 0.0)

class Exp2CQuarantinePolicy(ContactTracingPolicy):

    def __init__(self, graph, model):
        super().__init__(graph, model)
        self.riskiness = get_riskiness(1.0, 0.0, 0.0, 0.0)


	

class W10QuarantinePolicy(ContactTracingPolicy):

    def __init__(self, graph, model):
        super().__init__(graph, model)
        self.riskiness = get_riskiness(0.1, 0.1, 0.1)


class W20QuarantinePolicy(ContactTracingPolicy):

    def __init__(self, graph, model):
        super().__init__(graph, model)
        self.riskiness = get_riskiness(0.2, 0.2, 0.2)

class W30QuarantinePolicy(ContactTracingPolicy):

    def __init__(self, graph, model):
        super().__init__(graph, model)
        self.riskiness = get_riskiness(0.3, 0.3, 0.3)


class W40QuarantinePolicy(ContactTracingPolicy):

    def __init__(self, graph, model):
        super().__init__(graph, model)
        self.riskiness = get_riskiness(0.4, 0.4, 0.4)

class W60QuarantinePolicy(ContactTracingPolicy):

    def __init__(self, graph, model):
        super().__init__(graph, model)
        self.riskiness = get_riskiness(0.6, 0.6, 0.6)


class W80QuarantinePolicy(ContactTracingPolicy):

    def __init__(self, graph, model):
        super().__init__(graph, model)
        self.riskiness = get_riskiness(0.8, 0.8, 0.8)

def _riskiness(contact, graph, riskiness):
#    print(f"DBG riskiness {graph.e_types[contact]}:{riskiness[graph.get_layer_for_edge(contact)]}") 
    return riskiness[graph.get_layer_for_edge(contact)]
