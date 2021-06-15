import numpy as np
import pandas as pd
import logging
from itertools import chain

from policies.testing_policy import TestingPolicy
from utils.history_utils import TimeSeries
from models.agent_based_network_model import STATES


from policies.depo import Depo
from policies.quarantine_coefs import QUARANTINE_COEFS, RISK_FOR_LAYERS, RISK_FOR_LAYERS_MAX, RISK_FOR_LAYERS_MINI, RISK_FOR_LAYERS_60, RISK_FOR_LAYERS_10, RISK_FOR_LAYERS_30
from policies.quarantine_coefs import get_riskiness

from utils.global_configs import monitor
import utils.global_configs as cfgs


logging.basicConfig(level=logging.DEBUG)



class ContactTracingPolicy(TestingPolicy):

    """
    Testing + ContactTracing 
    """

    def __init__(self, graph, model):
        super().__init__(graph, model)

        if graph.QUARANTINE_COEFS is not None:
            self.coefs = graph.QUARANTINE_COEFS
        else:
            self.coefs = QUARANTINE_COEFS
        self.riskiness = get_riskiness(1.0, 0.8, 0.4)
        self.duration = 10
        self.duration_quara = 14
        self.days_back = 7
        self.phone_call_delay = 2
        self.enter_test_delay = 5

        self.negative_enter_test = np.zeros(self.model.num_nodes, dtype=bool)
        self.last_contact = np.full(self.model.num_nodes, fill_value=-1, dtype=int)
        
        self.auto_recover = False
        self.enter_test = True

        logging.info(f"CONTACT TRACING POLICY: riskiness is {self.riskiness}")

        # self.e_risk = self.riskiness[self.graph.e_types]

        # nodes marked as contacts and waiting for being informed
        self.waiting_room_phone_call = Depo(graph.number_of_nodes)

        # nodes waiting for enter-test
        self.waiting_room_enter_test = Depo(graph.number_of_nodes)

        # nodes waiting for enter-test result
        self.waiting_room_result_enter_test = Depo(graph.number_of_nodes)

        # nodes waiting for second test
        self.waiting_room_second_test = Depo(graph.number_of_nodes)


    def select_contacts(self, detected_nodes):
        
        if len(detected_nodes) == 0:
            return set()

        days_back_array = np.clip(
            (self.model.t - self.first_symptoms[detected_nodes]) + 2,
            0,
            self.days_back
        )

        if cfgs.MONITOR_NODE is not None and cfgs.MONITOR_NODE in detected_nodes:
            monitor(self.model.t,
                    f"is beeing asked for contacts for {days_back_array[np.where(detected_nodes == cfgs.MONITOR_NODE)[0][0]]} days back.")

        contacts = set()
        for i, node in enumerate(detected_nodes):

            days_back = days_back_array[i]
            for t, day_list in enumerate(self.model.contact_history[-days_back:]):
                if day_list is None: # first iterations
                    continue 
                if len(day_list[0]) == 0:
                    continue
                # daylist is tuple (source_nodes, dest_nodes, types) including both directions 
                my_edges = day_list[0] == node 
                selected_contacts = day_list[1][my_edges] 
                types = day_list[2][my_edges] 

                selected_contacts = self.filter(selected_contacts, types)
                self.last_contact[selected_contacts] = np.maximum(self.last_contact[selected_contacts], self.model.t - days_back + t)

                contacts.update(selected_contacts)

                # forward_edges = day_list[self.graph.e_source[day_list] == node]
                # backward_edges = day_list[self.graph.e_dest[day_list] == node] 
                # forward_edges = self.filter(forward_edges)
                # backward_edges = self.filter(backward_edges)

                # c1 = self.graph.e_dest[forward_edges]
                # c2 = self.graph.e_source[backward_edges]
                # #contacts.extend(list(c1))
                # #contacts.extend(list(c2))
                # # # chain works even for mix of lists and arrays
                # contacts = chain(contacts, c1, c2)
                # #contacts = contacts + list(c1) + list(c2)
                # # contacts = np.union1d(contacts, c1) 
                # # contacts = np.union1d(contacts, c2)
                # self.last_contact[c1] = np.maximum(self.last_contact[c1], self.model.t - days_back + t)
                # self.last_contact[c2] = np.maximum(self.last_contact[c2], self.model.t - days_back + t)
        
        return contacts  #.nonzero()[0] 

    def filter(self, contacts, types):
        risk = self.riskiness[types]
        r = np.random.rand(len(risk))
        return contacts[r<risk]

    # def filter(self, edges):
        # e_types = self.graph.e_types[edges] 
        # risk = self.riskiness[e_types] 
        # risk = self.e_risk[edges]
        # r = np.random.rand(len(risk))
        # collected_edges = edges[r<risk] 
        # return collected_edges

    def to_df(self):
        # todo
        return super().to_df()

    def stop(self):
        """ just finish necessary, but do not qurantine new nodes """
        self.stopped = True


    def select_test_candidates(self):

        test_candidates = super().select_test_candidates() 
        # exclude those that are already in quarantine 
        return self.depo.filter_locked(test_candidates)

    def process_detected_nodes(self, target_nodes):

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
            contacted  = contacted[self.model.node_detected[contacted] == False]
        if cfgs.MONITOR_NODE is not None and cfgs.MONITOR_NODE in contacted:
            monitor(self.model.t,
                    f"recieves a phone call and is sent to quarantine.")
        self.quarantine_nodes(contacted, last_contacts=True, duration=self.duration_quara)


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
                #print(all_contacts)
                all_contacts = self.waiting_room_phone_call.filter_locked(all_contacts)
                if len(all_contacts) > 0:
                    self.waiting_room_phone_call.lock_up(all_contacts, self.phone_call_delay)


    def enter_test_procedure(self, contacts):
        to_be_tested = self.waiting_room_enter_test.tick_and_get_released() 
        if cfgs.MONITOR_NODE is not None and cfgs.MONITOR_NODE in to_be_tested:
            monitor(self.model.t,
                    f"goes for enter test.")
        
        contacts = self.waiting_room_enter_test.filter_locked(contacts)
        enter_test_delay = np.clip(
            (self.last_contact[contacts] - self.model.t) + self.enter_test_delay, 
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
        
        return ill
        

    def leaving_procedure(self, nodes):

        if cfgs.MONITOR_NODE is not None and cfgs.MONITOR_NODE in nodes:
            monitor(self.model.t,
                    f"goes for leaving test.")
                
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

        if len(released) > 0:
            logging.info(f"releasing {len(released)} nodes from isolation")
            self.release_nodes(released)
            self.negative_enter_test[released] = False # for next time 
        

        
# variants (TODO: do this by config)

class CRLikePolicy(ContactTracingPolicy):

    def __init__(self, graph, model):
        super().__init__(graph, model)
        self.riskiness = get_riskiness(1, 0.1, 0.01)
        self.enter_test = True
        
        self.auto_recover = False 
        # reconfig
        for date in 10, 66, 125, 188, 300, 311: 
            if self.model.T >= date: 
                self.reconfig(date)


    def reconfig(self, day):

        if day == 66:
            self.riskiness = get_riskiness(1.0, 0.8, 0.4)
            self.enter_test = True
            
        if day == 10:
            self.riskiness = get_riskiness(1.0, 0.6, 0.2)
            
        if day == 125:
            self.riskiness = get_riskiness(0.8, 0.6, 0.2)
            self.auto_recover = True
        
        if day == 188:
#            self.riskiness = get_riskiness(0.6, 0.4, 0.1)
            self.riskiness = get_riskiness(0.5, 0.1, 0.05)
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
