import numpy as np
import pandas as pd
from policies.policy import Policy
from utils.history_utils import TimeSeries
import logging

from models.agent_based_network_model import STATES
from utils.config_utils import ConfigFile
from utils.graph_utils import compute_mean_degree

logging.basicConfig(level=logging.DEBUG)


class BasicSchoolPolicy(Policy):

    """
    BasicSchoolPolicy takes care of switching work days 
    and weekend.

    """

    def __init__(self, graph, model, config_file=None):
        super().__init__(graph, model)

        self.first_day = True
        self.stopped = False

        self.cf = ConfigFile()
        if config_file is not None:
            self.cf.load(config_file)

        # all layers will be turned off for weekend 
        self.mask_all_layers = {
            i: 0
            for i in range(len(self.graph.layer_weights))
        }
        self.back_up_layers = None

        # todo .. let it be a part of a graph 
        # ZS:       
        layers_apart_school =  [5, 6, 12] + list(range(41,72))
        #layers_apart_school =  [2, 7, 11] 
        self.school_layers = [
            x
            for x in range(len(self.graph.layer_weights))
            if x not in layers_apart_school
        ]
    
        
    def first_day_setup(self):
        # move teachers to R (just for one exp)
        # teachers = self.graph.nodes[self.graph.nodes_age >= 20]
        # self.model.move_to_R(teachers)

        # switch off all layers till day 35
        # ! be careful about colision with layer calendar 
        self.first_day_back_up = self.graph.layer_weights.copy()
        self.graph.set_layer_weights(self.mask_all_layers.values())

                
    def stop(self):
        """ just finish necessary, but do nothing new """
        self.stopped = True
    
    def run(self):

        if self.first_day:
            self.first_day_setup()
            self.first_day = False

            
        logging.info(
            f"Hello world! This is the {self.__class__.__name__} function speaking.  {'(STOPPED)' if self.stopped else ''}")

        if self.model.t % 7 == 5:
            logging.info("Start weekend, closing.")
            self.back_up_layers = self.graph.layer_weights.copy()
            self.graph.set_layer_weights(self.mask_all_layers.values())

        if self.model.t % 7 == 0:
            logging.info("End weekend, opening.")
            assert self.back_up_layers is not None, "Do not start the policy during weekend!" 
            self.graph.set_layer_weights(self.back_up_layers)

        if self.model.t == 35: # 35 is sunday! run it after end of weekend 
            self.graph.set_layer_weights(self.first_day_back_up)
            #print(f"t={self.model.t}")
            #print(self.graph.layer_weights)
            #exit()


    
        # if self.model.t % 7 == 1:
        #    # print every week the mean degree of second group
        #    students = self.graph.nodes[self.graph.nodes_age < 20]
        #    mean_degree = compute_mean_degree(self.graph, students)
        #    logging.debug(f"Day {self.model.t}: Mean degree of a student {mean_degree}")

            


class ClosePartPolicy(BasicSchoolPolicy):

    """
    ClosePartPolicy enables to close classes listed in config file.
    
    """

    def convert_class(self, a):
        _convert = np.vectorize(lambda x: self.graph.cat_table["class"][x])
        return  _convert(a)

    def _nodes_in_classes(self, list_of_classes):
        # todo - save node_classes? not to convert every time 
        node_classes = self.convert_class(self.graph.nodes_class)
        return self.graph.nodes[np.isin(node_classes, list_of_classes)]
        
        
    def nodes_to_quarantine(self, list_of_classes):
        """ Put all nodes belonging to listed classes to quarantine. """ 
        self.nodes_to_close = self._nodes_in_classes(list_of_classes)
        #self.graph.modify_layers_for_nodes(self.nodes_to_close,
        #                                   self.mask_all_layers)
        edges_to_close = self.graph.get_nodes_edges_on_layers(
            self.nodes_to_close,
            self.school_layers
        )
        self.graph.switch_off_edges(edges_to_close)

    def nodes_from_quarantine(self, list_of_classes):
        """ Releases all nodes belonging to listed classes from quarantine. """ 
        self.nodes_to_release = self._nodes_in_classes(list_of_classes)
        #        self.graph.recover_edges_for_nodes(self.nodes_to_release)
        edges_to_release = self.graph.get_nodes_edges_on_layers(
            self.nodes_to_release,
            self.school_layers
        )
        self.graph.switch_on_edges(edges_to_release)
        
        
    def first_day_setup(self):

        super().first_day_setup()


        close_teachers = self.cf.section_as_dict("CLOSED").get("close_teachers", "No")
        if close_teachers == "Yes":
            teachers = self.graph.nodes[self.graph.nodes_age >= 20]
            edges_to_close = self.graph.get_nodes_edges_on_layers(
                teachers,
                self.school_layers
            )
            self.graph.switch_off_edges(edges_to_close)
        
        
        # move teachers to R (just for one exp)
        #teachers = self.graph.nodes[self.graph.nodes_age >= 20]
        #self.model.move_to_R(teachers)

        # classes listed in config file goes to quarantine 
        classes_to_close = self.cf.section_as_dict(
            "CLOSED").get("classes", list())        
        if len(classes_to_close)>0:
            logging.info(f"Closing classes {classes_to_close}")
            self.nodes_to_quarantine(classes_to_close)
        else:
            logging.info("No classes clossed.")
        
class AlternatingPolicy(ClosePartPolicy):

    def __init__(self, graph, model, config_file=None):
        super().__init__(graph, model, config_file)
                                 
        group1 = self.cf.section_as_dict(
            "ALTERNATE").get("group1", list())
        group2 = self.cf.section_as_dict(
            "ALTERNATE").get("group2", list())

        self.groups = (group1, group2)
        self.passive_group, self.active_group  = 1, 0
        self.nodes_to_quarantine(self.groups[self.passive_group])

    def run(self):
        """ closes everything for the weekend, alternates classes on the second level """

        super().run()

        if self.model.t % 7 == 0:
            self.passive_group, self.active_group  = self.active_group, self.passive_group

            print("Active group: ", self.groups[self.active_group]) 
            print("Passive group: ", self.groups[self.passive_group]) 
            
            self.nodes_from_quarantine(self.groups[self.active_group])
            self.nodes_to_quarantine(self.groups[self.passive_group])
            
            logging.info(f"Day {self.model.t}: Groups changed. Active group is {self.active_group}")


        # if self.model.t % 7 == 1:
        #    # print every week the mean degree of second group
        #    group2 = self._nodes_in_classes(self.groups[1])
        #    mean_degree = compute_mean_degree(self.graph, group2)
        #    logging.debug(f"Day {self.model.t}: Mean degree of group2 {mean_degree}")

        #    group1 = self._nodes_in_classes(self.groups[0])
        #    mean_degree = compute_mean_degree(self.graph, group1)
        #    logging.debug(f"Day {self.model.t}: Mean degree of group1 {mean_degree}")
            
            
