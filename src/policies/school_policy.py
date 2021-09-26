# NOTE: this policy is intended to be run with a special graph for schools!!!!
# Do not use it for normal graphs (hodoninsko, lounsko, papertown, etc).
import global_configs as cfgs
from global_configs import monitor
from depo import Depo
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

    def __init__(self, graph, model, config_file=None, config_obj=None):
        super().__init__(graph, model)

        self.weekend_start = 5
        self.weekend_end = 0

        self.first_day = True
        self.stopped = False
        self.testing = False
        self.test_sensitivity = 0.4
        self.test_days = (0, 2)
        self.test_groups = None

        self.at_school = np.ones(self.graph.num_nodes, dtype=bool)
        self.nodes = np.arange(self.graph.num_nodes)
        teachers = self.graph.nodes_age >= 20
        self.at_school[teachers] = 0  # do not test teachers

        self.cf = ConfigFile()
        if config_file is not None:
            self.cf.load(config_file)
            test_sec = self.cf.section_as_dict("TESTING")
            if test_sec.get("testing", "No") == "Yes":
                self.testing = True
            if "sensitivity" in test_sec:
                self.test_sensitivity = test_sec["sensitivity"]
            if "days" in test_sec:
                self.test_days = test_sec["days"]
                if not type(self.test_days) is list:
                    self.test_days = (self.test_days,)
                else:
                    self.test_days = [int(x) for x in self.test_days]

        logging.info(f"testing {self.testing}")
        logging.info(f"test sensitivity {self.test_sensitivity}")

        # all layers will be turned off for weekend
        self.mask_all_layers = {
            i: 0
            for i in range(len(self.graph.layer_weights))
        }
        self.back_up_layers = None

        # todo .. let it be a part of a graph
        # ZS:
        layers_apart_school = [5, 6, 12] + list(range(41, 72))
        #        layers_apart_school =  [2, 7, 11]
        self.school_layers = [
            x
            for x in range(len(self.graph.layer_weights))
            if x not in layers_apart_school
        ]

        self.positive_test = np.zeros(self.graph.num_nodes, dtype=bool)
        self.depo = Depo(self.graph.number_of_nodes)

        self.stat_in_quara = TimeSeries(301, dtype=int)

    def nodes_to_quarantine(self, nodes):
        self.at_school[nodes] = False
        if cfgs.MONITOR_NODE is not None and cfgs.MONITOR_NODE in nodes:
            monitor(self.model.t,
                    "goes to the stay-at-home state.")
        edges_to_close = self.graph.get_nodes_edges_on_layers(
            nodes,
            self.school_layers
        )
        self.graph.switch_off_edges(edges_to_close)

    def nodes_from_quarantine(self, nodes):
        self.at_school[nodes] = True

        if cfgs.MONITOR_NODE is not None and cfgs.MONITOR_NODE in nodes:
            monitor(self.model.t,
                    "goes to the go-to-school state.")
        edges_to_release = self.graph.get_nodes_edges_on_layers(
            nodes,
            self.school_layers
        )
        self.graph.switch_on_edges(edges_to_release)

    def first_day_setup(self):
        # # move teachers to R (just for one exp)
        # teachers = self.graph.nodes[self.graph.nodes_age >= 20]
        # #self.model.move_to_R(teachers)
        # self.nodes_to_quarantine(teachers)

        # switch off all layers till day 35
        # ! be careful about colision with layer calendar
        self.first_day_back_up = self.graph.layer_weights.copy()
        self.graph.set_layer_weights(self.mask_all_layers.values())
        self.stat_in_quara[0:self.model.t] = 0

    def do_testing(self):

        released = self.depo.tick_and_get_released()
        if len(released) > 0:
            self.graph.recover_edges_for_nodes(released)
            if cfgs.MONITOR_NODE is not None and cfgs.MONITOR_NODE in released:
                monitor(self.model.t,
                        "node released from quarantine.")

        assert len(released) == 0 or self.model.t % 7 in self.test_days

        # monday or wednesday perform tests -> do it the night before
        if self.model.t % 7 in self.test_days:

            students_at_school = np.logical_and(
                self.at_school,
                self.graph.is_quarantined == 0
            )

            if self.test_groups is not None:
                should_not_be_tested = self.test_groups[self.test_passive]

                if cfgs.MONITOR_NODE is not None and cfgs.MONITOR_NODE in should_not_be_tested:
                    monitor(self.model.t,
                            "node should NOT be tested.")

                students_at_school[should_not_be_tested] = False

            if cfgs.MONITOR_NODE is not None and students_at_school[cfgs.MONITOR_NODE]:
                monitor(self.model.t,
                        "node is at school and will be tested.")

            # at school and positive
            possibly_positive = (
                self.model.memberships[STATES.I_n] +
                self.model.memberships[STATES.I_s] +
                self.model.memberships[STATES.I_a] +
                self.model.memberships[STATES.J_n] +
                self.model.memberships[STATES.J_s]
            ).ravel()
            if cfgs.MONITOR_NODE is not None and possibly_positive[cfgs.MONITOR_NODE]:
                monitor(self.model.t,
                        "node is positive.")

            possibly_positive_test = np.logical_and(
                students_at_school,
                possibly_positive
            )
            if cfgs.MONITOR_NODE is not None and possibly_positive_test[cfgs.MONITOR_NODE]:
                monitor(self.model.t,
                        "node is positive and is tested.")

            self.positive_test.fill(0)
            num = possibly_positive_test.sum()
            r = np.random.rand(num)
            self.positive_test[possibly_positive_test] = r < self.test_sensitivity

            if cfgs.MONITOR_NODE is not None and self.positive_test[cfgs.MONITOR_NODE]:
                monitor(self.model.t,
                        "had positive tested.")

            self.depo.lock_up(self.positive_test, 7)
            self.graph.modify_layers_for_nodes(list(self.nodes[self.positive_test]),
                                               self.graph.QUARANTINE_COEFS)

            if cfgs.MONITOR_NODE is not None and self.graph.is_quarantined[cfgs.MONITOR_NODE]:
                monitor(self.model.t,
                        "is in quarantine.")

        self.stat_in_quara[self.model.t] = self.depo.num_of_prisoners

    def stop(self):
        """ just finish necessary, but do nothing new """
        self.stopped = True

    def closing_and_opening(self):
        pass

    def run(self):

        if self.first_day:
            self.first_day_setup()
            self.first_day = False

        logging.info(
            f"Hello world! This is the {self.__class__.__name__} function speaking.  {'(STOPPED)' if self.stopped else ''}")

        if self.model.t % 7 == self.weekend_start:
            logging.info("Start weekend, closing.")
            self.back_up_layers = self.graph.layer_weights.copy()
            self.graph.set_layer_weights(self.mask_all_layers.values())

        if self.model.t % 7 == self.weekend_end:
            logging.info("End weekend, opening.")
            if self.back_up_layers is None:
                logging.warning("The school policy started during weekend!")
            else:
                self.graph.set_layer_weights(self.back_up_layers)

        if (self.first_day_back_up is not None and self.model.t >= 35
                and self.model.t % 7 == self.weekend_end):  # 35 is sunday! run it after end of weekend
            self.graph.set_layer_weights(self.first_day_back_up)
            self.first_day_back_up = None  # run it only once
            # print(f"t={self.model.t}")
            # print(self.graph.layer_weights)
            # exit()

        self.closing_and_opening()

        if self.model.t >= 35 and self.testing:
            self.do_testing()

        # if self.model.t % 7 == 1:
        #    # print every week the mean degree of second group
        #    students = self.graph.nodes[self.graph.nodes_age < 20]
        #    mean_degree = compute_mean_degree(self.graph, students)
        #    logging.debug(f"Day {self.model.t}: Mean degree of a student {mean_degree}")

    def to_df(self):
        index = range(0, self.model.t+1)
        columns = {
            f"school_policy_in_quara":  self.stat_in_quara[:self.model.t+1],
        }
        columns["day"] = np.floor(index).astype(int)
        df = pd.DataFrame(columns, index=index)
        df.index.rename('T', inplace=True)
        return df


class ClosePartPolicy(BasicSchoolPolicy):

    """
    ClosePartPolicy enables to close classes listed in config file.

    """

    def convert_class(self, a):
        _convert = np.vectorize(lambda x: (
            self.graph.cat_table["class"]+[None])[x])
        return _convert(a)

    def nodes_in_classes(self, list_of_classes):
        # todo - save node_classes? not to convert every time
        node_classes = self.convert_class(self.graph.nodes_class)
        return self.graph.nodes[np.isin(node_classes, list_of_classes)]

    def classes_to_quarantine(self, list_of_classes):
        """ Put all nodes belonging to listed classes to quarantine. """
        self.nodes_to_close = self.nodes_in_classes(list_of_classes)
        self.at_school[self.nodes_to_close] = False
        if cfgs.MONITOR_NODE is not None and cfgs.MONITOR_NODE in self.nodes_to_close:
            monitor(self.model.t,
                    "goes to the stay-at-home state.")

        # self.graph.modify_layers_for_nodes(self.nodes_to_close,
        #                                   self.mask_all_layers)
        edges_to_close = self.graph.get_nodes_edges_on_layers(
            self.nodes_to_close,
            self.school_layers
        )
        self.graph.switch_off_edges(edges_to_close)

    def classes_from_quarantine(self, list_of_classes):
        """ Releases all nodes belonging to listed classes from quarantine. """
        self.nodes_to_release = self.nodes_in_classes(list_of_classes)
        #        self.graph.recover_edges_for_nodes(self.nodes_to_release)
        self.at_school[self.nodes_to_release] = True
        if cfgs.MONITOR_NODE is not None and cfgs.MONITOR_NODE in self.nodes_to_release:
            monitor(self.model.t,
                    "goes to the go-to-school state.")
        edges_to_release = self.graph.get_nodes_edges_on_layers(
            self.nodes_to_release,
            self.school_layers
        )
        self.graph.switch_on_edges(edges_to_release)

    def first_day_setup(self):

        super().first_day_setup()

        close_teachers = self.cf.section_as_dict(
            "CLOSED").get("close_teachers", "No")
        if close_teachers == "Yes":
            teachers = self.graph.nodes[self.graph.nodes_age >= 20]
            edges_to_close = self.graph.get_nodes_edges_on_layers(
                teachers,
                self.school_layers
            )
            self.graph.switch_off_edges(edges_to_close)

        # move teachers to R (just for one exp)
        #teachers = self.graph.nodes[self.graph.nodes_age >= 20]
        # self.model.move_to_R(teachers)

        # classes listed in config file goes to quarantine
        classes_to_close = self.cf.section_as_dict(
            "CLOSED").get("classes", list())
        if len(classes_to_close) > 0:
            logging.info(f"Closing classes {classes_to_close}")
            self.classes_to_quarantine(classes_to_close)
        else:
            logging.info("No classes clossed.")


class AlternatingPolicy(ClosePartPolicy):

    def __init__(self, graph, model, config_file=None):
        super().__init__(graph, model, config_file)

        rotate_class_groups = self.cf.section_as_dict(
            "ALTERNATE").get("use_class_groups", "No") == "Yes"

        if not rotate_class_groups:
            group1 = self.cf.section_as_dict(
                "ALTERNATE").get("group1", list())
            group2 = self.cf.section_as_dict(
                "ALTERNATE").get("group2", list())

            nodes_group1 = self.nodes_in_classes(group1)
            nodes_group2 = self.nodes_in_classes(group2)
            self.groups = (nodes_group1, nodes_group2)
        else:
            nodes_group1 = self.graph.nodes[self.graph.nodes_class_group == 0]
            nodes_group2 = self.graph.nodes[self.graph.nodes_class_group == 1]

            self.groups = (nodes_group1, nodes_group2)

        self.passive_group, self.active_group = 1, 0
        self.nodes_to_quarantine(self.groups[self.passive_group])

        testing_cfg = self.cf.section_as_dict("TESTING_GROUPS")
        if testing_cfg:
            use_class_groups = testing_cfg.get(
                "use_class_groups", "No") == "Yes"

            if not use_class_groups:
                group1a = testing_cfg["group1a"]
                group1b = testing_cfg["group1b"]
                group2a = testing_cfg["group2a"]
                group2b = testing_cfg["group2b"]

                groupA = self.nodes_in_classes(group1a+group2a)
                groupB = self.nodes_in_classes(group1b+group2b)

                self.test_groups = (groupA, groupB)
                self.test_passive, self.test_active = 0, 1

            else:
                groupA = self.graph.nodes[self.graph.nodes_class_group == 0]
                groupB = self.graph.nodes[self.graph.nodes_class_group == 1]

                self.test_groups = (groupA, groupB)
                self.test_passive, self.test_active = 0, 1
        else:
            self.test_groups = None

    def closing_and_opening(self):
        """ closes everything for the weekend, alternates classes on the second level """

        if self.model.t % 7 == self.weekend_end:
            self.passive_group, self.active_group = self.active_group, self.passive_group

            self.nodes_from_quarantine(self.groups[self.active_group])
            self.nodes_to_quarantine(self.groups[self.passive_group])

            logging.info(f"Day {self.model.t}: Groups changed. Active group is {self.active_group}")

        if self.model.t % 14 == self.weekend_end:
            if self.test_groups is not None:
                self.test_active, self.test_passive = self.test_passive, self.test_active

        # if self.model.t % 7 == 1:
        #    # print every week the mean degree of second group
        #    group2 = self._nodes_in_classes(self.groups[1])
        #    mean_degree = compute_mean_degree(self.graph, group2)
        #    logging.debug(f"Day {self.model.t}: Mean degree of group2 {mean_degree}")


class AlternateFreeMonday(AlternatingPolicy):

    def __init__(self, graph, model, config_file=None):
        super().__init__(graph, model, config_file)

        self.weekend_start = 5
        self.weekend_end = 1
        self.testing = False


class AlternateAndMondayPCR(AlternatingPolicy):

    def __init__(self, graph, model, config_file=None):
        super().__init__(graph, model, config_file)

        self.weekend_start = 5
        self.weekend_end = 1

        self.testing = True
        self.test_sensitivity = 0.8
        self.test_days = (1,)
