import time
import numpy as np
import pandas as pd
from policies.policy import Policy
from utils.history_utils import TimeSeries
import logging

from models.agent_based_network_model import STATES
from utils.config_utils import ConfigFile

logging.basicConfig(level=logging.DEBUG)

def _process_calendar(filename):
    df = pd.read_csv(filename)
    return (
        dict(zip(df["T"], df["workers"].astype(int))),
        dict(zip(df["T"], df["elderly"].astype(int))),
    )

class Vaccination(Policy):

    """
    Vaccination Policy.
    """

    def __init__(self, graph, model, config_file=None):
        super().__init__(graph, model)

        self.first_day = True
        self.stopped = False
        self.delay = None

        # self.cf = ConfigFile()
        # if config_file is not None:
        #     self.cf.load(config_file)

        # -1 .. not vaccinated
        # >= 0 days from vaccination 
        self.vaccinated = np.full(self.graph.num_nodes, fill_value=-1, dtype=int)
        self.nodes = np.arange(self.graph.num_nodes)
        self.days_in_E = np.zeros(self.graph.num_nodes,  dtype=int) 
        self.target_for_R = np.zeros(self.graph.num_nodes, dtype=bool)  # auxiliary var      

        # statistics 
        self.stat_moved_to_R = TimeSeries(401, dtype=int)

        if config_file:
            cf = ConfigFile()
            cf.load(config_file)
            calendar_filename = cf.section_as_dict("CALENDAR").get("calendar_filename", None)
            if calendar_filename is None:
                raise "Missing calendar filename in vaccination policy config file."
            self.workers_calendar, self.elderly_calendar = _process_calendar(calendar_filename)
            self.delay = cf.section_as_dict("CALENDAR").get("delay", None)

            self.first_shot_coef = cf.section_as_dict("EFFECT")["first_shot"]
            self.second_shot_coef = cf.section_as_dict("EFFECT")["second_shot"]
        else:
            raise "Vaccination policy requires config file."

        self.old_to_vaccinate = list(np.argsort(self.graph.nodes_age))
        # self.index_to_go = len(self.sort_indicies)-1

        worker_id = self.graph.cat_table["ecactivity"].index("working")
        self.workers_to_vaccinate = list(self.nodes[self.graph.nodes_ecactivity == worker_id])
        # print(self.workers_to_vaccinate)
        # exit()
        
    def first_day_setup(self):
        pass 
                
    def stop(self):
        """ just finish necessary, but do nothing new """
        self.stopped = True
    
    def move_to_S(self):
        # take those who are first day E (are E AND are E the first day) 
        nodes_first_E = (self.model.memberships[STATES.E] == 1).ravel()
        self.days_in_E[nodes_first_E] += 1 
        # TODO: napsat to normalne logical_and
        nodes_first_E *= self.days_in_E == 1 #logical AND 

        if nodes_first_E.sum() == 0:
            return 

        # By 14 days after the first shot, the effect is zero (i.e.  an
        # infectedindividual becomes exposed and later symptomatic or asymptomaticas if
        # not vaccinated)•Between 14 and 20 days after the first shot, those, who are
        # infected(heading to theEcompartment) and are "intended" to be
        # asymptomatic(further go toIa, it is no harm to assume this decision is made
        # inforward) become recovered with probability0.29instead of
        # enteringtheEcompartment. Those, intended to be symptomatic (further gotoIp)
        # become recovered with0.46probability.•21days or more after first shot, this
        # probability of "recovery" is0.52for asymptomatic and0.6for symptomatic.•7days
        # after the second shot or later, the probability of "recovery" is0.9for
        # asymptomatic and0.92for symptomatic

        # divide nodes_first_E to asymptomatic candidates and symptomatic candidates
        # assert np.all(np.logical_or(
        #     self.model.state_to_go[nodes_first_E, 0] == STATES.I_n, 
        #     self.model.state_to_go[nodes_first_E, 0] == STATES.I_a
        # )), "inconsistent state_to_go"

        self.target_for_R.fill(0) 

        def decide_move_to_R(selected, prob):
            n = len(selected)
            print(f"generating {n} randoms")
            if n > 0:
                r = np.random.rand(n)
                self.target_for_R[selected] = r < prob
            
        # 14 - 20 days: 0.29 for A, 0.46 for S 
        # skip those with < 14 days 

        # for state, probs in (
        #         (STATES.I_n, [0.29, 0.52, 0.9]),
        #         (STATES.I_a, [0.46, 0.6, 0.92])
        # ):
        #     nodes_heading_to_state = nodes_first_E.copy() 
        #     nodes_heading_to_state[nodes_first_E] = self.model.state_to_go[nodes_first_E, 0] == state
        #     node_list = self.nodes[nodes_heading_to_state]
 
        #     if not(len(node_list) > 0):
        #         continue
        #     # skip those who are in first 14 days 
        #     node_list = node_list[self.vaccinated[node_list] >= 14]
        #     # select 14 - 21 
        #     selected = node_list[self.vaccinated[node_list] < 21] 
        #     decide_move_to_R(selected, probs[0]) 
        #     # skip them  
        #     node_list = node_list[self.vaccinated[node_list] >= 21]
        #     # selecte < second shot + 7 
        #     selected = node_list[self.vaccinated[node_list] < self.delay + 7] 
        #     decide_move_to_R(selected, probs[1]) 
        #     # skip them 
        #     node_list = node_list[self.vaccinated[node_list] >= self.delay + 7]
        #     decide_move_to_R(node_list, probs[2])

        # first shots
        
        node_list = self.nodes[nodes_first_E]
        
        if not(len(node_list) > 0):
            return

        # those who have only the first shot 
        first_shotters = node_list[
            np.logical_and(
                self.vaccinated[node_list] >= 14,
                self.vaccinated[node_list] < self.delay + 7
            )]
        r = np.random.rand(len(first_shotters))
        go_back = first_shotters[r < self.first_shot_coef]
        self.target_for_R[go_back] = True
        
        second_shotters = node_list[self.vaccinated[node_list] >= self.delay + 7]
        r = np.random.rand(len(second_shotters))
        go_back = second_shotters[r < self.second_shot_coef]
        self.target_for_R[go_back] = True
        
        self.stat_moved_to_R[self.model.t] = self.target_for_R.sum()
        self.model.move_target_nodes_to_S(self.target_for_R)
        self.days_in_E[self.target_for_R] = 0
        

    def process_vaccinated(self):
        self.move_to_S()
        
    def run(self):

        super().run()

        # update vaccinated days 
        already_vaccinated = self.vaccinated != -1
        self.vaccinated[already_vaccinated] += 1
        
        self.process_vaccinated()

        # update asymptotic rates  - OBSOLETE
        # Počítám, že první týden nemá vakcíná
        # žádnou účinnost, po týdnu 50%, po dvou týdnech 70%, po druhé
        # dávce 90% a po dalším týdnu 95%

        # older = self.graph.nodes_age > 65 
        # younger = np.logical_not(older) 
        
        # # update two weeks after first vaccination 
        # selected = self.vaccinated == 14 
        # self.model.asymptomatic_rate[np.logical_and(selected, older)] = 0.7
        # self.model.asymptomatic_rate[np.logical_and(selected, younger)] = 0.9  

        # # update two weeks after second vaccination 
        # selected = self.vaccinated == self.delay + 14  
        # self.model.asymptomatic_rate[np.logical_and(selected, older)] = 0.8
        # self.model.asymptomatic_rate[np.logical_and(selected, younger)] = 0.95  

        # selected = self.vaccinated == 7
        # self.model.asymptomatic_rate[selected] = 0.5 
        # selected = self.vaccinated == 14
        # self.model.asymptomatic_rate[selected] = 0.7
        # selected = self.vaccinated == self.delay
        # self.model.asymptomatic_rate[selected] = 0.9
        # selected = self.vaccinated == self.delay + 7 
        # self.model.asymptomatic_rate[selected] = 0.95
        
        logging.debug(f"asymptomatic rate {self.model.asymptomatic_rate.mean()}")
    
        
        if self.model.T in self.elderly_calendar:
            self.vaccinate_old(self.elderly_calendar[self.model.T])

        if self.model.T in self.workers_calendar:
            self.vaccinate_workers(self.workers_calendar[self.model.T]) 

    def vaccinate_old(self, num):
        if num == 0:
            return
        logging.info(f"T={self.model.T} Vaccinating {num} elderly.")
        index = len(self.old_to_vaccinate) 
        while num > 0 and index>0:
            index -= 1
            who = self.old_to_vaccinate[index]
            if self.vaccinated[who] != -1:
                continue 
            if self.model.node_detected[who]: # change to active case 
                continue
            if self.model.memberships[STATES.D, who, 0] == 1: # dead are not vaccinated
                continue
            self.vaccinated[who] = 0
            del self.old_to_vaccinate[index]
            num -= 1
        
    def vaccinate_workers(self, num):
        if num == 0:
            return
        logging.info(f"T={self.model.T} Vaccinating {num} workers.")
        num_workers = len(self.workers_to_vaccinate)
        if num_workers == 0:
            return

        # ids_to_vaccinate = self.workers_to_vaccinate[self.model.node_detected[self.workers_to_vaccinate] == False]
        # if len(ids_to_vaccinate) == 0:
        #     logging.warning("No more workers to vaccinate.")
        #     exit()
        #     return
        # ids_to_vaccinate = ids_to_vaccinate[self.model.memberships[STATES.D, ids_to_vaccinate, 0] != 1]
        
        ids_to_vaccinate = np.logical_and(
            self.model.node_detected[self.workers_to_vaccinate] == False,
            self.model.memberships[STATES.D, self.workers_to_vaccinate, 0] != 1
        ).nonzero()[0]
        
        if len(ids_to_vaccinate) < num:
            logging.info("Not enough workers to vaccinate.")
            num = len(ids_to_vaccinate)
            if num == 0:
                return
        selected_ids = np.random.choice(ids_to_vaccinate, size=num, replace=False)
        for index in selected_ids:
            who = self.workers_to_vaccinate[index]
            self.vaccinated[who] = 0
        for index in sorted(selected_ids, reverse=True):
            del self.workers_to_vaccinate[index]

            
        # # get all nodes that are S or Ss and were not vaccinated
        # target_nodes = np.logical_not(
        #     self.model.node_detected
        # )
        # target_nodes = np.logical_and(
        #     target_nodes[:,0],
        #     self.vaccinated == False
        # )
        # print(target_nodes.shape)
        # pool = self.nodes[target_nodes]

        # # select X of them to be vaccinated
        # to_vaccinate = np.random.choice(pool, size=self.num_to_vaccinate, replace=False)
        # self.vaccinated[to_vaccinate] = True
        # self.model.asymptomatic_rate[to_vaccinate] = 0.9 
        
        # #        self.model.move_to_R(to_vaccinate)

    def to_df(self):
        index = range(0+self.model.start_day-1, self.model.t+self.model.start_day) # -1 + 1 
        policy_name = type(self).__name__
        columns = {
            f"moved_to_R": self.stat_moved_to_R[:self.model.t+1],
        }
        columns["day"] = np.floor(index).astype(int)
        df = pd.DataFrame(columns, index=index)
        df.index.rename('T', inplace=True)
        return df


class VaccinationToR(Vaccination):

    def process_vaccinated(self):
        # # update two weeks after first vaccination
        nodes_in_S = self.nodes[self.model.memberships[STATES.S, :, 0] == 1]

            
        selected = nodes_in_S[self.vaccinated[nodes_in_S] == 14] 
        r = np.random.rand(len(selected))
        to_R = selected[r < self.first_shot_coef]

        self.target_for_R.fill(0) 
        self.target_for_R[to_R] = True 
        
        selected = nodes_in_S[self.vaccinated[nodes_in_S] == self.delay + 7]
        r = np.random.rand(len(selected))
        to_R  = selected[r < (self.second_shot_coef - self.first_shot_coef)]
        self.target_for_R[to_R] = True 

        self.stat_moved_to_R[self.model.t] = self.target_for_R.sum()
        self.model.move_target_nodes_to_R(self.target_for_R)
        self.days_in_E[self.target_for_R] = 0
                

class VaccinationToA(Vaccination):

    def update_asymptomatic_rates(self):
        # # update two weeks after first vaccination 
        selected = self.nodes[self.vaccinated == 14]
        srate = 1 - 0.179
        self.model.asymptomatic_rate[selected] = 1 - srate*(1-self.first_shot_coef)

        selected = self.nodes[self.vaccinated == self.delay + 7]
        self.model.asymptomatic_rate[selected] = 1 - srate*(1-self.second_shot_coef)
    
    def process_vaccinated(self):
        self.update_asymptomatic_rates()
        
class VaccinationToSA(VaccinationToA):
    
    def process_vaccinated(self):
        self.move_to_S()
        self.update_asymptomatic_rates() 
