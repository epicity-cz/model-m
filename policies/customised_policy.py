import pandas as pd
import numpy as np
import json
from policy import Policy
from policy_utils import load_scenario_dict
from functools import partial, reduce
from eva_policy import EvaQuarantinePolicy
from contact_tracing import ContactTracingPolicy

import logging

def _load_dictionary(filename: str):
    
    if filename is None:
        return None
    if filename.endswith(".json"):
        with open(filename, "r") as f:
            return { 
                int(key): value 
                for key, value in json.load(f).items()
            }
    else:
        df = pd.read_csv(filename)
        return dict(zip(df["T"], df.iloc[:,1]))


class CustomPolicy(Policy):

    def __init__(self,
                 graph,
                 model,
                 layer_changes_filename=None,
                 param_changes_filename=None,
                 policy_calendar_filename=None,
                 beta_factor_filename=None,
                 face_masks_filename=None,
                 theta_filename=None,
                 test_rate_filename=None,
                 superspreader_date=None,
                 superspreader_layer=None,
                 force_infect=None,
                 force_infect_layer=None,
                 init_filename=None,
                 reduction_coef1=None,
                 reduction_coef2=None,
                 new_beta=None,
                 daily_import=None,
                 **kwargs):
        super().__init__(graph, model)

        if layer_changes_filename is not None:
            self.layer_changes_calendar = load_scenario_dict(
                layer_changes_filename)
        else:
            self.layer_changes_calendar = None

        if policy_calendar_filename is not None:
            with open(policy_calendar_filename, "r") as f:
                self.policy_calendar = {
                    int(k): v
                    for k, v in json.load(f).items()
                }
        else:
            self.policy_calendar = None

        if param_changes_filename is not None:
            raise ValueError("Temporarily disabled. Sry.")
            with open(param_changes_filename, "r") as f:
                self.param_changes_calendar = json.load(f)
        else:
            self.param_changes_calendar = None

        self.face_masks_calendar = _load_dictionary(face_masks_filename) 
        if self.face_masks_calendar is None:
            logging.warning("DBG: NO MASKS ")
            
        self.beta_factor_calendar = _load_dictionary(beta_factor_filename) 

        self.theta_calendar = _load_dictionary(theta_filename) 
        if self.theta_calendar is None:
            logging.warning("DBG: Theta calendar")

        self.test_rate_calendar = _load_dictionary(test_rate_filename)

        self.policies = {}

        if superspreader_date is not None:
            self.superspreader_date = int(superspreader_date)
        else:
            self.superspreader_date = None

        if superspreader_layer is not None:
            self.superspreader_layer = int(superspreader_layer)
        else:
            self.superspreader_layer = 31 

        if force_infect is not None:
            self.force_infect = int(force_infect)
            assert self.force_infect 
        else:
            self.force_infect = None

        if force_infect_layer is not None:
            self.force_infect_layer = int(force_infect_layer)
        else:
            self.force_infect_layer = None

        if init_filename is not None:
            with open(init_filename, "r") as f:
                self.init_calendar = {
                    int(k): v
                    for k,v in json.load(f).items()
                }
        else:
            self.init_calendar = None

        if reduction_coef1 is not None:
            self.reduction_coef1 = float(reduction_coef1)
        else:
            self.reduction_coef1 = 0.9 
            raise ValueError("Missing coef1")

        if reduction_coef2 is not None:
            self.reduction_coef2 = float(reduction_coef2)
        else:
            self.reduction_coef2 = 0.2 
            raise ValueError("Missing coef2")

           
        self.new_beta = False if new_beta is None else new_beta == "Yes"
        if self.new_beta:
            logging.debug("Using new beta")
            assert self.beta_factor_calendar is not None 
            assert self.face_masks_calendar is not None
            assert self.beta_factor_calendar.keys() == self.face_masks_calendar.keys()

        self.nodes_infected  = None
        
        if daily_import is not None:
            self.daily_import = float(daily_import)
        else:
            self.daily_import = daily_import

        if "sub_policies" in kwargs:
            for sub_policy_name in kwargs["sub_policies"]:
                filename = kwargs[f"{sub_policy_name}_filename"]
                class_name = kwargs[f"{sub_policy_name}_name"]
                config = kwargs.get(f"{sub_policy_name}_config", None)
                
                policy = f"{filename}:{class_name}" if config is None else f"{filename}:{class_name}:{config}"
                self.policies[policy] = self.create_policy(filename, class_name, config)


    def create_policy(self, filename, object_name, config_file=None):
        PolicyClass = getattr(__import__(filename), object_name)
        if config_file:
            return PolicyClass(self.graph, self.model, config_file=config_file)
        else:
            return PolicyClass(self.graph, self.model)# todo: fix - all policies should accept cfg

        

        
    def update_layers(self, coefs):
        self.graph.set_layer_weights(coefs)

    def switch_on_superspread(self):
        logging.info("DBG Superspreader ON")
        self.graph.layer_weights[self.superspreader_layer] = 1.0

    def switch_off_superspread(self):
        logging.info("DBG Superspreader OFF")
        self.graph.layer_weights[self.superspreader_layer] = 0.0 


    #TODO fix for diff betas
    def update_beta(self, masks):

        orig_beta = self.model.init_kwargs["beta"]
        orig_beta_A = self.model.init_kwargs["beta_A"]

        reduction = (1 - self.reduction_coef1 * masks)
        
        new_value = orig_beta * reduction 
        new_value_A = orig_beta_A * reduction 


        logging.debug(f"{self.model.t} DBG BETA {new_value}")

        self.model.beta.fill(new_value)
        self.model.beta_A.fill(new_value_A)

        orig_beta_in_family = self.model.init_kwargs["beta_in_family"]
        orig_beta_A_in_family = self.model.init_kwargs["beta_A_in_family"]
    
        reduction = 1 - self.reduction_coef2 * (1-reduction)

        new_value = orig_beta_in_family * reduction         
        new_value_A = orig_beta_A_in_family * reduction 
        self.model.beta_in_family.fill(new_value)
        self.model.beta_A_in_family.fill(new_value_A)

        
        # for name, value in ("beta", orig_beta), ("beta_A", orig_beta_A):
        #     new_value = value * reduction
        #     if isinstance(new_value, (list)):
        #         np_new_value = np.array(new_value).reshape(
        #             (self.model.num_nodes, 1))
        #     else:
        #         np_new_value = np.full(
        #             fill_value=new_value, shape=(self.model.num_nodes, 1))
        #setattr(self.model, name, np_new_value)
        logging.debug(f"DBG beta: {self.model.beta[0][0]} {self.model.beta_in_family[0][0]}")

    # TODO: make general func update_param ?
    # (beta includes beta_A, theta includes various thetas)
    
    def update_beta2(self, masks, beta_factors=None):

        assert beta_factors is not None

        orig_beta_in_family = self.model.init_kwargs["beta_in_family"]
        orig_beta_A_in_family = self.model.init_kwargs["beta_A_in_family"]
    
        reduction = (1 - self.reduction_coef1*beta_factors) 

        new_value = orig_beta_in_family * reduction         
        new_value_A = orig_beta_A_in_family * reduction 
        self.model.beta_in_family.fill(new_value)
        self.model.beta_A_in_family.fill(new_value_A)

        #orig_beta = self.model.init_kwargs["beta"]
        #orig_beta_A = self.model.init_kwargs["beta_A"]
        
        # assumes betas are uniform, fixme 
        new_beta = (1-self.reduction_coef2*masks) * self.model.beta_in_family[0][0]
        new_beta_A = (1-self.reduction_coef2*masks) *self.model.beta_A_in_family[0][0]
        
        logging.debug(f"{self.model.t} DBG BETA {new_value}")

        self.model.beta.fill(new_value)
        self.model.beta_A.fill(new_value_A)

        logging.debug(f"DBG beta: {self.model.beta[0][0]} {self.model.beta_A[0][0]}")


    def update_test_rate(self, coef):
        orig_test_rate = self.model.init_kwargs["test_rate"]
        new_value = coef * orig_test_rate
        #self.model.test_rate = new_value
        self.model.theta_Is.fill(new_value)
    
    def update_theta(self, coef):
        orig_theta = self.model.init_kwargs["theta_Is"]
        new_value = orig_theta * coef 
        self.model.theta_Is.fill(new_value)
        # if isinstance(new_value, (list)):
        #     np_new_value = np.array(new_value).reshape(
        #         (self.model.num_nodes, 1))
        # else:
        #     np_new_value = np.full(
        #         fill_value=new_value, shape=(self.model.num_nodes, 1))
        # setattr(self.model, "theta_Is", np_new_value)
        logging.debug(f"DBG theta: {self.model.theta_Is[0][0]}")

    def run(self):

        if False and self.graph.is_quarantined is not None:
            # dbg check
            all_deposited = np.zeros(self.graph.number_of_nodes)
            for p in self.policies.values():
                all_deposited = all_deposited + p.depo.depo
                if isinstance(p, EvaQuarantinePolicy) or isinstance(p, ContactTracingPolicy):
                    all_deposited += p.waiting_room_second_test.depo
            assert sum(all_deposited > 0) == sum(self.graph.is_quarantined > 0),  f"{all_deposited.nonzero()[0]} \n {self.graph.is_quarantined.nonzero()[0]}"
              
            #print(all_deposited.nonzero()[0],
            #     self.graph.is_quarantined.nonzero()[0])
            assert np.all(
                all_deposited.nonzero()[0] == self.graph.is_quarantined.nonzero()[0]), f"{all_deposited.nonzero()[0]} \n {self.graph.is_quarantined.nonzero()[0]}"
                

        
        logging.info(f"CustomPolicy {int(self.model.t)}")
        today = int(self.model.t)
        
        if self.init_calendar is not None and today in self.init_calendar:
            num = self.init_calendar[today]
            self.model.move_to_E(num)

        if self.daily_import is not None:
            assert self.daily_import <= 1.0, "not implemented yet"   
            if np.random.rand() < self.daily_import:
                self.model.move_to_E(1)
            
        if self.policy_calendar is not None and today in self.policy_calendar:
            logging.info("changing quarantine policy")
            # change the quaratine policy function
            for action, policy in self.policy_calendar[today]:
                if action == "start":
                    vals = policy.strip().split(":") 
                    filename, object_name = vals[0], vals[1]
                    config_file = None if len(vals) == 2 else vals[2] 
                    PolicyClass = getattr(__import__(filename), object_name)
                    if config_file is None:
                        self.policies[policy] = PolicyClass(self.graph, self.model)
                    else:
                        self.policies[policy] = PolicyClass(self.graph, self.model, config_file=config_file)
                    #params = [ float(param) for param in vals[2:] ]
                    #self.policies[policy] = PolicyClass(self.graph, self.model, *params)
                elif action == "stop":
                    self.policies[policy].stop()
                else:
                    raise ValueError(f"Unknown action {action}")

        if self.param_changes_calendar is not None and today in self. param_changes_calendar:
            for action, param, new_value in self.param_changes_calendar[today]:
                if action == "set":
                    if isinstance(new_value, (list)):
                        np_new_value = np.array(new_value).reshape(
                            (self.model.num_nodes, 1))
                    else:
                        np_new_value = np.full(
                            fill_value=new_value, shape=(self.model.num_nodes, 1))
                    setattr(self.model, param, np_new_value)
                elif action == "*":
                    attr = getattr(self.model, param)
                    if type(new_value) == str:
                        new_value = getattr(self.model, new_value)
                    setattr(self.model, param, attr * new_value)
                else:
                    raise ValueError("Unknown value")

        if self.layer_changes_calendar is not None and today in self.layer_changes_calendar:
            logging.debug(f"{today} updating layers")
            self.update_layers(self.layer_changes_calendar[today])

        if self.force_infect is not None and self.model.t == self.force_infect: 
            # number_to_infect = 5 if self.force_infect_layer in (33,34,35) else 10
            number_to_infect = 1
            nodes_on_layer = self.graph.get_nodes(self.force_infect_layer)
            nodes_to_infect = np.random.choice(nodes_on_layer, number_to_infect, replace=False)
            self.model.force_infect(nodes_to_infect)
            self.nodes_infected = nodes_to_infect 
 #           self.model.test_rate[self.nodes_infected] = 1.0 
 #           self.model.theta_Is[self.nodes_infected] =  0.0 
 #           self.model.testable[self.nodes_infected] = True
            
#        if self.nodes_infected is not None:
#            if self.model.t == self.force_infect + 6:
#                self.model.theta_Is[self.nodes_infected] =  1.0 
#                self.nodes_infected = None


        if self.superspreader_date is not None:
            if self.model.t == self.superspreader_date:
                self.switch_on_superspread() 
            elif self.model.t - 1 == self.superspreader_date:
                self.switch_off_superspread()

        if self.face_masks_calendar is not None and today in self.face_masks_calendar:
            logging.debug(f"DBG face masks update", self.face_masks_calendar)
            if self.new_beta:
                self.update_beta2(self.face_masks_calendar[today], self.beta_factor_calendar[today])
            else:
                raise ValueError("Temporarily disabled.")

        if self.theta_calendar is not None and today in self.theta_calendar:
            logging.debug(f"DBG theta update")
            self.update_theta(self.theta_calendar[today])

        if self.test_rate_calendar is not None and today in self.test_rate_calendar:
            logging.debug(f"DBG test rate update")
            self.update_test_rate(self.test_rate_calendar[today])
            
        # perform registred policies
        for name, policy in self.policies.items():
            logging.info(f"run policy { name}")
            policy.run()

        

    def to_df(self):
        if not self.policies:
            return None

        dfs = [ 
            p.to_df() 
            for p in self.policies.values()
        ]
        dfs = [ d for d in dfs if d is not None ]
        if not dfs:
            return None
        my_merge = partial(pd.merge, on="T", how="outer")
            
        return reduce(my_merge, dfs)
        
        
