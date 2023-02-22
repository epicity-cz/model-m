from extended_network_model import STATES as states
from policy import Policy

class GuidedPolicy(Policy):

    def __init__(self, graph, model):
        super().__init__(graph, model)
        self.count = 10 
        self.layer_changes = load_scenario_dict(
            "../data/policy_params/close.csv")["C"] 
        self.orig_layers = None
        self.closed = False 

    def close_layers(self):
        print("LOG ****************** CLOSING ***************** ")
        self.orig_layers = self.graph.layer_weights 
        self.graph.set_layer_weights(self.layer_changes)
        self.closed = True 
        
    def open_layers(self): 
        print("LOG ****************** OPENING ***************** ")
        self.closed = False 
        self.graph.set_layer_weights(self.orig_layers)

    def run(self): 
        Id_count = self.state_counts[states.I_d][self.model.t]
        if not self.closed and Id_count > self.count: 
            self.update_layers()
        elif self.closed and Id_count < self.count:
            self.open_layers()
        
