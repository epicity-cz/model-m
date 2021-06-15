import logging

class Policy:

    def __init__(self, graph, model):
        self.graph = graph
        self.model = model

    def run(self):
        if self.first_day:
            self.first_day_setup()
            self.first_day = False
            
        logging.info(
            f"This is the {self.__class__.__name__} policy run.  {'(STOPPED)' if self.stopped else ''}")


    def to_df(self):
        """ returns data frame with statics 
        must contain a column T with dates (self.model.t values) 
        may be empty
        """
        print("DBG Warning NOT IMPLEMENTED YET (to_df)")
        return None 

# def bound_policy(func, graph, coefs=None):
#     """ Bounds the given function func with the particular graph.
#     Use to create a callback function for network model from
#     your policy function.
#     """
#     if coefs is None:
#         def policy_function(*args,  **kwargs):
#             ret = func(graph, *args, **kwargs)
#             if "graph" in ret:
#                 ret["graph"] = graph.final_adjacency_matrix()
#             return ret
#         return policy_function
#     else:
#         def policy_function(*args, **kwargs):
#             ret = func(graph, coefs, *args, **kwargs)
#             if "graph" in ret:
#                 ret["graph"] = graph.final_adjacency_matrix()
#             return ret
#         return policy_function
