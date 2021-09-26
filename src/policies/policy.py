import logging

class Policy:

    """
    Base Policy class.

    To implement a custom policy, derive your subclass.
    """

    def __init__(self, graph, model):
        self.graph = graph
        self.model = model
        self.stopped = False
        self.first_day = True

    def first_day_setup(self):
        pass 

    def run(self):
        if self.first_day:
            self.first_day_setup()
            self.first_day = False
            
        logging.info(
            f"This is the {self.__class__.__name__} policy run.  {'(STOPPED)' if self.stopped else ''}")


    def to_df(self):
        """ Returns data frame with policy related statistics.
        Must contain a column T with dates (self.model.T values). 
        May be empty.
        """
        logging.warning("NOT IMPLEMENTED YET (to_df)")
        return None

