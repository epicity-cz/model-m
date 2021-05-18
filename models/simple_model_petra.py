# -*- coding: utf-8 -*-
# how to rewrite roman's approach in petra's framework
import random

from engine import BaseEngine
from model import create_custom_model

# ENGINE PART
TIME_OF_SIMULATION = 30

HEALTHY = 0
INFECTED = 1
TIME_OF_INFECTION = 3


class Person():
    def __init__(self, id, init_state=HEALTHY):
        self.state = init_state
        self.time_of_state = 0
        self.id = id

    def infect(self, by_whom):
        self.state = INFECTED
        self.time_of_state = 0
        if by_whom == -1:
            print(self.id, " infected initially")
        else:
            print(self.id, " infected by", by_whom)

    def heal(self):
        self.state = HEALTHY
        self.time_of_state = 0
        print(self.id, " is healed")

    def stay_infected(self):
        self.time_of_state += 1
        if self.time_of_state == TIME_OF_INFECTION:
            self.heal()

    def stay_healthy(self):
        self.time_of_state += 1


class NoModel(BaseEngine):

    def inicialization(self):
        """ model inicialization """

        if self.random_seed:
            random.seed(self.random_seed)

        self.N = self.number_of_people
        self.num_nodes = self.N  # pocitat to ze takhle se jmenuje pocet nodu
        self.by_whom = [None] * self.num_nodes

        self.people = []
        inf_idx = random.sample(range(self.N), self.number_of_infected)
        for p in range(self.N):
            new_person = Person(p)
            if p in inf_idx:
                new_person.infect(-1)
            else:
                new_person.heal()
            self.people.append(new_person)

    def is_it_transmission(self, a, b):
        # ignore everything, just toss a coin with transmission_rate if b is infected
        if b.state == HEALTHY:
            return False
        else:
            if random.random() < self.trans_rate:
                return True
            else:
                return False

    def run_iteration(self):
        self.contacts_per_day = self.average_contacts
        for p in self.people:
            if p.state == HEALTHY:
                contacts = random.sample(range(self.N), self.contacts_per_day)
                for c in contacts:
                    if self.is_it_transmission(p, self.people[c]):
                        p.infect(c)
            if p.state == INFECTED:
                p.stay_infected()

    def run(self, T=TIME_OF_SIMULATION):
        for self.t in range(1, T+1):
            print("t = %.2f" % self.t)
            self.run_iteration()


# MODEL PART

model_definition = {
    "states": ["HEALTHY", "INFECTED"],
    "transitions": [
        ("HEALTHY", "INFECTED"),
        ("INFECTED", "HEALTHY")
    ],
    "init_arguments": {
        "time_of_infection": (3, "time of infection"),
        "average_contacts": (3, "average contacts"),
        "trans_rate": (0.8, "transition rate"),
        "number_of_infected": (1, "number of infected"),
        "number_of_people": (10, "number of people")
    }
}


def calc_propensities(model):
    """ na propensity se tu nehraje """
    pass


RoModel = create_custom_model("RoModel",
                              **model_definition,
                              calc_propensities=calc_propensities,
                              engine=NoModel)


if __name__ == "__main__":
    m = RoModel(None, random_seed=42)  # None je graph, ten je povinej
    m.run()
