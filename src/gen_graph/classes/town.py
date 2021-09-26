import itertools

import numpy as np

from classes.apartment import Apartment
from classes.household import Household
from classes.town_params import TownParams
from constants import constants
from lang.mytypes import List, Location, Optional
from program import loaders
from ruian.downloader import DownloadObjects
from utils import math_prob
from utils.has_id import HasId
from utils.log import StdLog


class Town(TownParams, HasId):
    apartments: List[Apartment]
    households_unasigned: List[Household]
    name: str
    location: Optional[Location]

    def __init__(self, zuj, filename, cache_dir):
        TownParams.__init__(self, zuj)
        self.gen_id()
        self.apartments = []
        self.households_unasigned = []  # ids of households
        self.tid = None
        self.ecats = np.zeros([len(constants.Gender), len(constants.EconoActivity)], dtype=int)
        self.location = None
        if not self.load(filename):
            self.load_csu(cache_dir)
            self.save(filename)

    def load_apartments(self, filename):
        self.ensure_apartments(filename)
        for (fh, x, y) in loaders.apartments_loader(filename):
            a = Apartment(self, (x, y), fh)
            self.apartments.append(a)

    def person_household(self, person):
        self.households_unasigned[-1].add_person(person)

    def household_factory(self):
        h = Household()
        self.households_unasigned.append(h)
        return h

    def add_person(self, person):
        self.ecats[person.sex, person.activity] += 1

    def place_households(self):
        # shuffle households
        math_prob.shuffle(self.households_unasigned)
        for i, hh in enumerate(self.households_unasigned):
            if i < len(self.apartments):
                self.apartments[i].add_household(hh)
            else:
                goon = True
                app = None
                while goon:
                    app = math_prob.choice(self.apartments)
                    if app.fh:
                        if abs(hh.age_average() - app.households[-1].age_average()) < 20:
                            goon = math_prob.yes_no(0.9)
                app.add_household(hh)
        self.households_unasigned = []

    def all_persons(self):
        return itertools.chain.from_iterable((app.all_persons() for app in self.apartments))

    def count_persons(self):
        return sum((app.count_persons() for app in self.apartments))

    def count_households(self):
        return sum((len(app.households) for app in self.apartments))

    def info(self):
        StdLog.log(f"Town {self.zuj} {self.name} "
                   f"Persons:{self.count_persons()} "
                   f"Households: {self.count_households()} "
                   f"Apartments:{len(self.apartments)}")
        StdLog.log("check ec")
        StdLog.log(f"{self.ecats / np.sum(self.ecats, axis=1, keepdims=True) - self.ec}")
        StdLog.log("")

    def ensure_apartments(self, filename):
        dwnldr = DownloadObjects()
        dwnldr.apartments(self.zuj, filename)
