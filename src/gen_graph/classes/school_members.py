from __future__ import annotations

import itertools
from collections import defaultdict

import numpy as np

from classes.apartment import Apartment
from classes.school import School
from constants import constants
from lang.mytypes import List, Any
from transport.travel_info import TravelInfo
from utils import math_prob


class DictByAge:
    total: int
    items: defaultdict[int, list]

    def __init__(self):
        self.total = 0
        self.items = defaultdict(list)

    def append(self, item):
        age = item.age
        self.items[age].append(item)
        self.total += 1

    def all_ages(self):
        return self.items.keys()

    def __getitem__(self, item):
        return self.items[item]

    def __iter__(self):
        return itertools.chain.from_iterable(self.items.values())


class SchoolMembers():
    elements: defaultdict[Any, DictByAge]

    def __init__(self, copy_from: SchoolMembers = None):
        if copy_from:
            self.elements = copy_from.elements.copy()
        else:
            self.elements = defaultdict(DictByAge)

    def copy(self):
        return SchoolMembers(self)

    def __getitem__(self, item):
        return self.elements[item]

    def put_to_school(self, school_type: School, person, sch_list: List[School], travel_info: TravelInfo):
        app: Apartment = person.household.apartment
        where = None
        if school_type.NEAREST:
            mind = np.Infinity
            for sch in sch_list:
                if self.elements[sch].total < sch.capacity:
                    dd = sch.distance_to(app.location) / sch.weight
                    if dd < mind and (school_type.ALLOW_TRAVEL or sch.town == person.town):
                        mind = dd
                        where = sch
        else:
            available = []
            for sch in sch_list:
                if self.elements[sch].total < sch.capacity:
                    available.append(sch)
            if len(available) == 0:
                return False
            where = math_prob.choice(available)
        if where is None:
            return False
        self.elements[where].append(person)
        travel_info.add_travel(constants.TravelType.SCHOOL, person.town, where.town, person, 1)
        return True
