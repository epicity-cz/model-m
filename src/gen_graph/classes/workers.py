from classes.person import Person
from classes.town import Town
from config.base_config import BaseConfig
from constants import constants
from lang.mytypes import Dict, List
from transport.travel_info import TravelInfo
from utils import math_prob


def wt_dict():
    return {key: [] for key in constants.WorkType}


class Workers:
    intowns: Dict[Town, Dict[constants.WorkType, List[Person]]]
    inregion: Dict[constants.WorkType, List[Person]]
    outofregion: Dict[constants.WorkType, List[Person]]
    employed: Dict[Town, Dict[constants.WorkType, List[Person]]]
    config: BaseConfig
    towns: List[Town]

    def __init__(self, config: BaseConfig, towns: List) -> None:
        self.intowns = {}  # towns x len(constants.WorkType)
        self.inregion = wt_dict()  # len(constants.WorkType)
        self.outofregion = wt_dict()  # len(constants.WorkType)
        self.employed = {}  # towns x len(constants.WorkType)
        self.config = config
        self.towns = towns
        for town in towns:
            self.intowns[town] = wt_dict()
            self.employed[town] = wt_dict()

    def copy(self):
        other = Workers(self.config, self.towns)
        other.intowns = {town: {wt: self.intowns[town][wt].copy() for wt in constants.WorkType} for town in other.towns}
        other.inregion = {wt: self.inregion[wt].copy() for wt in constants.WorkType}
        other.outofregion = {wt: self.outofregion[wt].copy() for wt in constants.WorkType}
        other.employed = {town: {wt: self.employed[town][wt].copy() for wt in constants.WorkType} for town in
                          other.towns}
        return other

    def add_person(self, person, town):
        if person.activity == constants.EconoActivity.WORKING:
            if person.commute == constants.CommutingTime.NOT_COMMUTING:
                self.intowns[town][person.worktype].append(person)
            elif person.commute <= self.config.LAST_INSIDE:
                self.inregion[person.worktype].append(person)
            else:
                self.outofregion[person.worktype].append(person)

    def shuffle(self):
        for town in self.intowns:
            for wt in self.intowns[town]:
                math_prob.shuffle(self.intowns[town][wt])
        for key in constants.WorkType:
            math_prob.shuffle(self.inregion[key])
            math_prob.shuffle(self.outofregion[key])

    def employ(self, town, wt: constants.WorkType, travel_info):
        return self.employ_local(town, wt, travel_info) or self.employ_commuting(town, wt, travel_info)

    def employ_local(self, town, wt: constants.WorkType, travel_info: TravelInfo):
        if town not in self.intowns:
            assert False
        if not self.intowns[town][wt]:
            return None
        person = self.intowns[town][wt].pop()
        travel_info.add_travel(constants.TravelType.WORK, town, town, person, 1)
        self.employed[town][wt].append(person)
        return person

    def employ_commuting(self, town, wt: constants.WorkType, travel_info: TravelInfo):
        for index, person in enumerate(self.inregion[wt]):
            if person.town is not town:
                travel_info.add_travel(constants.TravelType.WORK, person.town, town, person, 1)
                self.employed[town][wt].append(person)
                del self.inregion[wt][index]
                return person
        return None

    def localize_commuting(self, travel_info: TravelInfo):
        town_counts = [town.count_persons() for town in self.towns]
        for wt in constants.WorkType:
            while self.inregion[wt]:
                worker = self.inregion[wt].pop()
                home = worker.town
                while True:
                    other = math_prob.draw(town_counts, self.towns)
                    if other is not home:
                        break
                travel_info.add_travel(constants.TravelType.WORK, home, other, worker, 1)
                self.intowns[other][wt].append(worker)
