import itertools

from classes.apartment import Apartment
from classes.person import Person
from classes.renv import Renv
from config.base_config import BaseConfig
from constants import constants
from lang.mytypes import List
from params.cal_param import CalParamIdx
from utils import math_prob
from utils.log import StdLog


def run(renv: Renv, apartments: List[Apartment], config: BaseConfig):
    def age_adjust(p1: Person, p2: Person, base: CalParamIdx) -> CalParamIdx:
        if abs(p1.age - p2.age) <= config.FAMILY_AGE_SAME:
            return base
        factor = base
        if p1.age < config.FAMILY_FACTOR_LIMIT:
            factor += 1
        if p2.age < config.FAMILY_FACTOR_LIMIT:
            factor += 1
        return factor

    inter_hh = 0
    inter_local = 0
    inter_outside = 0
    for ap in apartments:
        for hh in ap.households:
            for (p, q) in itertools.combinations(hh.persons, 2):
                renv.add_edge(p, q, constants.Layer.FAMILY_INSIDE, ap.id, 1, age_adjust(p, q, CalParamIdx.FAM_ONE))
                inter_hh += 1

        for h1, h2 in itertools.combinations(ap.households, 2):
            for p, q in itertools.product(h1.persons, h2.persons):
                renv.add_edge(p, q, constants.Layer.FAMILY_IN_HOUSE, constants.NO_SUBLAYER, 1,
                              age_adjust(p, q, CalParamIdx.FAM_INTER_GEN_ONE))
                inter_local += 1

    for ap in apartments:
        for hh in ap.households:
            seniors = []
            average_age = 0
            for person in hh.persons:
                if person.age > config.FAMILY_SENIOR_AGE:
                    seniors.append(person)
                    average_age += person.age
            if len(seniors):
                average_age /= len(seniors)
                children = math_prob.draw(config.NUMBER_OF_CHILDREN)
                children_remaining = children - len(ap.households) + 1
                used_apartments = set()
                used_apartments.add(ap)
                while (children_remaining > 0):
                    other_apartment = math_prob.choice(apartments)
                    if other_apartment in used_apartments:
                        continue
                    used_apartments.add(other_apartment)
                    if len(other_apartment.households) == 0:
                        continue
                    other_household = math_prob.choice(other_apartment.households)
                    juniors = []
                    for person in other_household.persons:
                        if person.age < average_age - 20:
                            juniors.append(person)
                    if len(juniors):

                        prob = 1 / children
                        # seniors visit juniors
                        for senior in seniors:
                            renv.travel_info.add_travel(constants.TravelType.FAMILY, ap.town, other_apartment.town,
                                                        senior, prob)  # TODO prob FAM_SENIOR_VISIT
                            for other in other_household.persons:
                                renv.add_edge(senior, other, constants.Layer.FAMILY_VISITSORS_TO_VISITED,
                                              other_apartment.id, prob, CalParamIdx.FAM_SENIOR_VISIT)
                                inter_outside += 1

                        # juniors visit seniors
                        for junior in juniors:
                            renv.travel_info.add_travel(constants.TravelType.FAMILY, other_apartment.town, ap.town,
                                                        junior, 1)  # TODO prob FAM_JUNIOR_VISIT
                            for other in hh.persons:
                                renv.add_edge(junior, other, constants.Layer.FAMILY_VISITSORS_TO_VISITED, ap.id, 1,
                                              CalParamIdx.FAM_JUNIOR_VISIT)
                                inter_outside += 1

                        children_remaining -= 1

    StdLog.log(f"{inter_hh} interhousehold, "
               f"{inter_local} intergeneration local, "
               f"{inter_outside} intergeneration outside.")
