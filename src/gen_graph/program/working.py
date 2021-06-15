import numpy as np

from classes.exports import ExportNode
from classes.person import Person
from classes.renv import Renv
from config.base_config import BaseConfig
from constants import constants
from lang.mytypes import List, Iterator
from params.cal_param import CalParamIdx
from plos.plos_matrix import PlosMatrix
from plos.working_params import WorkingParams
from utils import math_prob
from utils.log import StdLog


def run(env: Renv, rwparam: WorkingParams, config: BaseConfig, all_persons: Iterator[Person],
        exports: List[ExportNode]):
    def process_workplaces(persons: List[Person],
                           wm_scale: float):
        remaining = len(persons)
        while remaining > 0:
            workplace_size = math_prob.poisson(2 * wm_scale)
            if remaining < workplace_size:
                workplace_size = remaining
            next_remaining = remaining - workplace_size
            if workplace_size > 1:
                env.mutual_contacts(persons[next_remaining:remaining], constants.Layer.WORK_CONTACTS,
                                    constants.NO_SUBLAYER,
                                    CalParamIdx.ONE,
                                    min(wm_scale, workplace_size))
            remaining = next_remaining

    def age_weight(person: Person, age_weights, plos: PlosMatrix):
        cat = plos.categories.value2cat(person.age)
        if cat >= len(age_weights):
            return 0
        return age_weights[cat]

    workers = env.w
    workers.localize_commuting(env.travel_info)
    wm = PlosMatrix()
    wm.read(config.plos_data(constants.PlosCats.WORK), np.ones((wm.size,)))
    cats = wm.categories.CATS
    age_weights = wm.totals()
    age65 = wm.categories.value2cat(65)

    # cat+1 left to zero from plos see #6
    age_weights[age65:cats] = age_weights[age65]

    age_weights *= cats / np.sum(age_weights) * rwparam.weights

    all_persons_list = list(all_persons)

    for town in env.towns:
        agews = {wt: [age_weight(p, age_weights, wm) for p in workers.intowns[town][wt]] for wt in constants.WorkType}

        StdLog.log(f"Creating working network in {town.name}")
        for wt in constants.WorkType:
            persons = workers.intowns[town][wt]
            wm_scale = config.WORK_MATRIX[wt][wt]
            process_workplaces(persons, wm_scale)

            for person in persons:
                age_w = age_weight(person, age_weights, wm)
                for wtt in constants.WorkType:
                    maxk = age_w * config.WORK_MATRIX[wt][wtt] / 2
                    if wtt is not wt and maxk > 0:
                        k = min(math_prob.poisson(maxk),
                                len(workers.intowns[town][wtt]))
                        if k:
                            for contact in math_prob.draw_distinct_weighted(agews[wtt], k, workers.intowns[town][wtt]):
                                if contact is not person:
                                    env.add_edge(person, contact, constants.Layer.WORK_CONTACTS, constants.NO_SUBLAYER,
                                                 1, CalParamIdx.ONE)

                maxk = int(age_w * config.CLIENT_MULT * config.WORK_2_CLIENT[wt])
                if maxk > 0:
                    k = math_prob.uniform(maxk)
                    if k > 0:
                        prob = 1 / config.CLIENT_MULT
                        for contact in math_prob.draw_distinct_uniform(k, all_persons_list):
                            if contact is not person:
                                env.travel_info.add_travel(constants.TravelType.SERVICE, person.town, contact.town,
                                                           contact,
                                                           prob)
                                env.add_edge(person, contact, constants.WORK_2_CLIENT_LAYERS[wt], constants.NO_SUBLAYER,
                                             prob, CalParamIdx.ONE)

    for wt in constants.WorkType:
        for person in workers.outofregion[wt]:
            exp = math_prob.choice(exports)
            env.add_export_edge(person, exp, constants.Layer.WORK_CONTACTS, constants.NO_SUBLAYER,
                                config.WORK_EXPORT_RATE)
