from scipy.optimize import minimize

from classes.mgraph import MGraph
from classes.person import Person
from config.base_config import BaseConfig
from constants import constants
from lang.mytypes import List
from params.parameters import Parameters, ParamIdx
from plos.plos_matrix import PlosMatrix


def from_graph(grph: MGraph, parameters: Parameters) -> List[PlosMatrix]:
    items = [PlosMatrix() for _ in constants.PlosCats]
    for (p1, p2, link) in grph.links():
        ld = constants.LAYER_DEFS[link.layer]
        prob = link.probability_real(parameters)
        pt1 = items[ld.first]
        if isinstance(p2, Person):
            pt1.add_contact(p1.age, p2.age, prob)
            pt1.add_person(p1, p1.age)

            pt2 = items[ld.second]
            pt2.add_contact(p2.age, p1.age, prob)
            pt2.add_person(p2, p2.age)
        else:
            pt1.add_external(p1.age, prob)

    return items


def distance(x, pm_ref, grph, param_slice, params, ploscat):
    params[param_slice] = x
    pmm = from_graph(grph, params)
    pm_pom = pm_ref.copy()
    pm_pom.set_nums(pmm[ploscat].nums)
    dist = pm_pom.pmdist2(pmm[ploscat])
    print(f"trying\t{x}\t{dist}")
    return dist


def minimize_params(params, param_slice, ploscat, config: BaseConfig, grph: MGraph):
    pm_ref = PlosMatrix()
    pm_ref.read1(config.plos_data(ploscat))
    x0 = params[param_slice]
    result = minimize(distance, x0, args=(pm_ref, grph, param_slice, params, ploscat),
                      options={'eps': config.CALIBRATE_EPS, 'gtol': config.CALIBRATE_GTOL})
    params[param_slice] = result.x
    print(result.x)
    print(result.success)
    print(result)
    return result


def calibrate(grph: MGraph, config: BaseConfig):
    params = Parameters([0.65, 0.7, 0.4, 0.25, 3.5, 1.5, 1.5, 1.0, 1.0])

    # minimize Home
    minimize_params(params, ParamIdx.slice_family(), constants.PlosCats.HOME, config, grph)

    # minimize School
    minimize_params(params, ParamIdx.slice_school(), constants.PlosCats.SCHOOL, config, grph)

    print(f"Calibrated parameters {params}")
    return params
