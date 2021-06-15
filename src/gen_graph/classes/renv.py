from __future__ import annotations

import itertools
from collections import defaultdict

from classes.gobject import GObject
from classes.mgraph import MGraph
from classes.school_members import SchoolMembers
from classes.workers import Workers
from config.base_config import BaseConfig
from constants import constants
from lang.mytypes import List, Optional, Any
from params.cal_param import CalParamIdx
from transport.travel_info_all import travel_info_by_name, TravelInfo
from utils import math_prob
from utils.log import StdLog


class Renv:
    config: BaseConfig
    w: Workers  # @puml link Workers
    school_members: SchoolMembers  # @puml link SchoolMembers
    school_teachers: defaultdict[Any, List]  # @puml link People
    travel_info: TravelInfo
    graph: MGraph
    scale: float

    def __init__(self, region_config: BaseConfig, towns: List, copy_from: Optional[Renv] = None):
        if copy_from:
            self.w = copy_from.w.copy()
            self.school_members = copy_from.school_members.copy()
            self.school_teachers = copy_from.school_teachers.copy()
            self.travel_info = copy_from.travel_info.copy()
            self.graph = copy_from.graph.copy()
            self.scale = copy_from.scale
        else:
            self.w = Workers(region_config, towns)
            self.school_members = SchoolMembers()
            self.school_teachers = defaultdict(list)
            self.travel_info = travel_info_by_name(region_config.TRAVEL_INFO)(region_config, towns)
            self.graph = MGraph()
            self.scale = 1
        self.config = region_config
        self.towns = towns

    def copy(self):
        cpy = Renv(self.config, self.towns, self)
        return cpy

    def teacher_class_contacts(self, teacher, students, layer, sublayer, w_layer: CalParamIdx):
        cnt = len(students)
        if cnt < 1:
            return

        if cnt <= self.config.MAX_ALL_EDGES:
            chosen = students
            prob = 1 / cnt
        else:
            num_rc = self.config.SCHOOL_TC_NUM_RC
            prob = 1 / num_rc
            chosen = math_prob.draw_distinct_uniform(num_rc, students)

        for student in chosen:
            self.add_edge(teacher, student, layer, sublayer, prob, w_layer, cnt)

    def mutual_contacts(self, persons: List, layer, sublayer, w_layer: CalParamIdx, plambda: float = None,
                        probabilities=None,
                        scale=1,
                        anumrc=10):
        persons_list = list(persons)
        cnt = len(persons_list)
        if cnt < 2:
            return
        certain = probabilities is None
        assert certain or len(probabilities) == cnt
        n = cnt - 1
        if certain:
            nn = n
        else:
            nn = sum(probabilities)
        prob = 1 / n

        if plambda:
            prob *= math_prob.lambdan(plambda, nn * scale)

        if cnt <= self.config.MUTUAL_LIMIT and certain:
            for p1, p2 in itertools.combinations(persons, 2):
                self.add_edge(p1, p2, layer, sublayer, scale * prob, w_layer, nn)
        else:
            numrc = anumrc
            if numrc * 2 >= n:
                assert numrc * 2 >= n, f"{numrc}, {n}"
            np = prob * n / numrc / 2.0
            for idx, p1 in enumerate(persons):
                if certain:
                    myprob = 1
                else:
                    myprob = probabilities[idx]
                sel = math_prob.draw_distinct_uniform(numrc, persons)

                for p2 in sel:
                    if p1 is not p2:
                        self.add_edge(p1, p2, layer, sublayer, myprob * np, w_layer, nn)

    def is_edge(self, p1, p2, layer):
        return self.graph.is_edge(p1, p2, layer)

    def add_edge(self, p1, p2, layer: constants.Layer, sublayer: int, probability: float, w_layer: CalParamIdx,
                 w_param=None, intensity=None):
        if p1 is p2:
            assert p1 is not p2

        ld = constants.LAYER_DEFS[layer]
        if not intensity:
            intensity = self.config.INTENSITIES[ld.default_intensity]
        self.graph.add_link(p1, p2, layer, sublayer, probability, w_layer, w_param, intensity)

    def add_export_edge(self, person, export, layer: constants.Layer, sublayer: int, rate: float, intensity=None):
        ld = constants.LAYER_DEFS[layer]
        if not intensity:
            intensity = self.config.INTENSITIES[ld.default_intensity]
        self.graph.add_link(person, export, layer, sublayer, 1, CalParamIdx.ONE, intensity * rate)

    def scaling(self):
        pass

    def find_staff(self, gobj: GObject, num_staff: int, wt: constants.WorkType):
        staff = []
        for _ in range(num_staff):
            employee = self.w.employ(gobj.town, wt, self.travel_info)
            if not employee:
                StdLog.log(f"Not enough stuff found for facility {gobj.id}")
                break
            staff.append(employee)
        return staff

    def staff_customer_contacts(self,
                                staff,
                                customers,
                                layer: constants.Layer,
                                sublayer: int,
                                num: int,
                                probabilities: List[float]):
        if len(staff) < num:
            StdLog.log(f"Not enough staff on sublayer {sublayer}")
            return
        for (idx, cust) in enumerate(customers):
            prob = probabilities[idx] if probabilities else 1
            selected = math_prob.draw_distinct_uniform(num, staff)
            for sel_item in selected:
                if sel_item is not cust:
                    self.add_edge(sel_item, cust, layer, sublayer, prob, CalParamIdx.ONE)
