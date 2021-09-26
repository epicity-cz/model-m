import contextlib
import itertools
import math

import numpy as np

from classes.exports import ExportNode
from classes.friends import FriendNetwork
from classes.g_stat import GStatResult
from classes.households_generator import HouseholdGenerator
from classes.person import Person
from classes.poi import Poi, poi_loader_v1, poi_loader_v2, POI_TYPES, Pub, Shop
from classes.renv import Renv
from classes.school import school_by_type, School
from classes.town import Town
from config.base_config import BaseConfig
from constants import constants
from lang.mytypes import Dict, List, Tuple, Any, Type
from params.cal_param import CalParamIdx
from params.parameters import Parameters
from plos.plos_matrix import PlosMatrix
from plos.plos_track import calibrate
from plos.working_params import WorkingParams
from program import family, working, shopping, loaders, transport
from program.loaders import exports_loader
from program.schools import fill_schools
from program.zujs import zujs_loader
from utils import math_prob
from utils.csv_utils import CsvWriter, flatten
from utils.log import Timer, StdLog, Log

np.seterr(divide='raise')


class Region:
    name: str
    zujs: Dict[str, Town]  # @puml link Town
    towns: List[Town]
    renv: Renv
    exports: List[ExportNode]  # @puml link ExportNode
    townconnections: Dict[Town, Dict[Town, float]]
    config: BaseConfig
    townconn: List[Tuple[Town, Town, Dict]]
    travel_info: Any
    schools: Dict[constants.SchoolType, List[School]]
    students: Dict[constants.SchoolType, List[Person]]
    pois: Dict[Type[Poi], List[Poi]]
    parameters: Parameters

    def __init__(self, region_config: BaseConfig):
        self.config = region_config
        self.name = self.config.NAME
        self.towns = []
        self.exports = []
        self.zujs = {}
        self.townconn = []
        self.travel_info = None
        self.schools = {st: [] for st in constants.SchoolType}
        self.students = {st: [] for st in constants.SchoolType}
        self.pois = {key: [] for key in POI_TYPES}

    @staticmethod
    def generate(region_config: BaseConfig):
        logger = Log(region_config.log_report())
        with Timer('Graph', StdLog):
            reg = Region(region_config)
            reg.init()
            reg.run(logger)
            reg.list()

    def init(self):
        with Timer(f'Adding towns'):
            self.add_towns()

        self.renv = Renv(self.config, self.towns, self.exports)

        with Timer('Adding persons'):
            if self.config.LOAD_PERSONS:
                self.load_persons(self.config.persons())
            else:
                self.generate_persons(self.config.households())

        StdLog.log('Assigning households to apartments')
        for town in self.towns:
            town.place_households()

        StdLog.log(f"Total persons {self.count_persons()}")

        self.load_pois()

        self.load_schools(self.config.schools())
        StdLog.log(f"Schools {[(st.name, len(self.schools[st])) for st in constants.SchoolType]}")

        StdLog.log("Shuffling workers")
        self.renv.w.shuffle()

        self.load_exports()

        for town in self.towns:
            town.info()

        # init graphs

    def run(self, logger):
        with Timer('schools'):
            self.run_school(logger)
        with Timer('leisure'):
            self.leisure()
        with Timer('family'):
            self.run_family(logger)
        with Timer('shopping'):
            self.run_shopping(logger)
        with Timer('work'):
            self.run_work(logger)
        with Timer('other'):
            self.other()
        with Timer('tstat'):
            self.tstat()
        with Timer('transport'):
            self.transport()
        with Timer('calibrate'):
            self.calibrate(logger)
        with Timer('scaling'):
            self.renv.scaling()

        with Timer('football'):
            self.football()
        with Timer('party 0'):
            self.party(0)
        with Timer('party 1'):
            self.party(1)
        with Timer('party 2'):
            self.party(2)

        res = self.gstat(logger)

        StdLog.log(f"Scaling by {res.degree}")
        self.renv.scale = 1 / res.degree

    def run_school(self, logger: Log):
        logger.log("School")
        fill_schools(self.renv, self.schools, self.students, Log(self.config.log_school()))

    def add_towns(self):
        for (zuj, orp, cnt, x, y, nazev) in zujs_loader(self.config):
            if self.config.contains(zuj, orp):
                with Timer(f'Reading town {zuj}'):
                    town = Town(zuj, self.config.town_dump(zuj), self.config.download())
                    town.load_apartments(self.config.buildings(town.zuj))
                    self.add_town(town)
                town.location = (x, y)
                town.name = nazev
                StdLog.log(f'Town {zuj} - {nazev}')

    def add_town(self, town):
        assert town.valid()
        self.zujs[town.zuj] = town
        self.towns.append(town)

    def load_persons(self, filename):
        last_key = 0
        for (zuj, sex, age, key) in loaders.osoby_loader(filename):
            town = self.zujs.get(zuj)
            if not town:
                StdLog.log(f"Ignoring Person OOR {key}", StdLog.DEBUG)
                continue
            p = self.person_factory(age, sex, town)
            if key != last_key:
                if key < last_key:
                    raise Exception("keys have to be ordered ascendantly")
                last_key = key
                town.household_factory()
            town.person_household(p)

    def generate_persons(self, filename):
        hg = HouseholdGenerator(filename, self.config.CSU_MAX_HH_SIZE, self.config.MAX_AGE)
        for town in self.towns:
            for (age, sex, new_household) in hg.generate_persons_town(town):
                p = self.person_factory(age, sex, town)
                if new_household:
                    town.household_factory()
                town.person_household(p)

    def person_factory(self, age, sex, town):
        p = Person(self.config, age, sex, town)
        self.renv.w.add_person(p, town)
        town.add_person(p)
        st = p.school_type
        if st is not None:
            self.students[st].append(p)
        return p

    def load_pois(self):
        if self.config.POI_VERSION == 2:
            poi_loader = poi_loader_v2
        else:
            poi_loader = poi_loader_v1
        for (zuj, poi) in poi_loader(self.config.poi()):
            town = self.zujs.get(zuj)
            if town:
                poi.town = town
                self.pois[type(poi)].append(poi)
            else:
                StdLog.log(f"Ignoring POI OOR {poi.name}", StdLog.DEBUG)
        for poi in self.pois:
            StdLog.log(f"{poi.__name__} : {len(self.pois[poi])}")

    def load_schools(self, filename):
        for (zuj, name, type, capacity, x, y, w) in loaders.schools_loader(filename):
            town = self.zujs.get(zuj)
            if town:
                school = school_by_type[type]([x, y], name, w, capacity, town)
                self.schools[type].append(school)
        if not len(self.schools[constants.SchoolType.NURSARY]):
            raise Exception("No nursary school")
        if not len(self.schools[constants.SchoolType.ELEMENTARY]):
            raise Exception("No elem school")

    def load_exports(self):
        if self.config.SINGLE_EXPORT:
            self.exports.append(ExportNode('single_export_noode'))
        else:
            for (code, name) in exports_loader(self.config.exports()):
                self.exports.append(ExportNode(name, code))
        StdLog.log(f"{len(self.exports)} export places added")

    def leisure(self):
        # co potrebuje
        # person: age, sex
        # distance

        ntwrk = FriendNetwork(self.all_persons(), self.config)
        StdLog.log(f"Edges= {len(ntwrk.graph.edges)}")
        StdLog.log(f"local cluster coef: {ntwrk.local_cluster(self.config.FRIENDS_SAMPLE)}")
        ntwrk.add_edges(self.renv.graph, StdLog, self.renv, self.pois[Pub])

        with CsvWriter(self.config.out_friends_nodes()) as f:
            f.writerow(['id', 'sex', 'age'])
            for n in ntwrk.graph.nodes():
                f.writerow([n.person.id, n.person.sex.value, n.person.age])

        with CsvWriter(self.config.out_friends_edges()) as f:
            f.writerow(['v1', 'v2'])
            for (u, v, params) in ntwrk.graph.edges(data=True):
                if params['main']:
                    f.writerow([u.person.id, v.person.id])

    def run_family(self, logger: Log):
        family.run(self.renv, list(self.all_apartments()), self.config)

    def run_shopping(self, logger: Log):
        shopping.run(self.renv, self.pois[Shop], self.all_persons(), self.config.out_shop_visits())

    def run_work(self, logger: Log):
        return
        # ignored
        wp = WorkingParams(0.6)
        logger.log(f"Calibrating work param: {wp}")
        wenv = self.renv.copy()
        num_workers = wenv.plos_track[constants.PlosCats.WORK].population()
        logger.log(f"Workers already assigned {num_workers}")
        working.run(wenv, wp, self.config, self.all_persons())
        total_workers = wenv.plos_track[constants.PlosCats.WORK].population()
        logger.log(f"Total workers assigned {total_workers}")
        wref = PlosMatrix()
        wref.read(self.config.plos_data(constants.PlosCats.WORK), wenv.plos_track[constants.PlosCats.WORK].nums)

        wp.scale(wref.average() / wenv.plos_track[constants.PlosCats.WORK].average() * self.config.WORK_RATIO_ADJUST)
        logger.log(f"Using work param:  {wp}")
        working.run(self.renv, wp, self.config, self.all_persons())

    def transport(self):
        self.renv.travel_info.transport(self.renv)
        transport.run(self.config.busy_sections())

    def calibrate(self, logger: Log):
        self.parameters = calibrate(self.renv.graph, self.config)

    def other(self):
        all_persons = list(self.all_persons())
        for person in all_persons:
            n = 0
            while n < 3:
                prob = self.config.OTHER_PROBABILITY
                if math_prob.yes_no(self.config.OTHER_IS_EXPORT):
                    self.renv.add_export_edge(constants.TravelType.OTHER, person, constants.Layer.OTHER,
                                              constants.NO_SUBLAYER, prob, CalParamIdx.ONE)
                    n += 1
                else:
                    other = math_prob.choice(all_persons)
                    if other is not person:
                        acc = math.exp(-abs(person.age - other.age) / self.config.OTHER_AGE_ADJUST)
                        if person.town is other.town:
                            acc *= self.config.OTHER_SAME_TOWN_ADJUST
                        if math_prob.yes_no(acc):
                            self.renv.add_edge(person, other, constants.Layer.OTHER, constants.NO_SUBLAYER, prob,
                                               CalParamIdx.ONE)
                            if math_prob.yes_no(.5):
                                t1 = person.town
                                t2 = other.town
                                pp = person
                            else:
                                t1 = other.town
                                t2 = person.town
                                pp = other
                            self.renv.travel_info.add_travel(constants.TravelType.OTHER, t1, t2, pp, prob)
                            n += 1

    def tstat(self):
        pass

    def football(self):
        # zatim vynechavam
        pass

    def party(self, index):
        # zatim vynechavam
        pass

    def list(self):
        self.list_nodes()
        self.list_edges()
        self.list_objects()
        self.list_nums()
        self.list_etype()

    def list_nodes(self):
        # SRC: region.hpp:2912-2918
        with CsvWriter(self.config.FILE_NODES) as f:
            f.writerow(['id', 'sex', 'age', 'ecactivity', 'worktype', 'commutetime', 'town', 'x', 'y', 'apartment'])
            for p in self.all_persons():
                ap = p.household.apartment
                f.writerow([
                    p.id,
                    ['M', 'F'][p.sex],
                    p.age,
                    p.activity.label(),
                    p.worktype.label(),
                    p.commute.label(),
                    p.town.zuj,
                    ap.location[0],
                    ap.location[1],
                    ap.id
                ])
        with CsvWriter(self.config.FILE_EXPORT) as f:
            f.writerow(['id', 'name', 'code'])
            for n in self.exports:
                f.writerow([n.id, n.name, n.code])

    def list_edges(self, cats=False):
        if cats:
            raise Exception("NIY")
        self.renv.graph.list_edges(self.config.FILE_EDGES, self.renv.scale, self.parameters)

    def list_objects(self):
        with CsvWriter(self.config.FILE_OBJECTS) as f:
            f.writerow(['id', 'x', 'y', 'town', 'type', 'data'])
            for town in self.towns:
                for apt in town.apartments:
                    f.writerow([apt.id, apt.location[0], apt.location[1], town.name, 'apartment', '{}'])

    def list_nums(self):
        with CsvWriter(self.config.FILE_NUMS) as f:
            f.writerow(['persons', 'exports', 'layers', 'apartments'])
            f.writerow([self.count_persons(), len(self.exports), len(constants.Layer),
                        sum(len(town.apartments) for town in self.towns)])

    def list_etype(self):
        with CsvWriter(self.config.FILE_LAYERS) as f:
            f.writerow(['id', 'name', 'weight'])
            for layer in constants.Layer:
                f.writerow([layer.value, layer, 1])

    def all_persons(self):
        return itertools.chain.from_iterable((town.all_persons() for town in self.towns))

    def count_persons(self):
        return sum((town.count_persons() for town in self.towns))

    def all_apartments(self):
        return itertools.chain.from_iterable((town.apartments for town in self.towns))

    def gstat(self, logger: Log):
        result = GStatResult()

        with contextlib.ExitStack() as stack:
            out_categories = []
            num_edges = 0  # numedges
            sum_probabilities = 0  # ps
            sum_weights = 0  # wps
            layer_probabilities = np.zeros(len(constants.Layer))
            layer_weights = np.zeros(len(constants.Layer))
            our_probabilities = np.zeros(len(constants.Layer))
            our_weights = np.zeros(len(constants.Layer))

            layer_edges = np.zeros(len(constants.Layer), dtype=int)

            for cat in constants.OurCats:
                cvscat = stack.enter_context(CsvWriter(self.config.out_cat_graph(cat)))
                out_categories.append(cvscat)
                cvscat.writerow(['age1', 'sex1', 'age2', 'sex2', 'dist'])

            with CsvWriter(self.config.out_gstat()) as out_gstat:
                out_gstat.writerow(
                    ['axe',
                     'sex',
                     *flatten([[f"{cat.label()}p", f"{cat.label()}r"] for cat in constants.OurCats])
                     ])

                active_persons = set()

                for (p, q, link) in self.renv.graph.links():
                    layer = link.layer
                    prob = link.probability_real(self.parameters)
                    weight = link.intensity * prob

                    sum_probabilities += prob
                    sum_weights += weight

                    layer_probabilities[layer] += prob
                    layer_weights[layer] += weight

                    our_category = constants.LAYER_DEFS[layer].our_category
                    our_probabilities[our_category] += prob
                    our_weights[our_category] += weight

                    layer_edges[layer] += 1
                    num_edges += 1
                    active_persons.add(p)
                    ld = constants.LAYER_DEFS[layer]
                    pm1 = result.pm[ld.first]
                    pma = result.pm[constants.PlosCats.ALL]

                    if isinstance(q, Person):
                        active_persons.add(q)

                        pm1.add_contact(p.age, q.age, prob)
                        pm1.add_person(p, p.age)

                        pm2 = result.pm[ld.second]
                        pm2.add_contact(q.age, p.age, prob)
                        pm2.add_person(q, q.age)

                        pma.add_contact(p.age, q.age, prob)
                        pma.add_person(p, p.age)
                    else:
                        pm1.add_external(p.age, prob)
                        pma.add_external(p.age, prob)

        npers = self.count_persons()

        logger.log("Statistics of graphs")
        logger.log(f"Number of edges,{num_edges}")
        logger.log("")
        logger.log("Potential contacts")
        logger.log(f"mean,{num_edges / npers}")
        logger.log(f"mean given nonzero,{num_edges / len(active_persons)}")
        logger.log("")
        logger.log("Expected contacts")
        logger.log(f"mean,{sum_probabilities / npers},weighted,{sum_weights / npers}")
        logger.log(f"mean given nonzero,{sum_probabilities / len(active_persons)}")

        for ploscat in constants.PlosCats:
            result.ref[ploscat].read(self.config.plos_data(ploscat), result.pm[ploscat].nums)

            logger.log(f"Reference {ploscat.label()}")
            logger.delimited([
                "Mean", result.pm[ploscat].average(),
                "ref", result.ref[ploscat].average(),
                "dist", result.pm[ploscat].pmdist(result.ref[ploscat]),
                "tdist", result.pm[ploscat].pmtdist(result.ref[ploscat])
            ])

            result.pm[ploscat].plot_bar(result.ref[ploscat], self.config.pic_bar(ploscat))
            result.pm[ploscat].plot(self.config.pic(ploscat))
            result.ref[ploscat].plot(self.config.pic_ref(ploscat))
        logger.log("Layers weights")
        StdLog.log("layer,aveintensity")

        for layer in constants.Layer:
            ld = constants.LAYER_DEFS[layer]
            logger.delimited([
                ld.our_category.label(),
                layer.label(),
                ld.default_intensity.label(),
                f"${round(layer_probabilities[layer] / npers, 3)}$",
                f"${round(layer_weights[layer] / npers, 3)}$",
                f"${round(layer_weights[layer] / npers, 3)}$",
                f"${layer_edges[layer]}$",
            ], '&')
            if layer_edges[layer] > 0:
                total = layer_weights[layer] / layer_edges[layer]
            else:
                total = '-'
            StdLog.log(f"{layer.label()},{total}")

        logger.log("Our categories")
        logger.log("Category,contacts,infectiousness")
        for oc in constants.OurCats:
            logger.delimited([
                oc.label(),
                our_probabilities[oc] / npers,
                our_weights[oc] / npers
            ])
        logger.log("\\\\")
        result.degree = sum_weights / npers
        return result
