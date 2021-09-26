import math
from collections import defaultdict

import networkx

from classes.mgraph import MGraph
from classes.person import Person
from classes.renv import Renv
from config.base_config import BaseConfig
from constants import constants, prokop
from lang.mytypes import Iterator
from params.cal_param import CalParamIdx
from utils import math_prob
from utils.csv_utils import CsvWriter
from utils.log import Log


class FriendLink:
    person: Person
    theorindegree: int
    actualindegree: int

    def __init__(self, person: Person):
        self.person = person
        self.theorindegree = 0
        self.actualindegree = 0


class FriendNetwork():
    graph: networkx.Graph
    config: BaseConfig

    def __init__(self, population: Iterator[Person], config: BaseConfig):
        self.config = config
        friends_nodes = [FriendLink(person) for person in population if person.age >= self.config.FIRST_IN_FRIENDS]
        math_prob.shuffle(friends_nodes)
        self.graph = networkx.Graph()
        self.graph.add_nodes_from(friends_nodes)

        weights = []
        for m, fl in enumerate(friends_nodes):
            if not m % 1000:
                print(m)
            n = max(math_prob.poisson(self.config.FRIENDS_LAMBDA), 1)
            if m:
                weights.append(1)
            if n > m:
                continue

            p = fl.person

            ss = min(2 * n, m)
            ns = math_prob.draw_distinct_weighted(weights, ss)
            pda = []
            distances = []
            for item in ns:
                q = friends_nodes[item].person
                acc = pow(config.TEN_YRS_ACCEPT, abs(p.age - q.age) / 10)
                if p.sex != q.sex:
                    acc *= config.OPP_SEX_ACCEPT

                distance = p.household.apartment.distance_to(q.household.apartment)
                acc *= math_prob.distw(distance, config.FRIEND_DIST_PREF)
                pda.append(acc)
                distances.append(distance)
            ks = math_prob.draw_distinct_weighted(pda, n)
            for item in ks:
                ni = friends_nodes[ns[item]]
                if ni.actualindegree == ni.theorindegree and math_prob.yes_no(config.FRIENDS_ACCEPT_EDGE):
                    self.graph.add_edge(fl, ni, main=True, distance=distances[item])
                    ni.actualindegree += 1
                ni.theorindegree += 1
                weights[ns[item]] += 1
            if len(ks) > 2:
                nr = math_prob.draw_distinct_uniform(2, ks)
                p = friends_nodes[ns[nr[0]]]
                q = friends_nodes[ns[nr[1]]]
                distance = p.person.household.apartment.distance_to(q.person.household.apartment)
                self.graph.add_edge(p, q, main=False, distance=distance)

    def add_edges(self, graph: MGraph, logger: Log, renv: Renv, pubs):
        dist_sum = 0
        counter = 0
        # key: pub, value tuple (list of people, list of their respective probabilities)
        inpubs = defaultdict(lambda: ([], []))
        for (u, v, data) in self.graph.edges(data=True):
            if (math_prob.yes_no(.5)):
                (u, v) = (v, u)
            p = u.person
            q = v.person
            if graph.is_edge(p, q, constants.Layer.LEISURE_PUB) or \
                    graph.is_edge(p, q, constants.Layer.LEISURE_VISIT) or \
                    graph.is_edge(p, q, constants.Layer.LEISURE_OUTDOOR) or \
                    p.household.apartment is q.household.apartment:
                # logger.log("Adding omitted")
                continue
            degree = max(self.graph.degree(u), self.graph.degree(v))
            prob = self.config.FRIENDS_KAPPA * (1 - math.exp(-self.config.FRIENDS_KAPPA * degree)) / degree
            if (prob > 1):
                prob = 1
            dist_sum += data['distance']
            counter += 1

            renv.travel_info.add_travel(constants.TravelType.LEISURE, p.town, q.town, p, prob)

            if p.age < self.config.FRIENDS_MIN_AGE_PUB or q.age < self.config.FRIENDS_MIN_AGE_PUB:
                probpub = 0
            else:
                probpub = (prokop.prob_visit(prokop.ProkopTable.PUBS_EVENING, p) +
                           prokop.prob_visit(prokop.ProkopTable.PUBS_EVENING, q)) / 2

            akce = math_prob.random() - probpub

            if akce < 0:
                # pub
                pub = p.household.apartment.choose_object(pubs, self.config.FRIENDS_PUB_DIST_PREF)
                renv.add_edge(p, q, constants.Layer.LEISURE_PUB, pub.id, prob, CalParamIdx.ONE)
                pbi = inpubs[pub]
                pbi[0].append(p)
                pbi[1].append(prob)
                pbi[0].append(q)
                pbi[1].append(prob)

            else:
                if akce > (1 - probpub) / 2:
                    # outdoor
                    renv.add_edge(p, q, constants.Layer.LEISURE_OUTDOOR, constants.NO_SUBLAYER, prob, CalParamIdx.ONE)
                else:
                    # visit
                    for inh in q.household.persons:
                        renv.add_edge(p, inh, constants.Layer.LEISURE_VISIT, q.household.apartment.id, prob,
                                      CalParamIdx.ONE)

        with CsvWriter(self.config.out_pub_visits()) as f:
            for pub in inpubs:
                (visitors, probabilities) = inpubs[pub]
                renv.mutual_contacts(visitors, constants.Layer.PUBS_CUSTOMERS, pub.id, CalParamIdx.ONE,
                                     self.config.EATING_RATE,
                                     probabilities)
                staff = renv.find_staff(pub, 2, constants.WorkType.EATING_HOUSING)
                renv.staff_customer_contacts(staff, visitors, constants.Layer.PUBS_WORKERS_TO_CLIENTS, pub.id, 1,
                                             probabilities)
                f.writerow([pub.name, sum(probabilities)])

        logger.log(f"Friend network of {len(self.graph.nodes)} "
                   f"num friend connections {counter} "
                   f"degree {counter / len(self.graph.nodes)} "
                   f"ave dist {dist_sum / counter} ")

    def local_cluster(self, sample):
        sample_max = len(self.graph.nodes()) / 1.2
        if sample > sample_max:
            print(f"graph has {len(self.graph.nodes())} vertices")
            print(f"degrading sample size from {sample} to {sample_max}")
            sample = sample_max

        cnt = 0
        nodes = list(self.graph.nodes())
        counter = sample
        while (counter):
            nd = math_prob.choice(nodes)
            edges = self.graph.edges([nd])
            if len(edges) > 1:
                counter -= 1
                [(n1, u), (n2, v)] = math_prob.sample(list(edges), 2)
                if self.graph.has_edge(u, v):
                    cnt += 1
        return cnt / sample
