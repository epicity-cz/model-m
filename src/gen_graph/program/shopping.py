from collections import defaultdict

from classes.person import Person
from classes.poi import Shop
from classes.renv import Renv
from constants import constants, prokop
from lang.mytypes import Iterator
from params.cal_param import CalParamIdx
from utils import math_prob
from utils.csv_utils import CsvWriter
from utils.log import StdLog

shop_type2prokop = {
    constants.ShopType.SMALLSHOP: prokop.ProkopTable.SMALL_SHOP,
    constants.ShopType.MEDIUMSHOP: prokop.ProkopTable.SUPERMARKET,
    constants.ShopType.SUPERMARKET: prokop.ProkopTable.HYPERMARKET,
    constants.ShopType.HYPERMARKET: prokop.ProkopTable.HYPERMARKET
}

dist_prefs = {
    prokop.ProkopTable.SMALL_SHOP: 0.6,
    prokop.ProkopTable.SUPERMARKET: 0.4,
    prokop.ProkopTable.HYPERMARKET: 0.2,
}

rates = {
    prokop.ProkopTable.SMALL_SHOP: 1,
    prokop.ProkopTable.SUPERMARKET: 1.5,
    prokop.ProkopTable.HYPERMARKET: 1.5,
}

# see https://github.com/epicity-cz/mgraph/issues/5
staff_size = {
    prokop.ProkopTable.SMALL_SHOP: 2,
    prokop.ProkopTable.SUPERMARKET: 4,
    prokop.ProkopTable.HYPERMARKET: 10,
}

alpha = {
    prokop.ProkopTable.SMALL_SHOP: 1.0 / (10.0 / (10.0 / 60.0)),
    prokop.ProkopTable.SUPERMARKET: 1.0 / (10.0 / ((13 + 22.0 / 60.0) / 60.0)) / 2.0,
    prokop.ProkopTable.HYPERMARKET: 1.0 / (16 / ((19 + 52.0 / 60.0) / 60.0)) / 4.0,
}


def run(env: Renv, shops: Iterator[Shop], persons: Iterator[Person], visits_file):
    StdLog.log("Assigning customers to shops")
    shops_by_type = defaultdict(list)
    for shop in shops:
        shops_by_type[shop_type2prokop[shop.type]].append(shop)
    shops_persons = defaultdict(lambda: defaultdict(list))
    shops_probabilities = defaultdict(lambda: defaultdict(list))
    for person in persons:
        for type in shops_by_type:
            prob = prokop.prob_visit(type, person)
            if math_prob.yes_no(prob):
                shop = person.household.apartment.choose_object(shops_by_type[type], dist_prefs[type])
                shops_persons[type][shop].append(person)
                shops_probabilities[type][shop].append(prob)
    StdLog.log("Creating shopping graph connections")

    with CsvWriter(visits_file) as f:
        f.writerow(['type', 'name', 'visits', 'capacity'])
        for type in shops_by_type:
            for shop in shops_by_type[type]:
                for (idx, person) in enumerate(shops_persons[type][shop]):
                    env.travel_info.add_travel(constants.TravelType.SHOPPING, person.town, shop.town, person,
                                               shops_probabilities[type][shop][idx])
                env.mutual_contacts(shops_persons[type][shop], constants.Layer.SHOPS_CUSTOMERS, shop.id,
                                    CalParamIdx.ONE,
                                    rates[type],
                                    shops_probabilities[type][shop], alpha[type])
                staff = env.find_staff(shop, staff_size[type], constants.WorkType.TRADE_AND_MOTO)
                env.staff_customer_contacts(staff, shops_persons[type][shop], constants.Layer.SHOPS_WORKERS_TO_CLIENTS,
                                            shop.id, 1,
                                            shops_probabilities[type][shop])
                f.writerow([type.label(), shop.name, sum(shops_probabilities[type][shop]), shop.capacity])
    StdLog.log("Done")
