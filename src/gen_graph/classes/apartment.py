import itertools

from classes.gobject import GObject
from classes.household import Household
from lang.mytypes import List, Location, Any


class Apartment(GObject):
    households: List[Household]
    fh: bool

    def __init__(self, town: Any, location: Location, fh):
        GObject.__init__(self, location)
        self.fh = fh
        self.households = []
        self.town = town

    def add_household(self, hh: Household):
        self.households.append(hh)
        hh.apartment = self

    def all_persons(self):
        return itertools.chain.from_iterable((hh.persons for hh in self.households))

    def count_persons(self):
        return sum((len(hh.persons) for hh in self.households))
