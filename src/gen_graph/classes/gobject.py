from lang.mytypes import Location, List, Any
from utils import math_prob
from utils.geography import distance
from utils.has_id import HasId


class GObject(HasId):
    location: Location
    town: Any

    def __init__(self, location: Location):
        self.location = location
        self.gen_id()

    def distance_to(self, other):
        if isinstance(other, GObject):
            location = other.location
        else:
            location = other
        return distance(self.location, location)

    def choose_object(self, lst: List[Any], distpref: float):
        ws = [math_prob.distw(self.distance_to(poi), distpref) * poi.weight for poi in lst]
        return math_prob.draw(ws, lst)

    @classmethod
    def get_type(cls):
        return 'gobject'

    @classmethod
    def dump_header(cls):
        return ['x', 'y']

    def dump(self):
        return [self.location[0], self.location[1]]
