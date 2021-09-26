from classes.poi import Poi
from constants.constants import SchoolType
from lang.mytypes import List, Any
from utils.has_id import HasId


class School(Poi, HasId):
    capacity: int
    teachers: List
    town: Any
    NEAREST = None
    ALLOW_TRAVEL = None

    def __init__(self, location, r1, w, capacity, town):
        Poi.__init__(self, location, r1, '', w)
        self.capacity = capacity
        self.teachers = []
        self.town = town


class SchoolNursary(School):
    NEAREST = True
    ALLOW_TRAVEL = False
    _ID_CLASS = School


class SchoolElementary(School):
    NEAREST = True
    ALLOW_TRAVEL = True
    _ID_CLASS = School


class SchoolHigh(School):
    NEAREST = False
    ALLOW_TRAVEL = True
    _ID_CLASS = School


school_by_type = {
    SchoolType.NURSARY: SchoolNursary,
    SchoolType.ELEMENTARY: SchoolElementary,
    SchoolType.HIGH: SchoolHigh,
}
