import math

from lang.mytypes import Location


def distance(loc1: Location, loc2: Location):
    return math.sqrt((loc1[0] - loc2[0]) ** 2 +
                     (loc1[1] - loc2[1]) ** 2)
