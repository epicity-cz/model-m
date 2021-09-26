from classes.gobject import GObject
from constants import constants
from lang.mytypes import Location
from program.loaders import CsvReader, UTF8


class Poi(GObject):
    name: str
    address: str
    weight: float

    def __init__(self, location: Location, r1: str, r2: str, weight: float, par1=None, par2=None):
        GObject.__init__(self, location)
        self.name = r1 if r1 else f"ID_{self.id}"
        self.address = r2
        if weight == '':
            self.weight = 1
        else:
            self.weight = float(weight)

    def valid(self):
        return True


class Shop(Poi):
    capacity: float
    type: constants.ShopType

    shop_sub_type = {
        's': constants.ShopType.SMALLSHOP,
        'm': constants.ShopType.MEDIUMSHOP,
        'p': constants.ShopType.SUPERMARKET,
        'h': constants.ShopType.HYPERMARKET,
        'u': constants.ShopType.INVALID,
    }

    def __init__(self, location: Location, r1: str, r2: str, weight: float, par1, par2):
        Poi.__init__(self, location, r1, r2, weight)

        self.capacity = par2
        sst = Shop.shop_sub_type.get(par1)
        if sst is None:
            raise Exception("unknown shoptype")
        self.type = sst

    def valid(self):
        return self.type != constants.ShopType.INVALID


class Pub(Poi):
    pass


class Church(Poi):
    pass


POI_TYPES = [Shop, Pub, Church]


def poi_loader_v1(filename):
    poi_type = {
        'supermarket': Shop,
        'bar': Pub,
        'pub': Pub,
        'restaurant': Pub,
        'coffee_shop': Pub,
        'church': Church,
    }
    ignored = []

    for (tp, zuj, r1, r2, x, y, w, subt, turn) in \
            CsvReader(filename,
                      [0, 1, 5, 6, 7, 8, 10, 12, 15],
                      ["type", "zuj_code", "lat", "lon", "name", "first_row", "second_row", "x", "y", "source",
                       "weight", "weight_note", "subtype", "subtype_note", "subtype_source", "turnout",
                       "turnout_weekends", "turnout_note", "turnout_source"],
                      UTF8
                      ):
        cls = poi_type.get(tp)
        if cls:
            poi = cls((float(x), float(y)), r1, r2, w, subt, turn)
            if poi.valid():
                yield zuj, poi
        else:
            if not tp in ignored:
                print(f"ignoring {tp}")
                ignored.append(tp)


def poi_loader_v2(filename):
    poi_type = {
        'shop': Shop,
        'pub': Pub,
    }
    for (tp, zuj, x, y, subt) in \
            CsvReader(filename,
                      [0, 1, 4, 5, 6],
                      ['type', 'zuj', 'lat', 'lon', 'x', 'y', 'par'],
                      UTF8
                      ):
        cls = poi_type.get(tp)
        if cls:
            poi = cls((float(x), float(y)), None, None, 1, subt, 0)
            if poi.valid():
                yield zuj, poi
