from lang.mytypes import EnumZ, auto, Enum


class Gender(EnumZ):
    MEN = auto()
    WOMEN = auto()

    def label(self):
        return ['Men', 'Women'][self.value]


class EconoActivity(EnumZ):
    WORKING = auto()
    AT_HOME = auto()
    STUDENT = auto()
    RETIRED = auto()


class CommutingTime(EnumZ):
    # @formatter:off
    NOT_COMMUTING = auto()
    ONE_QUATER    = auto()
    HALF          = auto()
    THREE_QUATER  = auto()
    HOUR          = auto()
    HOUR_AND_HALF = auto()
    MORE          = auto()
    # @formatter:on

    def label(self):
        return [
            "not commuting",
            "less than 15min",
            "15 to 30 min",
            "30 to 45 mins",
            "45 to 60 mins",
            "60 to 90 mins",
            "more than 90 mins"
        ][self.value]


class WorkType(EnumZ):
    # @formatter:off
    AGRICULTURE    = auto()
    INDUSTRY       = auto()
    CONSTRUCTION   = auto()
    TRADE_AND_MOTO = auto()
    TRANSPORTATION = auto()
    EATING_HOUSING = auto()
    IT             = auto()
    BANKS          = auto()
    ADMIN_ETC      = auto()
    PUBLIC_SECTOR  = auto()
    EDUCATION      = auto()
    HEALTH         = auto()
    UNDEF          = -1
    # @formatter:on


WORK_AGE_CATS = [15, 25, 35, 45, 55, 60, 65]


def get_age_idx(age):
    for i in range(len(WORK_AGE_CATS) - 1, -1, -1):
        if WORK_AGE_CATS[i] <= age:
            return i
    return None


class ShopType(EnumZ):
    # @formatter:off
    SMALLSHOP   = auto()
    MEDIUMSHOP  = auto()
    SUPERMARKET = auto()
    HYPERMARKET = auto()
    INVALID = auto()
    # @formatter:on


class POIType(EnumZ):
    # @formatter:off
    SUPERMARKET = auto()
    PUB         = auto()
    CHURCH      = auto()
    # @formatter:on


class SchoolType(EnumZ):
    # @formatter:off
    NURSARY    = auto()
    ELEMENTARY = auto()
    HIGH       = auto()
    # @formatter:on


class ExportType(Enum):
    PLACE = auto()


class Layer(EnumZ):
    # @formatter:off
    #    NO_LAYER = 0
    FAMILY_INSIDE                          = auto()
    FAMILY_IN_HOUSE                        = auto()
    FAMILY_VISITSORS_TO_VISITED            = auto()
    NURSARY_CHILDREN_INCLASS               = auto()
    NURSARY_TEACHERS_TO_CHILDREN           = auto()
    LOWER_ELEMENTARY_CHILDREN_INCLASS      = auto()
    LOWER_ELEMENTARY_TEACHERS_TO_CHILDREN  = auto()
    HIGHER_ELEMENTARY_CHILDREN_INCLASS     = auto()
    HIGHER_ELEMENTARY_TEACHERS_TO_CHILDREN = auto()
    HIGHSCHOOL_CHILDREN_INCLASS            = auto()
    HIGHSCHOOL_TEACHERS_TO_CHILDREN        = auto()
    NURSARY_CHILDREN_CORIDORS              = auto()
    ELEMENTARY_CHILDREN_CORIDORS           = auto()
    HIGHSCHOOL_CHILDREN_CORIDORS           = auto()
    NURSARY_TEACHERS                       = auto()
    ELEMENTARY_TEACHERS                    = auto()
    HIGHSCHOOL_TEACHERS                    = auto()
    LEISURE_OUTDOOR                        = auto()
    LEISURE_VISIT                          = auto()
    LEISURE_PUB                            = auto()
    WORK_CONTACTS                          = auto()
    WORK_WORKERS_TO_CLIENTS_DISTANT        = auto()
    WORK_WORKERS_TO_CLIENTS_PHYSICAL_SHORT = auto()
    WORK_WORKERS_TO_CLIENTS_PHYSICAL_LONG  = auto()
    PUBLIC_TRANSPORT                       = auto()
    SHOPS_CUSTOMERS                        = auto()
    SHOPS_WORKERS_TO_CLIENTS               = auto()
    PUBS_CUSTOMERS                         = auto()
    PUBS_WORKERS_TO_CLIENTS                = auto()
    OTHER                                  = auto()
    SUPERSPREADER                          = auto()
    FOOTBAL                                = auto()
    PARTY_1                                = auto()
    PARTY_2                                = auto()
    PARTY_3                                = auto()
    # @formatter:on


NO_SUBLAYER = -1


class PlosCats(EnumZ):
    # @formatter:off
    HOME     = auto()
    WORK     = auto()
    SCHOOL   = auto()
    OTHER    = auto()
    ALL      = auto()
    # @formatter:on


class OurCats(EnumZ):
    # @formatter:off
    FAMILY    = auto()
    SCHOOL    = auto()
    WORK      = auto()
    PUBS      = auto()
    VISITS    = auto()
    OUTDOOR   = auto()
    SHOPPING  = auto()
    SERVICES  = auto()
    TRANSPORT = auto()
    OTHER     = auto()
    # @formatter:on


class Intensities(EnumZ):
    # @formatter:off
    CLOSE_LONG_TERM     = auto()
    PHYSICAL_LONG_TERM  = auto()
    PHYSICAL_SHORT_TERM = auto()
    DININING            = auto()
    DISTANT_LONG_TERM   = auto()
    CLOSE_RANDOM        = auto()
    SERVICE_CLIENT      = auto()
    CLOSE_OPEN_AIR      = auto()
    # @formatter:on


class LayerAttrs(EnumZ):
    # @formatter:off
    CAT_FIRST  = auto()
    CAT_SECOND = auto()
    OUR_CAT    = auto()
    INTENSITY  = auto()
    # @formatter:on


class LayerDef():
    first: PlosCats
    second: PlosCats
    our_category: OurCats
    default_intensity: Intensities

    def __init__(self, first: PlosCats, second: PlosCats, our_category: OurCats, default_intensity: Intensities):
        self.first = first
        self.second = second
        self.our_category = our_category
        self.default_intensity = default_intensity


# @formatter:off
LAYER_DEFS = [
    LayerDef(PlosCats.HOME     , PlosCats.HOME     , OurCats.FAMILY    , Intensities.CLOSE_LONG_TERM),     # FAMILY_INSIDE
    LayerDef(PlosCats.HOME     , PlosCats.HOME     , OurCats.FAMILY    , Intensities.DISTANT_LONG_TERM),   # FAMILY_IN_HOUSE
    LayerDef(PlosCats.OTHER    , PlosCats.HOME     , OurCats.FAMILY    , Intensities.CLOSE_LONG_TERM),     # FAMILY_VISITSORS_TO_VISITED
    LayerDef(PlosCats.OTHER    , PlosCats.OTHER    , OurCats.SCHOOL    , Intensities.CLOSE_LONG_TERM),     # NURSARY_CHILDREN_INCLASS
    LayerDef(PlosCats.WORK     , PlosCats.OTHER    , OurCats.SCHOOL    , Intensities.CLOSE_LONG_TERM),     # NURSARY_TEACHERS_TO_CHILDREN
    LayerDef(PlosCats.SCHOOL   , PlosCats.SCHOOL   , OurCats.SCHOOL    , Intensities.CLOSE_LONG_TERM),     # LOWER_ELEMENTARY_CHILDREN_INCLASS
    LayerDef(PlosCats.SCHOOL   , PlosCats.SCHOOL   , OurCats.SCHOOL    , Intensities.DISTANT_LONG_TERM),   # LOWER_ELEMENTARY_TEACHERS_TO_CHILDREN
    LayerDef(PlosCats.SCHOOL   , PlosCats.SCHOOL   , OurCats.SCHOOL    , Intensities.CLOSE_LONG_TERM),     # HIGHER_ELEMENTARY_CHILDREN_INCLASS
    LayerDef(PlosCats.SCHOOL   , PlosCats.SCHOOL   , OurCats.SCHOOL    , Intensities.DISTANT_LONG_TERM),   # HIGHER_ELEMENTARY_TEACHERS_TO_CHILDREN
    LayerDef(PlosCats.SCHOOL   , PlosCats.SCHOOL   , OurCats.SCHOOL    , Intensities.CLOSE_LONG_TERM),     # HIGHSCHOOL_CHILDREN_INCLASS
    LayerDef(PlosCats.SCHOOL   , PlosCats.SCHOOL   , OurCats.SCHOOL    , Intensities.DISTANT_LONG_TERM),   # HIGHSCHOOL_TEACHERS_TO_CHILDREN
    LayerDef(PlosCats.OTHER    , PlosCats.OTHER    , OurCats.SCHOOL    , Intensities.CLOSE_RANDOM),        # NURSARY_CHILDREN_CORIDORS
    LayerDef(PlosCats.SCHOOL   , PlosCats.SCHOOL   , OurCats.SCHOOL    , Intensities.CLOSE_RANDOM),        # ELEMENTARY_CHILDREN_CORIDORS
    LayerDef(PlosCats.SCHOOL   , PlosCats.SCHOOL   , OurCats.SCHOOL    , Intensities.CLOSE_RANDOM),        # HIGHSCHOOL_CHILDREN_CORIDORS
    LayerDef(PlosCats.WORK     , PlosCats.WORK     , OurCats.WORK      , Intensities.DISTANT_LONG_TERM),   # NURSARY_TEACHERS
    LayerDef(PlosCats.SCHOOL   , PlosCats.SCHOOL   , OurCats.WORK      , Intensities.DISTANT_LONG_TERM),   # ELEMENTARY_TEACHERS
    LayerDef(PlosCats.SCHOOL   , PlosCats.SCHOOL   , OurCats.WORK      , Intensities.DISTANT_LONG_TERM),   # HIGHSCHOOL_TEACHERS
    LayerDef(PlosCats.OTHER    , PlosCats.OTHER    , OurCats.OUTDOOR   , Intensities.CLOSE_OPEN_AIR),      # LEISURE_OUTDOOR
    LayerDef(PlosCats.OTHER    , PlosCats.HOME     , OurCats.VISITS    , Intensities.DININING),            # LEISURE_VISIT
    LayerDef(PlosCats.OTHER    , PlosCats.OTHER    , OurCats.PUBS      , Intensities.DININING),            # LEISURE_PUB
    LayerDef(PlosCats.WORK     , PlosCats.WORK     , OurCats.WORK      , Intensities.DISTANT_LONG_TERM),   # WORK_CONTACTS
    LayerDef(PlosCats.WORK     , PlosCats.OTHER    , OurCats.SERVICES  , Intensities.SERVICE_CLIENT),      # WORK_WORKERS_TO_CLIENTS_DISTANT
    LayerDef(PlosCats.WORK     , PlosCats.OTHER    , OurCats.SERVICES  , Intensities.PHYSICAL_SHORT_TERM), # WORK_WORKERS_TO_CLIENTS_PHYSICAL_SHORT
    LayerDef(PlosCats.WORK     , PlosCats.OTHER    , OurCats.SERVICES  , Intensities.PHYSICAL_LONG_TERM),  # WORK_WORKERS_TO_CLIENTS_PHYSICAL_LONG
    LayerDef(PlosCats.OTHER    , PlosCats.OTHER    , OurCats.TRANSPORT , Intensities.CLOSE_RANDOM),        # PUBLIC_TRANSPORT
    LayerDef(PlosCats.OTHER    , PlosCats.OTHER    , OurCats.SHOPPING  , Intensities.CLOSE_RANDOM),        # SHOPS_CUSTOMERS
    LayerDef(PlosCats.WORK     , PlosCats.OTHER    , OurCats.SHOPPING  , Intensities.SERVICE_CLIENT),      # SHOPS_WORKERS_TO_CLIENTS
    LayerDef(PlosCats.OTHER    , PlosCats.OTHER    , OurCats.PUBS      , Intensities.CLOSE_RANDOM),        # PUBS_CUSTOMERS
    LayerDef(PlosCats.WORK     , PlosCats.OTHER    , OurCats.PUBS      , Intensities.SERVICE_CLIENT),      # PUBS_WORKERS_TO_CLIENTS
    LayerDef(PlosCats.OTHER    , PlosCats.OTHER    , OurCats.OTHER     , Intensities.CLOSE_RANDOM),        # OTHER
    LayerDef(PlosCats.OTHER    , PlosCats.OTHER    , OurCats.OTHER     , Intensities.CLOSE_RANDOM),        # SUPERSPREADER
    LayerDef(PlosCats.OTHER    , PlosCats.OTHER    , OurCats.OTHER     , Intensities.CLOSE_RANDOM),        # FOOTBAL
    LayerDef(PlosCats.OTHER    , PlosCats.OTHER    , OurCats.OTHER     , Intensities.DININING),            # PARTY_1
    LayerDef(PlosCats.OTHER    , PlosCats.OTHER    , OurCats.OTHER     , Intensities.DININING),            # PARTY_2
    LayerDef(PlosCats.OTHER    , PlosCats.OTHER    , OurCats.OTHER     , Intensities.DININING),            # PARTY_3
]
# @formatter:on

WORK_2_CLIENT_LAYERS = [
    Layer.WORK_WORKERS_TO_CLIENTS_DISTANT,  # AGRICULTURE
    Layer.WORK_WORKERS_TO_CLIENTS_DISTANT,  # INDUSTRY
    Layer.WORK_WORKERS_TO_CLIENTS_DISTANT,  # CONSTRUCTION
    Layer.WORK_WORKERS_TO_CLIENTS_DISTANT,  # TRADE_AND_MOTO
    Layer.WORK_WORKERS_TO_CLIENTS_DISTANT,  # TRANSPORTATION
    Layer.WORK_WORKERS_TO_CLIENTS_DISTANT,  # EATING_HOUSING
    Layer.WORK_WORKERS_TO_CLIENTS_DISTANT,  # IT
    Layer.WORK_WORKERS_TO_CLIENTS_DISTANT,  # BANKS
    Layer.WORK_WORKERS_TO_CLIENTS_DISTANT,  # ADMIN_ETC
    Layer.WORK_WORKERS_TO_CLIENTS_DISTANT,  # PUBLIC_SECTOR
    Layer.WORK_WORKERS_TO_CLIENTS_PHYSICAL_SHORT,  # EDUCATION
    Layer.WORK_WORKERS_TO_CLIENTS_PHYSICAL_LONG  # HEALTH
]


class TravelType(EnumZ):
    SCHOOL = auto()
    WORK = auto()
    FAMILY = auto()
    LEISURE = auto()
    SHOPPING = auto()
    SERVICE = auto()
    OTHER = auto()


class TravelRatio(EnumZ):
    OTHER = auto()
    WORK = auto()
    STUDY = auto()


TRAVEL_TYPE_2_RATIO = {
    TravelType.SCHOOL: TravelRatio.STUDY,
    TravelType.WORK: TravelRatio.WORK,
    TravelType.FAMILY: TravelRatio.OTHER,
    TravelType.LEISURE: TravelRatio.OTHER,
    TravelType.SHOPPING: TravelRatio.OTHER,
    TravelType.SERVICE: TravelRatio.OTHER,
    TravelType.OTHER: TravelRatio.OTHER,
}
