from constants import constants
from constants.constants import EnumZ, auto

_DATA = [
    # PUBS_EVENING
    [
        [[0.6, 0.55], [0.56, 0.45], [0.51, 0.42]],
        [[0.52, 0.47], [0.54, 0.38], [0.39, 0.3]],
        [[0.48, 0.53], [0.38, 0.3], [0.36, 0.32]],
        [[0.7, 0.37], [0.49, 0.37], [0.3, 0.28]],
        [[0.55, 0.54]]],
    # SMALL_SHOP
    [
        [[0.07, 0.06], [0.06, 0.08], [0.05, 0.07]],
        [[0.02, 0.02], [0.09, 0.08], [0.13, 0.05]],
        [[0, 0], [0.06, 0.13], [0.05, 0.07]],
        [[0, 0.09], [0.25, 0.09], [0, 0.02]],
        [[0.04, 0.04]]],
    # SUPERMARKET
    [
        [[0.16, 0.19], [0.16, 0.23], [0.15, 0.22]],
        [[0.09, 0.15], [0.12, 0.19], [0.14, 0.29]],
        [[0.29, 0], [0.23, 0.19], [0.14, 0.27]],
        [[0.25, 0.29], [0, 0.22], [0.13, 0.24]],
        [[0.11, 0.13]]],
    # HYPERMARKET
    [
        [[0.43, 0.52], [0.38, 0.53], [0.36, 0.51]],
        [[0.25, 0.34], [0.35, 0.45], [0.34, 0.3]],
        [[0.31, 0.39], [0.36, 0.56], [0.34, 0.42]],
        [[0.3, 0.54], [0.78, 0.53], [0.33, 0.64]],
        [[0.34, 0.4]]]
]


class ProkopActivity(EnumZ):
    WORKING = auto()
    UNEMPLOYED = auto()
    RETIRED = auto()
    HOME = auto()
    STUDENT = auto()


class ProkopTable(EnumZ):
    PUBS_EVENING = auto()
    SMALL_SHOP = auto()
    SUPERMARKET = auto()
    HYPERMARKET = auto()


def age2index(age):
    if age < 18:
        return None
    if age < 34:
        return 0
    if age < 54:
        return 1
    return 2


def eact2index(activity: constants.EconoActivity):
    return {
        constants.EconoActivity.WORKING: ProkopActivity.WORKING,
        constants.EconoActivity.AT_HOME: ProkopActivity.HOME,
        constants.EconoActivity.STUDENT: ProkopActivity.STUDENT,
        constants.EconoActivity.RETIRED: ProkopActivity.RETIRED}[activity]


def prob_visit(tbl: ProkopTable, person):
    age_index = age2index(person.age)
    if age_index is None:
        return 0
    sex_index = 0 if person.sex is constants.Gender.MEN else 1
    return _DATA[tbl][eact2index(person.activity)][age_index][sex_index]
