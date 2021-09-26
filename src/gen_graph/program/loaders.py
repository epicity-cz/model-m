from constants import constants
from utils.csv_utils import CsvReader, UTF8, WIN


def households_loader(filename):
    for (age, sex, key) in CsvReader(filename, [0, 1, 2]):
        yield (int(age), constants.Gender(int(sex)), int(key))


def osoby_loader(filename):
    for (zuj, sex, age, key) in CsvReader(filename, [0, 6, 7, 10]):
        if zuj:
            if sex == "615":
                sex = constants.Gender.MEN
            elif sex == "616":
                sex = constants.Gender.WOMEN
            else:
                raise Exception(f"Unknown sex code '{sex}'")
            age = int(age)
            key = int(key)
            yield (zuj, sex, age, key)


def apartments_loader(filename):
    for (tp, cnt, x, y) in CsvReader(filename, [2, 3, 5, 6]):
        if not cnt == '':
            fh = "7" == tp
            for _ in range(int(cnt)):
                yield (fh, float(x), float(y))


def schools_loader(filename):
    school_type = {
        'nursary': constants.SchoolType.NURSARY,
        'elementary': constants.SchoolType.ELEMENTARY,
        'highschools': constants.SchoolType.HIGH,
    }

    for (zuj, tp, cap, x, y, name) in \
            CsvReader(filename,
                      [0, 2, 3, 4, 5, 6],
                      ["zuj", "ruian", "typ", "kapacita", "x", "y", "nazev"],
                      UTF8
                      ):
        ww = 1
        capacity = float(cap)

        type = school_type.get(tp)
        if type is not None and x and y:
            yield (zuj, name, type, capacity, float(x), float(y), ww)


def exports_loader(filename):
    return CsvReader(filename,
                     [0, 1],
                     ["code", "name"],
                     WIN
                     )


def transport_loader(filename):
    for (frm, to, tm) in \
            CsvReader(filename,
                      [0, 1, 2],
                      ["from", "to", "time"],
                      UTF8
                      ):
        yield (frm, to, float(tm))


def plos_loader(filename, size):
    return CsvReader(filename,
                     range(size),
                     [f"X{i}" for i in range(1, size + 1)],
                     UTF8
                     )
