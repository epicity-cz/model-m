import os

from config.base_config import BaseConfig
from utils.csv_utils import CsvReader, CsvWriter, UTF8

# nacti vsechny skoly
SKOLY_IN = 'data/schools/Skoly_a_skolska_zarizení.csv'
SKOLY_FILTERED = 'data/schools/schools_hodoninsko.csv'

school_types = {
    'A': 'nursary',
    'B': 'elementary',
    'C': 'highschools',
    'D': 'highschools',
}


def main(cfg: BaseConfig):
    ruians = set()
    schools = []
    adresses = dict()

    print("ctu skoly")
    for (nazev, ruian, typ, kapacita) in CsvReader(SKOLY_IN, [7, 15, 23, 32], encoding=UTF8):
        ruians.add(ruian)
        tp1 = typ[0]
        if tp1 in school_types:
            schools.append((nazev, ruian, school_types[tp1], kapacita))

    print("ctu adresy")
    base = cfg.ruians_read()
    for filename in os.listdir(base):
        if filename.endswith('.csv'):
            print(f"ctu {filename}")
            for (ruian, zuj, y, x) in CsvReader(base + filename, [0, 1, 16, 17], delimiter=';'):
                if ruian in ruians:
                    adresses[ruian] = (zuj, x, y)

    print("vystup")
    with CsvWriter(cfg.schools()) as f:
        f.writerow(['zuj', 'ruian', 'typ', 'kapacita', 'x', 'y', 'nazev'])
        for (nazev, ruian, typ, kapacita) in schools:
            if ruian in adresses:
                (zuj, x, y) = adresses[ruian]
                f.writerow([zuj, ruian, typ, kapacita, x, y, nazev])
            else:
                print(f"ignored (no ruian) {ruian} - {nazev}")


def check_schools(region):
    zujs = region.towns()
    with CsvWriter(SKOLY_FILTERED) as f:
        f.writerow(['zuj', 'ruian', 'typ', 'kapacita', 'x', 'y', 'nazev'])
        for (zuj, ruian, typ, kapacita, x, y, nazev) in CsvReader(region.schools(), encoding=UTF8):
            if zuj in zujs:
                f.writerow([zuj, ruian, typ, kapacita, x, y, nazev])


if __name__ == '__main__':
    hodoninsko = BaseConfig.create_from_inifile('hodoninsko')
    main(hodoninsko)
    check_schools(hodoninsko)
