from __future__ import annotations

import configparser

from constants.constants import PlosCats, OurCats, CommutingTime


class BaseConfig:

    def __init__(self):
        self.FILE_NODES = 'p.csv'
        self.FILE_EXPORT = 'e.csv'
        self.FILE_EDGES = 'edges.csv'
        self.FILE_LAYERS = 'etypes.csv'
        self.FILE_NUMS = 'nums.csv'
        self.FILE_OBJECTS = 'objects.csv'
        self.NAME = 'unknown'
        self.TOWNS = []
        self.ORPS = []

        self.LOAD_PERSONS = True

        self.CLASS_SIZE = 20
        self.LAST_INSIDE = CommutingTime.HALF
        self.ADDITIONAL_TEACHERS = 2
        self.MUTUAL_LIMIT = 25
        self.MAX_ALL_EDGES = 20
        self.SCHOOL_TC_NUM_RC = 8

        self.FIRST_IN_FRIENDS = 10
        self.OPP_SEX_ACCEPT = 0.2
        self.TEN_YRS_ACCEPT = 0.3
        self.FRIEND_DIST_PREF = 0.65
        self.FRIENDS_KAPPA = 4
        self.FRIENDS_LAMBDA = 6
        self.FRIENDS_MIN_AGE_PUB = 18
        self.FRIENDS_PUB_DIST_PREF = 0.5
        self.FRIENDS_SAMPLE = 5000
        self.FRIENDS_ACCEPT_EDGE = 0.99
        self.FAMILY_AGE_SAME = 10
        self.FAMILY_FACTOR_LIMIT = 60
        self.FAMILY_SENIOR_AGE = 50

        self.FIRST_NURSARY = 3
        self.LAST_NURSARY = 5
        self.FIRST_ELEM = 6
        self.LAST_FIRST_ELEM = 10
        self.LAST_ELEM = 14
        self.FIRST_HIGHSCHOOL = 15
        self.LAST_HIGHSCHOOL = 18
        self.LAST_POT_STUDENT = 24
        self.FIRST_RETIRED = 55

        self.PROB_OF_UNIV = [0.7, 0.6]

        self.WORK_RATIO_ADJUST = 0.91

        self.OTHER_AGE_ADJUST = 8
        self.OTHER_SAME_TOWN_ADJUST = 0.1
        self.OTHER_PROBABILITY = 0.21
        self.MAX_AGE = 100
        self.EATING_RATE = 2

        # @see constants.Intensities
        self.INTENSITIES = [
            0.255905135951299,
            0.209577963668343,
            0.209577963668343,
            0.156907515722802,
            0.108580705683382,
            0.031870115278543,
            0.017146327621734,
            0.010434272405554
        ]

        self.NUMBER_OF_CHILDREN = [
            0.012099082595569,
            0.240393454951354,
            0.568886930449098,
            0.172548607723645,
            0.006050837543054,
            0.00002
        ]

        self.WORK_MATRIX = [
            [3.07509956747975, 0, 0, 1.2300398269919, 0, 0.615019913495949, 0, 0, 0, 0, 0, 0],
            [0, 1.2300398269919, 1.2300398269919, 0.615019913495949, 0.615019913495949, 0, 0, 0, 0, 0, 0, 0],
            [0, 4.92314629152027, 1.84505974048785, 0, 1.2300398269919, 0, 0, 0, 0, 0, 0, 0],
            [0.348298013429761, 1.64771752507156, 0, 1.2300398269919, 1.2300398269919, 0.615019913495949, 0, 0, 0, 0, 0,
             0],
            [0, 3.67007900587988, 1.8339261432124, 2.73975561754314, 1.2300398269919, 0, 0, 0, 0, 0, 0, 0],
            [0.61150952586184, 0, 0, 2.15959047250404, 0, 0.615019913495949, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0.615019913495949, 0.615019913495949, 0.615019913495949, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0.449678511089706, 1.2300398269919, 0.615019913495949, 0.615019913495949, 0, 0],
            [0, 0, 0, 0, 0, 0, 0.149791145824319, 0.204867555988135, 1.2300398269919, 1.2300398269919, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0.201854797811839, 1.21195100600672, 1.2300398269919, 0.615019913495949,
             0.615019913495949],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0.673550359143441, 1.2300398269919, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0.613789053095357, 0, 1.2300398269919]]

        self.CLIENT_MULT = 10

        self.WORK_2_CLIENT = [
            1.2300398269919,
            0.615019913495949,
            0.615019913495949,
            1.2300398269919,
            0.615019913495949,
            4.92015930796759,
            0.615019913495949,
            1.84505974048785,
            3.6901194809757,
            3.07509956747975,
            4.30513939447165,
            4.30513939447165]

        self.WORK_EXPORT_RATE = 0.4 * 6

        self.CSU_MAX_HH_SIZE = 6

        self.TRAVEL_INFO = 'TravelInfoDelaunay'

        # ostatni prace studium   michal.simecek@cdv.cz
        self.TRAVEL_RATIOS_ALL = [
            [34.9, 11.3, 1.1],  # auto
            [32.3, 4.5, 2.9],  # pesky
            [8.1, 2.5, 2.0],  # mhd
            [0.3, 0.1, 0.0],  # N/A
        ]

    def town_dump(self, name):
        return f'data/towns/{name}.dmp'

    def buildings(self, zuj):
        return f'data/towns/obec-{zuj}.csv'

    def get_filename(self, ini):
        return f'config/{ini}.ini'

    def households(self):
        return f'config/gen_graph/households.csv'

    def persons(self):
        return f'config/gen_graph/{self.NAME}_osoby.csv'

    def poi(self):
        return f'config/gen_graph/{self.NAME}_poi.csv'

    def schools(self):
        return f'config/gen_graph/schools_cz_loc.csv'

    def exports(self):
        return f'config/gen_graph/{self.NAME}_exportplaces.csv'

    def transport(self):
        return f'config/gen_graph/{self.NAME}_transport.csv'

    def out_gstat(self):
        return f'data/output/gstat.csv'

    def out_cat_graph(self, cat: OurCats):
        return f'data/output/graphs/{cat.NAME.lower()}.csv'

    def out_friends_nodes(self):
        return f'data/output/friendnodes.csv'

    def out_friends_edges(self):
        return f'data/output/friendedges.csv'

    def out_pub_visits(self):
        return f'data/output/pubvisits.csv'

    def out_shop_visits(self):
        return f'data/output/shopvisits.csv'

    def busy_sections(self):
        return f'data/output/busysections.txt'

    def log_report(self):
        return f'data/output/buildreport.txt'

    def log_school(self):
        return f'data/output/schoolbuild.txt'

    def plos_data(self, category: PlosCats):
        return f'config/gen_graph/plos17/{category.label()}.csv'

    def pic_bar(self, category: PlosCats):
        return f'data/output/pics/{category.label()}.eps'

    def pic(self, category: PlosCats, postfix=''):
        return f'data/output/pics/{category.label()}_mat{postfix}.eps'

    def pic_ref(self, category: PlosCats):
        return self.pic(category, "_ref")

    def ruians(self):
        return 'data/gen_graph/ruian/'

    def ruians_read(self):
        return self.ruians() + 'CSV/'

    def zujs(self):
        return 'data/gen_graph/zujs.csv'

    def cisob(self):
        return 'data/gen_graph/cisob.csv'

    def contains(self, zuj, orp):
        return (zuj in self.TOWNS) or (orp in self.ORPS)

    @property
    def TOWNS_OB(self):
        return ''

    @TOWNS_OB.setter
    def TOWNS_OB(self, value):
        self.TOWNS = value.split(',')

    @property
    def TOWNS_ORP(self):
        return ''

    @TOWNS_ORP.setter
    def TOWNS_ORP(self, value):
        self.ORPS = value.split(',')

    def cast(self, value, tp):
        if tp is bool:
            return value.upper() in ['1', 'T', 'TRUE', 'Y', 'YES']
        return tp(value)

    def set_key(self, key: str, value: str):
        key = key.upper()
        if not hasattr(self, key):
            raise ValueError(f"Unknown config key '{key}")
        nt = type(getattr(self, key))
        try:
            setattr(self, key, self.cast(value, nt))
        except:
            raise ValueError(f"Cannot interpret value {value} as {nt}")

    def read_ini(self, inifile):
        cf = configparser.ConfigParser()
        cf.read_file(inifile)
        graph_mapping = {
            'NODES': 'FILE_NODES',
            'EXTERNALS': 'FILE_EXPORT',
            'EDGES': 'FILE_EDGES',
            'LAYERS': 'FILE_LAYERS',
            'NUMS': 'FILE_NUMS',
            'OBJECTS': 'FILE_OBJECTS',
        }
        for (key, value) in cf.items('GRAPH'):
            if key in graph_mapping:
                nk = graph_mapping[key]
                self.set_key(nk, value)
        for (key, value) in cf.items('GRAPH_GEN'):
            self.set_key(key, value)

    @staticmethod
    def create_from_inifile(inifile):
        cfg = BaseConfig()
        with open(cfg.get_filename(inifile), 'r') as f:
            cfg.read_ini(f)
        return cfg
