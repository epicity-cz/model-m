import xml.etree.ElementTree as ET

import numpy as np
import urllib3

from utils.file_cache import file_cache

TABLES = {
    '112': 'v157',
    '113': 'v133',
    '114': 'v101',
    '115': 'v97',
    '116': 'v185',
    '122': 'v146',
}


# condition ( disctionary, value )
# value je

# None             nema tento dictionary
# ciselnik=value   ma subnode s hodnotou z ciselniiku

# ciselnik=*        ma subnode s jakoukoliv hodnotou z ciselniku        cond_has_any
# ciselnik!*        nema subnode s jakoukoliv hodnotou z ciselniku      cond_has_none
# ciselnik!value   nema subnode s touto hodnotou z ciselniku            cond_not_equal

# (function, parameters)
# (ciselnik, value)
# (ciselnik, '*')
# (ciselnik, value | *, '=' | '!')

#
def cond_has_any(dict, nodes, param):
    rev = {value: key for (key, value) in dict.items()}
    for node in nodes:
        if rev[node.text()].startswith(param):
            return True
    return False


def cond_has_none(dict, nodes, param=None):
    rev = {value: key for (key, value) in dict.items()}
    if param is None:
        # nodes musi byt prazdne
        return not nodes
    for node in nodes:
        if rev[node.text].startswith(param):
            return False
    return True


def cond_not_equal(dict, nodes, param):
    raise Exception('Not implemented yet')
    pass


class XmlReader:
    def __init__(self):
        self.ns = ''

    def set_data(self, data):
        self.root = ET.fromstring(data)
        self.fill_dict()

    def fill_dict(self):
        ukazatele = {}
        for item in self.root.findall(f'./{self.ns}metaSlovnik/{self.ns}ukazatel/{self.ns}uka'):
            ukazatele[item.find(f'{self.ns}kod').text] = item.get('ID')

        vecne = {}
        for item in self.root.findall(f'./{self.ns}metaSlovnik/{self.ns}vecneUpresneni/{self.ns}element'):
            vecne[item.find(f'{self.ns}ciselnik').text + '=' + item.find(f'{self.ns}kod').text] = item.get('ID')

        self.dicts = {'uka': ukazatele, 'vec': vecne}

    def value_int(self, constrains):
        return int(self.value(constrains))

    def value(self, constrains):
        xpath = ''
        has_except = False
        for (dict, val, *rest) in constrains:
            if isinstance(val, str):
                cond = self.dicts[dict][val]
                xpath += f"[{self.ns}{dict}='{cond}']"
            else:
                has_except = True
        xpath = f'./{self.ns}data/{self.ns}udaj{xpath}'
        if not has_except:
            xpath += f'/{self.ns}hod'
            return self.root.find(xpath).text

        for node in self.root.findall(xpath):
            ok = True
            for (dict, check_function, *params) in constrains:
                if not isinstance(check_function, str):
                    if not check_function(self.dicts[dict], node.findall(f'{self.ns}{dict}'), *params):
                        ok = False
                        continue
            if ok:
                return node.find(f'{self.ns}hod').text
        raise Exception('node not found')

    def array(self, all, rows, cols, read_fce):
        return np.array([[read_fce(all + row + col) for col in cols] for row in rows])


class Reader(XmlReader):
    def __init__(self, zuj, table, cache_dir):
        super().__init__()
        self.zuj = zuj
        self.table = str(table)
        self.cache_dir = cache_dir
        self.ns = '{http://vdb.czso.cz/xml/export}'
        self.set_data(file_cache(f'{self.cache_dir}/otob_{self.zuj}_{self.table}.xml')(self.get_data)())

    def get_data(self):
        with urllib3.PoolManager() as http:
            r = http.request('GET',
                             'https://vdb.czso.cz/vdbvo2/faces/xmlexp',
                             {'page': 'vystup-objekt',
                              'z': 'T',
                              'f': 'TABULKA',
                              'pvo': f'OTOB{self.table}',
                              'u': f'{TABLES[self.table]}__VUZEMI__43__{self.zuj}',
                              'str': TABLES[self.table],
                              'kodjaz': '203',
                              'nasexp': 'ss',
                              'expJenKody': 'N',
                              'expdefinice': 'N',
                              'datovytyp': 'N',
                              'expatrib': 'N',
                              'expcasdb': 'N'
                              })
        return r.data
