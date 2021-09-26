import os
import shutil

import urllib3

from config.base_config import BaseConfig
from ruian.downloader import DownloadAddresses
from utils.csv_utils import CsvReader, CsvWriter, UTF8, WIN


def zujs_loader(cfg: BaseConfig):
    prepare(cfg)
    for (zuj, orp, cnt, x, y, name) in CsvReader(cfg.zujs(), None, ['zuj', 'orp', 'cnt', 'x', 'y', 'name'], UTF8):
        yield (zuj, orp, int(cnt), float(x), float(y), name)


def download_cisob(filename):
    with urllib3.PoolManager() as http:
        with http.request('GET',
                          'http://apl.czso.cz/iSMS/cisexp.jsp',
                          {
                              'kodcis': '65',
                              'typdat': '1',
                              'cisvaz': '43_1182',
                              'cisjaz': '203',
                              'format': '2',
                              'separator': ','
                          },
                          preload_content=False) as r, open(filename, 'wb') as out_file:
            if r.status != 200:
                raise Exception(f"Unable to download cisob status={r.status}, {r.reason}", r)
            shutil.copyfileobj(r, out_file)


def read_cisob(filename):
    orp_list = {}
    for (orp, orp_name, zuj, zuj_name) in CsvReader(filename, [4, 5, 8, 9],
                                                    ['KODJAZ', 'TYPVAZ', 'AKRCIS1', 'KODCIS1', 'CHODNOTA1', 'TEXT1',
                                                     'AKRCIS2', 'KODCIS2', 'CHODNOTA2', 'TEXT2'], WIN):
        orp_list[zuj] = orp
    return orp_list


def prepare(cfg: BaseConfig):
    # nacti vsechny RUIAN
    # pocitej prumernou polohu
    zujs = dict()
    print("downloading adresses")
    dwnl = DownloadAddresses()
    dwnl.addresses(cfg.ruians())

    if os.path.exists(cfg.zujs()):
        return

    print("ctu adresy")
    for filename in os.listdir(cfg.ruians_read()):
        if filename.endswith('.csv'):
            print(f"ctu {filename}")
            for (zuj, nazev, y, x) in CsvReader(cfg.ruians_read() + filename, [1, 2, 16, 17], delimiter=';'):
                if x and y:
                    if zuj not in zujs:
                        zujs[zuj] = [nazev, 1, float(x), float(y)]
                    else:
                        zujs[zuj][1] += 1
                        zujs[zuj][2] += float(x)
                        zujs[zuj][3] += float(y)
    download_cisob(cfg.cisob())
    orps = read_cisob(cfg.cisob())

    print("vypis")
    with CsvWriter(cfg.zujs()) as f:
        f.writerow(['zuj', 'orp', 'cnt', 'x', 'y', 'name'])
        for zuj, [nazev, cnt, x, y] in zujs.items():
            f.writerow([zuj, orps[zuj], cnt, x / cnt, y / cnt, nazev])
