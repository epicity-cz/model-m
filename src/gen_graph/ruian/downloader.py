# code fragments from RUIAN downloader by Jachym Cepicky (jachym.cepicky opengeolabs.cz)

import datetime
import os
import shutil
import tempfile
from calendar import monthrange
from zipfile import ZipFile

import urllib3
from osgeo import ogr

from utils.csv_utils import CsvWriter


class DownloadZip:
    URL = ''

    def get_date(self):
        now = datetime.datetime.now()
        month = now.month - 1
        year = now.year
        if month == 0:
            month = 12
            year = now.year - 1
        days = monthrange(year=year, month=month)
        return "{}{:02d}{}".format(year, month, days[1])

    def download(self, tmp_dir_name, out_dir_name, **params):
        out_temp_name = tempfile.mktemp(dir=tmp_dir_name, suffix=".zip")
        params['date'] = self.get_date()
        url = self.URL.format(**params)
        with urllib3.PoolManager() as http:
            with http.request('GET', url, preload_content=False) as r, open(out_temp_name, 'wb') as out_file:
                shutil.copyfileobj(r, out_file)

        with ZipFile(out_temp_name, "r") as myzip:
            data_file = myzip.namelist()[0]
            myzip.extractall(out_dir_name)

        return os.path.join(out_dir_name, data_file)

    def temp_create(self):
        return tempfile.mkdtemp(prefix="ruian")

    def temp_clean(self, tmp_dir_name):
        shutil.rmtree(tmp_dir_name)


class DownloadObjects(DownloadZip):
    URL = 'https://vdp.cuzk.cz/vymenny_format/soucasna/{date}_OB_{kod}_UKSH.xml.zip'

    def process(self, xml_file, csv_file):
        source = ogr.Open(xml_file)
        assert source

        layer = source.GetLayerByName("StavebniObjekty")
        assert layer

        typy_objektu = [6, 7]
        kody = ["ZpusobVyuzitiKod = {}".format(obj) for obj in typy_objektu]
        layer.SetAttributeFilter(" OR ".join(kody))

        with CsvWriter(csv_file) as f:
            f.writerow(["Kod", "TypStavebnihoObjektuKod", "ZpusobVyuzitiKod",
                        "PocetBytu", "PocetPodlazi", "x", "y"
                        ])
            feature = layer.GetNextFeature()
            while feature:
                assert feature
                geom = feature.GetGeometryRef().GetPoint()
                f.writerow([
                    feature.GetField("Kod"),
                    feature.GetField("TypStavebnihoObjektuKod"),
                    feature.GetField("ZpusobVyuzitiKod"),
                    feature.GetField("PocetBytu"),
                    feature.GetField("PocetPodlazi"),
                    -geom[1], -geom[0]
                ])
                feature = layer.GetNextFeature()

    def apartments(self, zuj, filename):
        if os.path.exists(filename):
            return
        out_dir_name = self.temp_create()
        xml_file_name = self.download(out_dir_name, out_dir_name, kod=zuj)
        self.process(xml_file_name, filename)
        self.temp_clean(out_dir_name)


class DownloadAddresses(DownloadZip):
    URL = 'https://vdp.cuzk.cz/vymenny_format/csv/{date}_OB_ADR_csv.zip'

    def addresses(self, dir_path):
        if os.path.exists(dir_path):
            return
        os.makedirs(dir_path)
        tmp_dir_name = self.temp_create()
        self.download(tmp_dir_name, dir_path)
        self.temp_clean(tmp_dir_name)
