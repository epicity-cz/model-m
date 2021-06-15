import csv
import itertools

UTF8 = 'utf-8'
WIN = 'windows-1250'


def flatten(param):
    return itertools.chain(*param)


class CsvWriter():
    def __init__(self, file_name):
        self.file_name = file_name

    def __enter__(self):
        self.file = open(self.file_name, 'w', newline='')
        return csv.writer(self.file)

    def __exit__(self, exc_type, exc_value, exc_tb):
        self.file.close()


def CsvReader(filename, columns=None, header=None, encoding=WIN, delimiter=','):
    with open(filename, encoding=encoding) as file:
        rdr = csv.reader(file, delimiter=delimiter)
        real_header = next(rdr)
        if header and (real_header != header):
            raise Exception(f"CSV {filename} format error")
        for row in csv.reader(file, delimiter=delimiter):
            if len(row):
                yield (row[i] for i in columns) if columns else row
