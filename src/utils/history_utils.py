import numpy as np


class BaseSeries():

    """
    Base class for all series-classes.
    Series stores thinks that are produced in time manner, i.e. each iteration a new item is produced.
    Series should increase their length automatically.
    """

    def __init__(self):
        self.values = None

    def __getitem__(self, idx):
        return self.values[idx]

    def __setitem__(self, idx, data):
        try:
            self.values[idx] = data
        except IndexError as e:
            if idx == len(self.values):
                self.bloat(100)
                self.values[idx] = data
            else:
                print(len(self.values))
                print(idx)
                raise e

    def save(self, filename):
        np.save(self.values, filename)

    def __len__(self):
        return len(self.values)

    def len(self):
        return len(self.values)

    def asarray(self):
        return self.values

    def bloat(self, len):
        pass


class TimeSeries(BaseSeries):

    """
    Series of values of `dtype`.
    """

    def __init__(self, len, dtype=float):
        super().__init__()
        self.type = dtype
        self.values = np.zeros(len, dtype)

    def bloat(self, len):
        self.values = np.pad(
            self.values,
            [(0, len)],
            mode='constant', constant_values=0)

    def finalize(self, tidx):
        """ throw away ending zeros """
        self.values = self.values[:tidx+1]

    def get_values(self):
        return self.values


class TransitionHistory(BaseSeries):

    """
    Table of values of defined width. Only length increases in time. 
    """

    def __init__(self, len, dtype=int, width=3):
        super().__init__()
        self.values = np.zeros((len, width), dtype=dtype)
        self.width = width
        self.dtype = dtype

    def bloat(self, len):
        new_space = np.zeros((len, self.width), dtype=self.dtype)
        self.values = np.vstack([self.values, new_space])

    def finalize(self, tidx):
        """ throw away ending zeros """
        self.values = self.values[:tidx+1, :]


class ShortListSeries():
    """
    List of a given length.
    If full and a new value is appended, the oldest value is dropped. 
    """

    def __init__(self, length):
        self.values = []
        self.length = length

    def append(self, member):
        self.values.append(member)
        if len(self.values) > self.length:
            self.values.pop(0)

    def __getitem__(self, idx):
        return self.values[idx]

    def __len__(self):
        return len(self.values)
