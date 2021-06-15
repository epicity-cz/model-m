import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np

from lang.mytypes import List, Set
from program.loaders import plos_loader

np.seterr(divide='raise')


class PlosMatrixCats:
    CATS = 0

    @classmethod
    def size(cls):
        return cls.CATS + 1

    @classmethod
    def value2cat(cls, value) -> int:
        pass

    @classmethod
    def label(cls, cat: int) -> str:
        pass

    @classmethod
    def labels(cls) -> List[str]:
        return [cls.label(cat) for cat in range(cls.CATS + 1)]


class PlosMatrixNoCats(PlosMatrixCats):
    def __init__(self, cats):
        self.__class__.CATS = cats

    @classmethod
    def value2cat(cls, value) -> int:
        return value

    @classmethod
    def label(cls, cat: int) -> str:
        return f"{cat}"


class PlosMatrixAgeCats(PlosMatrixCats):
    CATS = 16

    @classmethod
    def value2cat(cls, value) -> int:
        return min(value // 5, cls.CATS)

    @classmethod
    def label(cls, cat: int) -> str:
        return str((cat + 1) * 5)  # f"{cat * 5}-{(cat + 1) * 5}"


class PlosMatrix:
    catagories: PlosMatrixCats
    nums: np.ndarray
    sums: np.ndarray
    persons: Set
    size: int

    def __init__(self, categories: PlosMatrixCats = PlosMatrixAgeCats):
        self.categories = categories
        self.size = categories.size()
        self.nums = np.zeros([self.size], dtype=int)
        self.sums = np.zeros([self.size, self.size])
        self.persons = set()

    def copy(self):
        cpy = PlosMatrix(self.categories)
        cpy.nums = self.nums.copy()
        cpy.sums = self.sums.copy()
        cpy.persons = self.persons.copy()
        return cpy

    def population(self):
        return np.sum(self.nums)

    def add_person(self, person, age):
        if person not in self.persons:
            self.nums[self.categories.value2cat(age)] += 1
            self.persons.add(person)

    def add_contact(self, age1, age2, prob=1):
        self.sums[self.categories.value2cat(age1), self.categories.value2cat(age2)] += prob

    def table(self, size=None):
        with np.errstate(divide='ignore', invalid='ignore'):
            ret = self.sums / self.nums[:, None]
        ret[self.nums == 0, :] = 0
        return ret[:size, :size]

    def __getitem__(self, key):
        n = self.nums[key[0]]
        if not n:
            return 0.0
        return self.sums[key] / n

    def total(self, row):
        n = self.nums[row]
        if not n:
            return 0.0
        return np.sum(self.sums[row,]) / n

    def totals(self):
        return np.sum(self.table(), axis=1)

    def average(self):
        return np.sum(self.sums) / np.sum(self.nums)

    def reduced_size(self, small=True):
        return self.size - 1 if small else None

    def weight(self, other, size=None):
        return self.nums[:size, None] * other.nums[None, :size]

    def pmdist(self, other, small=True):
        assert self.sums.shape == other.sums.shape
        size = self.reduced_size(small)
        w = self.weight(other, size)
        return np.sum(w * np.abs(self.table(size) - other.table(size))) / np.sum(w)

    def pmdist2(self, other, small=True):
        assert self.sums.shape == other.sums.shape
        size = self.reduced_size(small)
        w = self.weight(other, size)
        return np.sum(w * np.power(self.table(size) - other.table(size), 2)) / np.sum(w)

    def pmtdist(self, other, small=True):
        assert self.sums.shape == other.sums.shape
        size = self.reduced_size(small)
        return np.sum(np.abs(self.sums - other.sums)[:size, :size])

    def read1(self, filename):
        self.nums[:-1] = 1
        self.nums[-1] = 0
        i = 0
        for row in plos_loader(filename, self.size - 1):
            self.sums[i, :(self.size - 1)] = list(row)
            i += 1
        self.sums[:, self.size - 1] = 0
        self.sums[self.size - 1,] = 0

    def set_nums(self, nums):
        assert np.shape(nums) == (self.size,)
        self.nums[:-1] = nums[:-1]
        self.nums[-1] = 0
        self.sums *= self.nums[:, None]

    def read(self, filename, nums):
        assert np.shape(nums) == (self.size,)
        self.nums[:-1] = nums[:-1]
        self.nums[-1] = 0
        i = 0
        for row in plos_loader(filename, self.size - 1):
            self.sums[i, :(self.size - 1)] = list(row)
            i += 1
        self.sums *= self.nums[:, None]

    def plot(self, filename=None):
        plt.figure()
        c = plt.pcolor(self.table().T, cmap='Blues', norm=colors.PowerNorm(gamma=0.5))
        plt.xticks(ticks=np.arange(self.size) + .5, labels=self.categories.labels())
        plt.yticks(ticks=np.arange(self.size) + .5, labels=self.categories.labels())
        plt.tick_params(axis='both', length=0)
        plt.colorbar(c, ticks=[0, .5, 1, 1.5, 2])
        if filename:
            plt.savefig(filename)

    def plot_bar(self, other, filename=None):
        fig = plt.figure()

        ax = fig.add_subplot(111)  # Create matplotlib axes
        ax2 = ax.twinx()  # Create another axes that shares the same x-axis as ax.

        ind = np.arange(self.size)
        width = 0.35
        ax.bar(ind, self.totals(), width=width)
        ax.bar(ind + width, other.totals(), width=width)
        ax2.bar(ind + width / 2, self.nums, width=width)
        plt.xticks(ticks=ind + width / 2, labels=self.categories.labels())
        if filename:
            plt.savefig(filename)
