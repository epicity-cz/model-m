from __future__ import annotations

import numpy as np


class Histogram:
    nums: np.ndarray
    limits: np.ndarray
    lengths: np.ndarray
    total: int

    def __init__(self, limits: np.ndarray, max_age: int, total_pop: int, source: np.ndarray):
        assert source.shape[0] == limits.shape[0]
        self.limits = limits
        self.total = total_pop
        self.nums = -(source * total_pop)
        all_limits = np.append(limits, max_age)
        self.lengths = all_limits[1:] - all_limits[:-1]

    def age_index(self, age):
        return np.searchsorted(self.limits, age, side='right') - 1

    def add(self, age, sex):
        self.nums[self.age_index(age), sex] += 1

    def remove(self, age, sex):
        self.nums[self.age_index(age), sex] -= 1

    def add_delta(self, delta: np.ndarray):
        self.nums += delta

    def distances(self, deltas: np.ndarray):
        distances = (deltas + self.nums[None, :, :])
        distances = np.abs(distances.cumsum(axis=1))

        return np.sum(distances, axis=(1, 2)) / self.total
