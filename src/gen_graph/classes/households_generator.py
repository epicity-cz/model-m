from __future__ import annotations

from collections import defaultdict

import numpy as np

from classes.town_params import TownParams
from constants import constants
from lang.mytypes import DefaultDict, Tuple, List
from program import loaders
from utils import math_prob
from utils.histogram import Histogram
from utils.log import StdLog

MAX_MOVE = 10


class HouseholdGenerator:
    max_hh_size: int
    max_age: int
    households_by_size: DefaultDict[int, List[List[Tuple[int, int]]]]
    max_move: int
    moves_all: np.ndarray

    def __init__(self, filename, max_hh_size: int, max_age, max_move=MAX_MOVE, loader=loaders.households_loader):
        self.max_hh_size = max_hh_size
        self.max_age = max_age
        self.load_households(loader(filename))
        self.max_move = max_move
        self.moves_all = np.arange(-max_move, max_move + 1)

    def generate_persons_town(self, town: TownParams):
        StdLog.log(f"generate_persons_town {town.name}")
        age_limits = town.age_limits
        age_limits_full = np.concatenate([[np.NINF], age_limits, [self.max_age]])
        total_pop = town.inhabitants
        h_current = Histogram(age_limits, self.max_age, total_pop, town.age_sex)

        population, hh_indexes = self.generate_population(h_current, total_pop, town.apartments_counts)
        self.change_sex(h_current, total_pop, population)
        self.fit_population(population, hh_indexes, total_pop, h_current, age_limits_full, age_limits)
        return self.population_by_households(population, hh_indexes)

    def load_households(self, loader):
        last_key = 0
        households = []
        last_hh = []
        for (age, sex, key) in loader:
            if age > self.max_age:
                age = self.max_age
            p = (age, sex)
            if key != last_key:
                if key < last_key:
                    raise Exception("keys have to be ordered ascendantly")
                last_key = key
                last_hh = []
                households.append(last_hh)
            last_hh.append(p)
        self.households_by_size = defaultdict(list)
        for hh in households:
            size = len(hh)
            if size > self.max_hh_size:
                size = self.max_hh_size
            self.households_by_size[size].append(hh)

    def generate_population(self, h_current: Histogram, total_pop: int, apartments_counts: np.ndarray):
        population_by_appartments = sum((index + 1) * cnt for (index, cnt) in enumerate(apartments_counts))
        assert total_pop >= population_by_appartments
        additionals = total_pop - population_by_appartments
        remaining = total_pop
        population = np.zeros([total_pop, 2], dtype=int)  # age,sex
        hh_indexes = []

        def add_household(household, expected_size):
            nonlocal remaining, base, population, h_current, additionals
            hh_indexes.insert(0, remaining)
            cur_size = len(household)
            this_additionals = cur_size - expected_size
            if this_additionals > 0:
                if this_additionals > additionals:
                    this_additionals = additionals
                additionals -= this_additionals
                hh_iter = household[:(expected_size + this_additionals)]
            else:
                hh_iter = household
            for (age, sex) in hh_iter:
                h_current.add(age, sex)
                remaining -= 1
                if remaining < 0:
                    assert "PROBLEM"
                population[remaining, :] = (age, sex)

        # fill all households by number
        for i in range(self.max_hh_size, 0, -1):
            base = self.households_by_size[i]
            for idx in np.random.randint(len(base), size=apartments_counts[i - 1]):
                add_household(base[idx], i)

        # fill remaining population
        ap_counts = list(apartments_counts)
        while remaining > 0:
            idx = math_prob.draw(ap_counts)
            hh_size = idx + 1
            if (hh_size > remaining):
                hh_size = remaining
            hh = math_prob.choice(self.households_by_size[hh_size])
            add_household(hh, hh_size)

        hh_indexes.insert(0, 0)
        return population, hh_indexes

    def change_sex(self, h_current, total_pop, population):
        # change sex
        changes_needed = np.round(np.sum(h_current.nums, axis=0)[0])
        if changes_needed > 0:
            from_sex = constants.Gender.MEN
        else:
            from_sex = constants.Gender.WOMEN
            changes_needed = -changes_needed

        to_sex = 1 - from_sex
        while changes_needed > 0:
            idx = math_prob.uniform(total_pop)
            if population[idx, 1] == from_sex:
                population[idx, 1] = to_sex
                changes_needed -= 1
                age = population[idx, 0]
                h_current.remove(age, from_sex)
                h_current.add(age, to_sex)

    def fit_single_loop(self, population, age_limits, age_limits_full, hh_cur_indexes, h_current) -> int:
        counter = 0
        age_indexes = self.get_age_indexes(population[:, 0], age_limits_full)
        for hi in range(len(hh_cur_indexes) - 1):
            hh_slice = slice(hh_cur_indexes[hi], hh_cur_indexes[hi + 1])
            hh_ages = population[hh_slice, 0]
            mv_slice = self.allowed_moves(hh_ages, self.max_move)
            moves_allowed = self.moves_all[mv_slice]
            zero_index = np.searchsorted(moves_allowed, 0)
            if moves_allowed[zero_index] != 0:
                assert False

            deltas = self.get_deltas(age_indexes[hh_slice, mv_slice], population[hh_slice, 1],
                                     age_limits.shape[0], zero_index)
            if np.any(np.sum(deltas, axis=1)):
                StdLog.log("PROBLEM")
                # np.testing.assert_array_equal(np.sum(deltas, axis=1), np.zeros([deltas.shape[0], deltas.shape[1]]))
            distances = h_current.distances(deltas)
            min_move = np.random.choice(np.flatnonzero(distances == distances.min()))
            if distances[min_move] < distances[zero_index]:
                # StdLog.log(f"Moving hh {hi} {moves_allowed[min_move]}")
                # move household by moves_allowed[min_move]
                population[hh_slice, 0] += moves_allowed[min_move]
                h_current.add_delta(deltas[min_move])
                counter += 1
        return counter

    def fit_population(self, population, hh_indexes, total_pop, h_current, age_limits_full, age_limits):
        # once for whole households, then for individual persons
        for hh_cur_indexes in [hh_indexes, range(total_pop + 1)]:
            counter = 1
            while counter > 0:
                counter = self.fit_single_loop(population, age_limits, age_limits_full, hh_cur_indexes, h_current)

    def population_by_households(self, population, hh_indexes):
        hh_indexes_set = set(hh_indexes)
        return ((population[idx, 0], constants.Gender(int(population[idx, 1])), idx in hh_indexes_set) for idx in
                range(population.shape[0]))

    def get_deltas(self, age_indexes: np.ndarray, sexes: np.ndarray, age_cats: int, zero_idx: int):
        # population x moves x age_cats x 2_sex
        (people, moves) = age_indexes.shape
        result = np.zeros([moves, age_cats, len(constants.Gender)], dtype=int)

        all_sexes = np.tile(sexes[:, None], (1, moves)).reshape(-1)
        all_indexes = age_indexes.reshape(-1)
        all_moves = np.tile(np.arange(moves)[:, None], (people, 1)).reshape(-1)

        # result[all_moves, all_indexes, all_sexes] += 1 #all_values
        np.add.at(result, (all_moves, all_indexes, all_sexes), 1)
        result[:, :, :] -= result[zero_idx, :, :]
        return result

    def allowed_moves(self, ages, max_move):
        """
        :param ages:
        :param max_move:
        :return:
        """
        min_move_idx = max(0, max_move - ages.min())
        max_move_idx = max_move + 1 + min(max_move, self.max_age - ages.max())
        return slice(min_move_idx, max_move_idx)

    def get_age_indexes(self, population: np.ndarray, age_limits_full: np.ndarray):
        """
        :param population: [age,sex] of all people
        :param age_limits_full: starting points of age compartements (including -INF and max_age)
        :return: rows for people, columns for moves
        indexes into age_limits for every move
        """
        ages = self.moves_all[None, :] + population[:, None]
        indexes = np.searchsorted(age_limits_full, ages, side='right') - 2
        indexes[indexes > age_limits_full.shape[0] - 3] = -1
        return indexes
