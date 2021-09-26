import math
import random as randomlib
# do not remove even if not used directly in this file
# noinspection PyUnresolvedReferences
from random import choice, sample, shuffle, random, randint

# noinspection PyUnresolvedReferences
from numpy.random import poisson

from lang.mytypes import List, Optional


def yes_no(prob: float):
    return random() < prob


def uniform(count):
    return randint(0, count - 1)


def draw(probs: List[float], population: Optional[List] = None):
    idx = randomlib.choices(range(len(probs)), weights=probs)[0]
    if population:
        return population[idx]
    return idx


def draw_distinct(probs: Optional[List[float]], samples: int, population: Optional[List] = None):
    if probs:
        w_probs = probs.copy()
    else:
        w_probs = [1 for _ in population]
    ret = []
    for i in range(samples):
        idx = draw(w_probs)
        ret.append(population[idx] if population else idx)
        w_probs[idx] = 0
    return ret


def draw_distinct_uniform(sample_size, population: list):
    return randomlib.sample(population, sample_size)


def draw_distinct_weighted(weights: List[float], sample_size, population: Optional[List] = None):
    return draw_distinct(weights, sample_size, population)


def lambdan(lmbda: float, n: float):
    nu = round(n)
    p = math.exp(-lmbda)
    s = 0
    sp = p
    for i in range(1, nu):
        p *= lmbda / i
        sp += p
        s += i * p
    return s + (1 - sp) * nu


def distw(d, distpref):
    return math.exp(d / 1000 * math.log(1 - distpref))
