import numpy as np

from constants import constants
from csu import town_loader
from utils.persistence import Persistent


class TownParams(Persistent):
    age_limits: np.ndarray
    age_sex: np.ndarray  # Gender x age_cat
    ec: np.ndarray  # Gender x EconoActivity
    probofwork: np.ndarray  # Gender x WORK_AGE_CATS
    eactivity: np.ndarray
    commuting: np.ndarray
    apartments_counts: np.ndarray
    zuj: str
    inhabitants: int

    @classmethod
    def dump_attributes(cls):
        return ['age_sex',
                'ec',
                'probofwork',
                'eactivity',
                'commuting',
                'apartments_counts',
                'zuj',
                'inhabitants',
                'age_limits',
                ]

    def __init__(self, zuj):
        self.zuj = zuj

    def valid(self):
        return self.ec.shape == (len(constants.Gender), len(constants.EconoActivity)) and \
               self.probofwork.shape == (len(constants.Gender), len(constants.WORK_AGE_CATS))

    def load_csu(self):
        (self.age_limits,
         self.age_sex,
         self.inhabitants,
         self.ec,
         self.probofwork,
         self.eactivity,
         self.commuting,
         self.apartments_counts) = town_loader.load(self.zuj)
