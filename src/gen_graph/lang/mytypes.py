# do not remove even if not used directly in this file
# noinspection PyUnresolvedReferences
from enum import IntEnum, auto, Enum
# noinspection PyUnresolvedReferences
from typing import Tuple, Dict, List, Any, Iterator, Type, Optional, Set, DefaultDict

import numpy as np

Location = Tuple[float, float]


# zero based enum
class EnumZ(IntEnum):
    def _generate_next_value_(name, start: int, count: int, last_values: List[Any]) -> int:
        return IntEnum._generate_next_value_(name, 0, count, last_values)

    def label(self):
        return self.name.lower()


class AutoName(Enum):
    def _generate_next_value_(name, start, count, last_values):
        return name

    def label(self):
        return self.name.lower()


class Params:
    value: np.ndarray

    @classmethod
    def indexes(cls) -> EnumZ: pass

    def __str__(self):
        return " ".join([f"{i}={self.value[i]}" for i in self.indexes()])

    def scale(self, ratio):
        self.value *= ratio

    def as_weights(self): pass
