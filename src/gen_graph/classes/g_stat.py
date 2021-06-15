from constants import constants
from lang.mytypes import List
from plos.plos_matrix import PlosMatrix


class GStatResult:
    pm: List[PlosMatrix]
    ref: List[PlosMatrix]
    degree: float

    def __init__(self):
        self.pm = [PlosMatrix() for _ in constants.PlosCats]
        self.ref = [PlosMatrix() for _ in constants.PlosCats]
