import numpy as np

from lang.mytypes import EnumZ, auto


class ParamIdx(EnumZ):
    FAMILY_HOME_DELTA = auto()
    FAMILY_PROB_INTER_GEN = auto()
    FAMILY_JUNIOR_VISIT = auto()
    FAMILY_SENIOR_VISIT = auto()

    SCHOOL_CC_RATE = auto()
    SCHOOL_TCC_RATE = auto()
    SCHOOL_TT_RATE = auto()
    SCHOOL_C_RATE = auto()
    SCHOOL_TC_RATE = auto()

    @staticmethod
    def slice_family():
        return slice(ParamIdx.FAMILY_HOME_DELTA, ParamIdx.FAMILY_SENIOR_VISIT + 1)

    @staticmethod
    def slice_school():
        return slice(ParamIdx.SCHOOL_CC_RATE, ParamIdx.SCHOOL_TC_RATE + 1)


class Parameters(np.ndarray):
    def __new__(cls, input_array):
        if input_array is None:
            obj = np.zeros(len(ParamIdx)).view(cls)
        else:
            obj = np.asarray(input_array).view(cls)
            assert obj.shape == (len(ParamIdx),)
        return obj
