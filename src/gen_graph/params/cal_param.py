from lang.mytypes import EnumZ, auto
from params.parameters import ParamIdx
from utils import math_prob

calibrate_functions = [
    lambda params, plink: 1,
    lambda params, plink: params[ParamIdx.FAMILY_HOME_DELTA],
    lambda params, plink: params[ParamIdx.FAMILY_HOME_DELTA] ** 2,
    lambda params, plink: params[ParamIdx.FAMILY_PROB_INTER_GEN],
    lambda params, plink: params[ParamIdx.FAMILY_PROB_INTER_GEN] * params[ParamIdx.FAMILY_HOME_DELTA],
    lambda params, plink: params[ParamIdx.FAMILY_PROB_INTER_GEN] * params[ParamIdx.FAMILY_HOME_DELTA] ** 2,
    lambda params, plink: params[ParamIdx.FAMILY_JUNIOR_VISIT],
    lambda params, plink: params[ParamIdx.FAMILY_SENIOR_VISIT],
    lambda params, plink: params[ParamIdx.SCHOOL_CC_RATE],
    lambda params, plink: params[ParamIdx.SCHOOL_TCC_RATE],
    lambda params, plink: params[ParamIdx.SCHOOL_TT_RATE],
    lambda params, plink: params[ParamIdx.SCHOOL_C_RATE],
    lambda params, plink: params[ParamIdx.SCHOOL_TCC_RATE],
    lambda params, plink: math_prob.lambdan(params[ParamIdx.SCHOOL_CC_RATE], plink),
    lambda params, plink: math_prob.lambdan(params[ParamIdx.SCHOOL_TCC_RATE], plink),
    lambda params, plink: math_prob.lambdan(params[ParamIdx.SCHOOL_TT_RATE], plink),
    lambda params, plink: math_prob.lambdan(params[ParamIdx.SCHOOL_C_RATE], plink),
    lambda params, plink: math_prob.lambdan(params[ParamIdx.SCHOOL_TC_RATE], plink),
]


class CalParamIdx(EnumZ):
    ONE = auto()
    FAM_ONE = ONE
    FAM_HOME_DELTA = auto()
    FAM_HOME_DELTA_SQUERED = auto()

    FAM_INTER_GEN_ONE = auto()
    FAM_INTER_GEN_HOME_DELTA = auto()
    FAM_INTER_GEN_HOME_DELTA_SQUERED = auto()

    FAM_JUNIOR_VISIT = auto()
    FAM_SENIOR_VISIT = auto()

    SCHOOL_CC_RATE = auto()
    SCHOOL_TCC_RATE = auto()
    SCHOOL_TT_RATE = auto()
    SCHOOL_C_RATE = auto()
    SCHOOL_TC_RATE = auto()

    SCHOOL_CC_RATE_LAMBDAN = auto()
    SCHOOL_TCC_RATE_LAMBDAN = auto()
    SCHOOL_TT_RATE_LAMBDAN = auto()
    SCHOOL_C_RATE_LAMBDAN = auto()
    SCHOOL_TC_RATE_LAMBDAN = auto()

    @staticmethod
    def calculate(idx, params, plink):
        return calibrate_functions[idx](params, plink)
