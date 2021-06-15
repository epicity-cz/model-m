from config.base_config import BaseConfig
from constants import constants
from lang.mytypes import Any, Optional
from utils import math_prob
from utils.has_id import HasId


class Person(HasId):
    household: Any
    town: Any
    sex: constants.Gender
    activity: constants.EconoActivity
    worktype: constants.WorkType
    commute: constants.CommutingTime
    school_type: Optional[constants.SchoolType]

    def __init__(self, region_config: BaseConfig, age: float, sex: constants.Gender, town):
        self.gen_id()
        if age > region_config.MAX_AGE:
            age = region_config.MAX_AGE
        self.age = age
        self.sex = sex
        self.activity = constants.EconoActivity.AT_HOME
        self.worktype = constants.WorkType.UNDEF
        self.commute = constants.CommutingTime.NOT_COMMUTING
        self.town = town
        self.school_type = None

        age_idx = constants.get_age_idx(age)
        if age_idx is not None:
            prob = town.probofwork[sex][age_idx]
            working = math_prob.yes_no(prob)
        else:
            working = False
        if working:
            self.activity = constants.EconoActivity.WORKING
            self.worktype = constants.WorkType(math_prob.draw(town.eactivity))
        else:
            if region_config.FIRST_ELEM <= age <= region_config.LAST_HIGHSCHOOL:
                self.activity = constants.EconoActivity.STUDENT
                if age <= region_config.LAST_ELEM:
                    self.school_type = constants.SchoolType.ELEMENTARY
                else:
                    self.school_type = constants.SchoolType.HIGH
            elif region_config.LAST_HIGHSCHOOL < age <= region_config.LAST_POT_STUDENT:
                if math_prob.yes_no(region_config.PROB_OF_UNIV[sex]):
                    self.activity = constants.EconoActivity.STUDENT
                else:
                    self.activity = constants.EconoActivity.AT_HOME
            elif region_config.FIRST_RETIRED <= age:
                self.activity = constants.EconoActivity.RETIRED
            else:
                self.activity = constants.EconoActivity.AT_HOME
                if region_config.FIRST_NURSARY <= age <= region_config.LAST_NURSARY:
                    self.school_type = constants.SchoolType.NURSARY
        if working:
            self.commute = constants.CommutingTime(math_prob.draw(town.commuting[sex]))
        elif self.activity == constants.EconoActivity.STUDENT and region_config.LAST_ELEM < self.age:
            self.commute = constants.CommutingTime(math_prob.draw(town.commuting[2]))
        else:
            self.commute = constants.CommutingTime.NOT_COMMUTING
