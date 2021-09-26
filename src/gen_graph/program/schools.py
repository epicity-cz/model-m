import math

from classes.person import Person
from classes.renv import Renv
from classes.school import school_by_type
from constants import constants
from lang.mytypes import List, Dict
from params.cal_param import CalParamIdx
from utils import math_prob
from utils.log import StdLog, LogSilent


def fill_schools(env: Renv, schools: Dict,
                 candidates: Dict[constants.SchoolType, List[Person]], logger=LogSilent):
    # SRC: graph.cpp:83-123

    StdLog.log(f"Filling schools with students")

    for st in constants.SchoolType:
        math_prob.shuffle(candidates[st])
        ne = 0
        school_type = school_by_type[st]
        for p in candidates[st]:
            if not env.school_members.put_to_school(school_type, p, schools[st], env.travel_info):
                ne += 1
        logger.log(f"{st.name} overflow {ne}")

    add_schools(env, logger,
                schools[constants.SchoolType.NURSARY],
                constants.Layer.NURSARY_CHILDREN_INCLASS,
                constants.Layer.NURSARY_CHILDREN_CORIDORS,
                constants.Layer.NURSARY_TEACHERS_TO_CHILDREN,
                constants.Layer.NURSARY_TEACHERS,
                env.config.FIRST_NURSARY,
                env.config.LAST_NURSARY,
                True)

    add_schools(env, logger,
                schools[constants.SchoolType.ELEMENTARY],
                constants.Layer.LOWER_ELEMENTARY_CHILDREN_INCLASS,
                constants.Layer.ELEMENTARY_CHILDREN_CORIDORS,
                constants.Layer.LOWER_ELEMENTARY_TEACHERS_TO_CHILDREN,
                constants.Layer.ELEMENTARY_TEACHERS,
                env.config.FIRST_ELEM,
                env.config.LAST_FIRST_ELEM,
                False)

    add_schools(env, logger,
                schools[constants.SchoolType.ELEMENTARY],
                constants.Layer.HIGHER_ELEMENTARY_CHILDREN_INCLASS,
                constants.Layer.ELEMENTARY_CHILDREN_CORIDORS,
                constants.Layer.HIGHER_ELEMENTARY_TEACHERS_TO_CHILDREN,
                constants.Layer.ELEMENTARY_TEACHERS,
                env.config.LAST_FIRST_ELEM + 1,
                env.config.LAST_ELEM,
                True)

    add_schools(env, logger,
                schools[constants.SchoolType.HIGH],
                constants.Layer.HIGHSCHOOL_CHILDREN_INCLASS,
                constants.Layer.HIGHSCHOOL_CHILDREN_CORIDORS,
                constants.Layer.HIGHSCHOOL_TEACHERS_TO_CHILDREN,
                constants.Layer.HIGHSCHOOL_TEACHERS,
                env.config.FIRST_HIGHSCHOOL,
                env.config.LAST_HIGHSCHOOL,
                True)


def add_schools(env: Renv, logger, sch_list, cclasslink, ccorlink, tlink, ttlink,
                first_age,
                last_age,
                do_random):
    for school in sch_list:
        count_classes = 0
        logger.log(f"\nSchool {school.name}")
        sm = env.school_members[school]
        for age in range(first_age, last_age + 1):
            logger.log(f"Age {age}", end='')
            if age in sm.all_ages():
                zaci = sm[age]
                total = len(zaci)
                num_classes = math.ceil(total / env.config.CLASS_SIZE)
                remains = total
                in_class = math.ceil(total / num_classes)
                count_classes += num_classes

                for start in range(0, total, in_class):
                    inc = min(in_class, remains)
                    remains -= inc
                    cls = zaci[start:start + inc]
                    logger.log(f" {inc}", end='')
                    env.mutual_contacts(cls, cclasslink, school.id, CalParamIdx.SCHOOL_CC_RATE_LAMBDAN)
                    teacher = env.w.employ(school.town, constants.WorkType.EDUCATION, env.travel_info)
                    if not teacher:
                        logger.log(f" (no teacher found)", end='')
                    else:
                        env.school_teachers[school].append(teacher)
                        env.teacher_class_contacts(teacher, cls, tlink, school.id, CalParamIdx.SCHOOL_TCC_RATE_LAMBDAN)
            logger.log('')
        if do_random:
            # hire additional teachers
            additionals = env.config.ADDITIONAL_TEACHERS * len(env.school_teachers[school])
            for _ in range(additionals):
                teacher = env.w.employ(school.town, constants.WorkType.EDUCATION, env.travel_info)
                if not teacher:
                    logger.log(f"Cannot hire all additional teachers.")
                    break
                else:
                    env.school_teachers[school].append(teacher)

            env.mutual_contacts(env.school_teachers[school], ttlink, school.id, CalParamIdx.SCHOOL_TT_RATE_LAMBDAN)
            students = list(sm)
            if count_classes > 1:
                env.mutual_contacts(students, ccorlink, school.id, CalParamIdx.SCHOOL_C_RATE_LAMBDAN)
            for teacher in env.school_teachers[school]:
                env.teacher_class_contacts(teacher, students, tlink, school.id, CalParamIdx.SCHOOL_TC_RATE_LAMBDAN)
