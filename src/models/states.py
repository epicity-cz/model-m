# states for agent based model


class STATES():
    S = 0
    S_s = 1
    E = 2
    I_n = 3
    I_a = 4
    I_s = 5
    J_s = 6
    J_n = 7
    R = 8
    D = 9
    EXT = 10

    pass


state_codes = {
    STATES.S:     "S",
    STATES.S_s:   "S_s",
    STATES.E:     "E",
    STATES.I_n:   "I_n",
    STATES.I_a:   "I_a",
    STATES.I_s:   "I_s",
    STATES.J_s:   "J_s",
    STATES.J_n:   "J_n",
    STATES.R:   "R",
    STATES.D:   "D",
    STATES.EXT: "EXT"
}
