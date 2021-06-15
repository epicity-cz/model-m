import numpy as np

from csu import reader

MEN = [('vec', '102=1')]  # Pohlaví muž
WOMEN = [('vec', '102=2')]  # Pohlaví žena
NOGENDER = [('vec', reader.cond_has_none, '102=')]

GENDERS = [MEN, WOMEN]

VEKY_PROBOF = [
    [('vec', '1035=1300150024')],  # Věk pro agregace  15 - 24 let
    [('vec', '1035=1300250034')],  # Věk pro agregace  25 - 34 let
    [('vec', '1035=1300350044')],  # Věk pro agregace  35 - 44 let
    [('vec', '1035=1300450054')],  # Věk pro agregace  45 - 54 let
    [('vec', '1035=1300550059')],  # Věk pro agregace  55 - 59 let
    [('vec', '1035=1300600064')],  # Věk pro agregace  60 - 64 let
    [('vec', '1035=1300650110')],  # Věk pro agregace  65 - 110 let
    #    [('vec', '1035=9799999999')],  # Věk pro agregace  Nezjištěno
]

VEKY_AGE_SEX = [
    [('vec', '1035=1100000002')],  # Věk pro agregace  0 - 2 roky
    [('vec', '1035=1300030004')],  # Věk pro agregace  3 - 4 roky
    [('vec', '1035=1000050005')],  # Věk pro agregace  5 let
    [('vec', '1035=1300060009')],  # Věk pro agregace  6 - 9 let
    [('vec', '1035=1300100014')],  # Věk pro agregace  10 - 14 let
    [('vec', '1035=1300150017')],  # Věk pro agregace  15 - 17 let
    [('vec', '1035=1300180019')],  # Věk pro agregace  18 - 19 let
    [('vec', '1035=1300200024')],  # Věk pro agregace  20 - 24 let
    [('vec', '1035=1300250029')],  # Věk pro agregace  25 - 29 let
    [('vec', '1035=1300300034')],  # Věk pro agregace  30 - 34 let
    [('vec', '1035=1300350039')],  # Věk pro agregace  35 - 39 let
    [('vec', '1035=1300400044')],  # Věk pro agregace  40 - 44 let
    [('vec', '1035=1300450049')],  # Věk pro agregace  45 - 49 let
    [('vec', '1035=1300500054')],  # Věk pro agregace  50 - 54 let
    [('vec', '1035=1300550059')],  # Věk pro agregace  55 - 59 let
    [('vec', '1035=1300600064')],  # Věk pro agregace  60 - 64 let
    [('vec', '1035=1300650069')],  # Věk pro agregace  65 - 69 let
    [('vec', '1035=1300700074')],  # Věk pro agregace  70 - 74 let
    [('vec', '1035=1300750079')],  # Věk pro agregace  75 - 79 let
    [('vec', '1035=1300800084')],  # Věk pro agregace  80 - 84 let
    [('vec', '1035=1300850110')],  # Věk pro agregace  85 - 110 let

    #        '1035=1100000014', #Věk pro agregace  0 - 14 let
    #        '1035=1300150064', #Věk pro agregace  15 - 64 let
    #        '1035=1300650110', #Věk pro agregace  65 - 110 let
]

AGE_LIMITS = np.array([
    0,
    3,
    5,
    6,
    10,
    15,
    18,
    20,
    25,
    30,
    35,
    40,
    45,
    50,
    55,
    60,
    65,
    70,
    75,
    80,
    85
])


def load(zuj):
    reader112 = reader.Reader(zuj, 112)
    reader113 = reader.Reader(zuj, 113)
    reader114 = reader.Reader(zuj, 114)
    reader115 = reader.Reader(zuj, 115)
    reader116 = reader.Reader(zuj, 116)
    reader122 = reader.Reader(zuj, 122)

    by_age = reader112.array(
        [('uka', '3162')],  # Počet obyvatel s obvyklým pobytem
        VEKY_AGE_SEX,
        GENDERS,
        reader112.value_int
    )
    age_sex = by_age / by_age.sum()

    inhabitants = reader112.value_int([
        ('uka', '3162'),  # Počet obyvatel s obvyklým pobytem
        ('vec', reader.cond_has_none),
    ])

    ecactivity = reader113.array(
        [('uka', '3162')],  # Počet obyvatel s obvyklým pobytem
        [
            [('vec', '3249=51')],
            # Ekonomická aktivita - agregace Zaměstnaní včetně pracujících studentů a učňů # G 12+13+14+15
            [('vec', '3249=52')],  # Ekonomická aktivita - agregace Nezaměstnaní # G16
            [('vec', '3072=7')],  # Ekonomická aktivita Ostatní s vlastním zdrojem obživy # G21
            [('vec', '3072=13')],
            # Ekonomická aktivita Osoby v domácnosti, děti předškolního věku, ostatní závislé osoby # G22
            [('vec', '3072=8')],  # Ekonomická aktivita Žáci, studenti, učni # G23
            [('vec', '3072=6')],  # Ekonomická aktivita Nepracující důchodci # G20
            #                ('vec', '3072:99'),  # Ekonomická aktivita Nezjištěno # G24
        ],
        GENDERS,
        reader113.value_int
    )

    ecactivity = np.vstack([ecactivity[0], ecactivity[1:4].sum(axis=0), ecactivity[4:6]])
    ecactivity = ecactivity / ecactivity.sum(axis=0)

    ec = ecactivity.T

    probofdata = reader116.array(
        [('uka', '3162')],  # Počet obyvatel s obvyklým pobytem
        VEKY_PROBOF,
        [
            NOGENDER,
            WOMEN,
        ],
        reader116.value_int
    )
    probofdata[:, 0] -= probofdata[:, 1]

    group_counts = np.vstack([
        by_age[5:8].sum(axis=0),
        by_age[8:10].sum(axis=0),
        by_age[10:12].sum(axis=0),
        by_age[12:14].sum(axis=0),
        by_age[14:16],
        by_age[16:21].sum(axis=0),
    ])
    probofwork = (probofdata / group_counts).T

    ecinnost = reader114.array(
        [('uka', '3162')],  # Počet obyvatel s obvyklým pobytem
        [
            #	Klasifikace ekonomických činností (CZ-NACE) - úroveň 1 - Sekce
            [('vec', '5103=A')],  # Zemědělství, lesnictví, rybářství
            [('vec', '5724=05399999')],
            [('vec', '5103=F')],  # Stavebnictví
            [('vec', '5103=G')],  # Velkoobchod a maloobchod; opravy a údržba motorových vozidel
            [('vec', '5103=H')],  # Doprava a skladování
            [('vec', '5103=I')],  # Ubytování, stravování a pohostinství
            [('vec', '5103=J')],  # Informační a komunikační činnosti
            [('vec', '5103=K')],  # Peněžnictví a pojišťovnictví
            [('vec', '5724=68829999')],  # Činnosti v oblasti nemovitostí, ...
            [('vec', '5103=O')],  # Veřejná správa a obrana; povinné sociální zabezpečení
            [('vec', '5103=P')],  # Vzdělávání
            [('vec', '5103=Q')],  # Zdravotní a sociální péče
        ],
        [NOGENDER],
        reader114.value_int
    )

    eactivity = ecinnost[:, 0] / ecinnost.sum()

    zam_muz_zena = reader114.array(
        [('uka', '3162')],  # Počet obyvatel s obvyklým pobytem
        GENDERS,
        [[('vec', '3249=51')]],  # Ekonomická aktivita - agregace , Zaměstnaní včetně pracujících studentů a učňů
        reader114.value_int
    )

    transport = reader115.array(
        [
            ('uka', '3162'),  # Počet obyvatel s obvyklým pobytem
            ('vec', '3323=50'),
            # Frekvence dojížďky do zaměstnání a školy - agregace , Denní vyjížďka do zaměstnání a školy (v ČR i mimo ČR)
        ],
        [
            MEN,
            WOMEN,
            [('vec', '3249=56')]  # Ekonomická aktivita - agregace , Žáci, studenti a učni
        ],
        [
            [('vec', reader.cond_has_none, '3322=')],  # Doba trvání denní dojížďky/docházky - agregace
            [('vec', '3322=50')],
            # Doba trvání denní dojížďky/docházky - agregace , Doba denní dojížďky do 14 minut
            [('vec', '3322=51')],
            # Doba trvání denní dojížďky/docházky - agregace , Doba denní dojížďky 15 - 29 minut
            [('vec', '3322=52')],
            # Doba trvání denní dojížďky/docházky - agregace , Doba denní dojížďky 30 - 44 minut
            [('vec', '3322=53')],
            # Doba trvání denní dojížďky/docházky - agregace , Doba denní dojížďky 45 - 59 minut
            [('vec', '3322=54')],
            # Doba trvání denní dojížďky/docházky - agregace , Doba denní dojížďky 60 - 89 minut
            [('vec', '3322=55')],
            # Doba trvání denní dojížďky/docházky - agregace , Doba denní dojížďky 90 a více minut
        ],
        reader115.value_int
    )
    koefs = transport[:, 0:1].astype(float)

    # v prvnim sloupci jsou pocty cestujicich
    # uprav muzi,zeny
    koefs[0:2] = koefs[0:2] / zam_muz_zena

    # ratio of >15 yr between students
    ratio_students = by_age[5:7, :].sum() / by_age[3:7, :].sum()

    students = reader113.value_int([
        ('uka', '3162'),  # Počet obyvatel s obvyklým pobytem
        ('vec', '3072=8'),  # Ekonomická aktivita , Žáci, studenti, učni
    ])

    koefs[2] = 1
    rat = ratio_students * students
    if rat > 0:
        koefs[2] = min(transport[2, 0] / ratio_students / students, 1)

    table = transport[:, 1:]

    commuting = np.hstack([1 - koefs, table / table.sum(axis=1).reshape(3, 1) * koefs])

    apartments_counts = reader122.array(
        [('uka', '2607')],  # Počet bytů
        [[('vec', reader.cond_has_none, '3240=')]],
        [
            [('vec', '7600=1')],  # Hodnotový číselník
            [('vec', '7600=2')],  # Hodnotový číselník
            [('vec', '7600=3')],  # Hodnotový číselník
            [('vec', '7600=4')],  # Hodnotový číselník
            [('vec', '7600=5')],  # Hodnotový číselník
            [('vec', '7601=00060070')],  # Hodnotový číselník - agregace 6-70
        ],
        reader122.value_int
    )[0]
    return (AGE_LIMITS, age_sex, inhabitants, ec, probofwork, eactivity, commuting, apartments_counts)
