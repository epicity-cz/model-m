from scipy.stats import gamma, lognorm
import matplotlib.pyplot as plt
import seaborn as sns
import random
import numpy as np


def m_rand(shape):
    return gamma.rvs(a=6.25, loc=0.83, size=shape)


def m_rand_lognorm(shape):
    return lognorm.rvs(s=1.69, loc=0.59, size=shape)


def my_tes():
    a = m_rand(1000000)
    print(a.shape)
    ax = sns.distplot(a, kde=False)
    plt.show()


def uncumulate(l):
    res = [x - y
           for (x, y) in zip(l, [0]+l[:-1])]

    s = sum(res)
    res[-1] += 1.0-s
    return res


# def ro_test():
cdf_incubation = [0, 0.002079467, 0.045532967, 0.158206035, 0.303711753, 0.446245776, 0.569141375, 0.668484586,
                  0.746107988, 0.805692525, 0.851037774, 0.885435436, 0.911529759, 0.931365997, 0.946495014,
                  0.958080947, 0.966993762, 0.973882948, 0.979233968, 0.983410614, 0.986686454, 0.98926803,
                  0.991311965, 0.992937571, 0.994236158, 0.995277934, 0.996117131, 0.996795835, 0.997346849,
                  0.997795859, 0.998163058, 0.998464392, 0.998712499, 0.998917441, 0.999087255, 0.999228384,
                  0.999346016, 0.999444337, 0.999526742, 0.999595989, 0.999654327]
cdf_incubation = uncumulate(cdf_incubation)

cdf_incubation_n = len(cdf_incubation)

cdf_presymptomatic = [0, 0.003482312, 0.260336004, 0.712431613, 0.921306035, 0.981046861, 0.995561091,
                      0.998943994, 0.999739899, 0.999933149, 0.999982026]
cdf_presymptomatic_n = len(cdf_presymptomatic)
cdf_presymptomatic = uncumulate(cdf_presymptomatic)

cdf_infectious = [0, 0.000972053, 0.027261966, 0.130505192, 0.310094911, 0.513116044, 0.68952814, 0.818010738,
                  0.900473837, 0.948609221, 0.974707539, 0.988046458, 0.994542641, 0.997581623, 0.998955742,
                  0.999559221, 0.999817647, 0.999925895, 0.999970363, 0.999988316, 0.999995453]
cdf_infectious_n = len(cdf_infectious)
cdf_infectious = uncumulate(cdf_infectious)

cdf_rna = [0, 0.000001254, 0.000101098, 0.001293381, 0.006802684, 0.021597332, 0.049985578, 0.093656605,
           0.151354276, 0.219736581, 0.294566617, 0.371686953, 0.447602813, 0.519715908, 0.586323319, 0.646491358,
           0.69988333, 0.746588174, 0.786972879, 0.821566373, 0.850974254, 0.875819917, 0.896706521, 0.914194456,
           0.928789776, 0.940940047, 0.951034918, 0.959409525, 0.966349392, 0.972095958, 0.976852173, 0.980787825,
           0.984044431, 0.986739588, 0.988970795, 0.990818739, 0.992350106, 0.993619956, 0.99467372, 0.99554887,
           0.996276305, 0.996881508, 0.997385495, 0.997805609, 0.998156162, 0.998448976, 0.998693819, 0.998898769, 0.999070513, 0.999214588]
cdf_rna_n = len(cdf_rna)
cdf_rna = uncumulate(cdf_rna)


def random_from_cdf(k, cw, n=1):
    return np.random.choice(k, p=cw, size=n)
#    return np.random.choice(what, size=n)


def days_in_R(n=1):
    return(random_from_cdf(cdf_rna_n, cdf_rna,  n))


def days_in_E(n=1):
    return(random_from_cdf(cdf_incubation_n, cdf_incubation, n))


def days_in_I(n=1):
    return(random_from_cdf(cdf_infectious_n, cdf_infectious, n))


def days_in_A(n=1):
    return(random_from_cdf(cdf_presymptomatic_n, cdf_presymptomatic, n))


if __name__ == "__main__":

    N = 10**3

    if True:
        import timeit

        def petra():
            return m_rand_lognorm(N)

        def roman():
            return days_in_E(N)

        print(f"Scipy: {timeit.timeit(petra, number=1000)}")
        print(f"Roman: {timeit.timeit(roman, number=1000)}")

        buf = days_in_E(10**6)

        def petra2():
            return np.random.choice(buf, size=N)

        def roman2():
            return days_in_E(N)

        print(f"Petra2: {timeit.timeit(petra2, number=1000)}")
        print(f"Roman2: {timeit.timeit(roman2, number=1000)}")

    else:
            #    petra = [round(x) for x in m_rand(N)]
        petra = []
        roman = []

        buf = days_in_I(10**6)

        for _ in range(1000):
            petra.extend(np.random.choice(buf, size=N))
            roman.extend(days_in_I(N))

        print(np.mean(petra), np.median(petra))
        print(np.max(petra), np.max(roman))

        # petra = np.random.choice(roman, size=100000)
        # roman = days_in_I(100000)

        fig, axs = plt.subplots(nrows=2, figsize=(10, 7))
        axs[0].hist(petra, color="pink", label="petra", bins=20)
        axs[0].hist(roman, color="blue", label="roman", bins=20)

        axs[1].hist(roman, color="blue", label="roman", bins=20)
        axs[1].hist(petra, color="pink", label="petra", bins=20)

        axs[0].set(xlim=(0, 21))
        axs[1].set(xlim=(0, 21))

        axs[0].legend()
        axs[1].legend()
        fig.suptitle("days in I")

        # Show plot
        plt.show()
