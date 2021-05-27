import numpy as np


def _random_from_probs(what, p, n=1):
    return np.random.choice(what, p=p, size=n)


def _check_sorted(value_list):
    """ list of numpy arrays
    checks x1i < x2i < ... < xni

    returns True/False arrays
    True are valid points
    """
    # print()
    # print(value_list)
    # print()
    partial_results_list = [
        np.atleast_1d(x < y)
        for x, y in zip(value_list[:-1], value_list[1:])
    ]
    #    print()

    #    print("partial_results_list", partial_results_list, len(partial_results_list))
    if len(partial_results_list) > 1:
        return np.logical_and(*partial_results_list)
    else:
        return partial_results_list[0]


def gen_tuple1(n, shape, *args):
    """ generate n-tuple of  values, r_1 < r_2 < .... < r_n

    shape ... shape of generated "values" 

    args* - random duration generators objects 

    Example:
         >>> gen_tuple(3, rng1, rng2, rng3)
    """

    def _gen(s):
        result = []
        for i in range(n):
            result.append(args[i].get(n=s))
        return result

    assert len(args) == n

    result = _gen(shape)
    # print("-----------------")
    # print(result)
    # print("-----------------")

    check = _check_sorted(result)
    while not np.all(check):
        print("DBG Houston we have a problem")
        indices_to_fix = np.where(check == False)[0]
        new_values = _gen(indices_to_fix.shape[0])  # list of length n again
        # but with shorter items

        for i in range(n):
            result[i][indices_to_fix] = new_values[i].reshape(-1, 1)
        check = _check_sorted(result)

    return result


def gen_tuple2(n, shape, *args):
    result = []
    for i in range(n):
        values = args[i].get(n=shape)
        if i > 0:
            values = np.clip(values, result[i-1]+1, None)
        result.append(values)

    return result

def gen_tuple(n, shape, *args):
    return gen_tuple2(n, shape, *args)

class RandomDuration():

    def __init__(self, probs, precompute=False):
        self.N = len(probs)
        self.probs = probs

        if precompute:
            buf = _random_from_probs(self.N, self.probs, 10**6)

    def get(self, n=1):
        values = _random_from_probs(self.N, self.probs, n)

        return values


if __name__ == "__main__":

    import matplotlib.pyplot as plt

    def uncumulate(l):
        res = [x - y
               for (x, y) in zip(l, [0]+l[:-1])
               ]
        s = sum(res)
        res[-1] += 1.0-s
        return res

    cdf_incubation = [0, 0.002079467, 0.045532967, 0.158206035, 0.303711753, 0.446245776, 0.569141375, 0.668484586,
                      0.746107988, 0.805692525, 0.851037774, 0.885435436, 0.911529759, 0.931365997, 0.946495014,
                      0.958080947, 0.966993762, 0.973882948, 0.979233968, 0.983410614, 0.986686454, 0.98926803,
                      0.991311965, 0.992937571, 0.994236158, 0.995277934, 0.996117131, 0.996795835, 0.997346849,
                      0.997795859, 0.998163058, 0.998464392, 0.998712499, 0.998917441, 0.999087255, 0.999228384,
                      0.999346016, 0.999444337, 0.999526742, 0.999595989, 0.999654327]
    p_incubation = uncumulate(cdf_incubation)

    values = []
    values2 = []
    durations = RandomDuration(p_incubation)
    pre_durations = RandomDuration(p_incubation, precompute=True)

    for _ in range(10000):

        values.extend(durations.get(100))
        values2.extend(durations.get(100))

    print(np.mean(values), np.median(values))
    print(np.mean(values2), np.median(values2))
    print(np.max(values), np.max(values2))
    max_value = max(values + values2)
    min_value = min(values + values2)

    fig, axs = plt.subplots(nrows=2, figsize=(10, 7))
    axs[0].hist(values, color="pink", label="onfly",
                bins=range(min_value, max_value+1))
    axs[0].hist(values2, color="blue", label="precomputed",
                bins=range(min_value, max_value+1))

    axs[1].hist(values2, color="blue", label="precomputed",
                bins=range(min_value, max_value+1))
    axs[1].hist(values, color="pink", label="onfly",
                bins=range(min_value, max_value+1))

    axs[0].set_xticks(range(min_value, max_value+1))
    axs[1].set_xticks(range(min_value, max_value+1))

    axs[0].legend()
    axs[1].legend()
    fig.suptitle("days in E")

    # Show plot
    plt.show()
