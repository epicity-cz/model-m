import matplotlib.pyplot as plt

from config.base_config import BaseConfig
from program.zujs import zujs_loader

SCALE = 300


def graph():
    hodoninsko = BaseConfig.create_from_inifile('hodoninsko')
    lounsko = BaseConfig.create_from_inifile('lounsko')

    xs = []
    ys = []
    ss = []
    cc = []
    xs1 = []
    ys1 = []
    ss1 = []
    xs2 = []
    ys2 = []
    ss2 = []
    orplist = []
    for (zuj, orp, cnt, x, y, nazev) in zujs_loader(hodoninsko):
        if orp not in orplist:
            orplist.append(orp)
        xs.append(-y)
        ys.append(-x)
        ss.append(cnt)
        cc.append(orplist.index(orp))
        if hodoninsko.contains(zuj, orp):
            xs1.append(-y)
            ys1.append(-x)
            ss1.append(cnt)
        if lounsko.contains(zuj, orp):
            xs2.append(-y)
            ys2.append(-x)
            ss2.append(cnt)

    plt.figure(figsize=(8, 6), dpi=300)
    plt.scatter(xs, ys, marker='.', s=[.5 + SCALE * s / max(ss) for s in ss], c=cc)
    plt.scatter(xs1, ys1, marker='.', s=[10 + SCALE * s / max(ss) for s in ss1])
    plt.scatter(xs2, ys2, marker='.', s=[10 + SCALE * s / max(ss) for s in ss2])

    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()


if __name__ == '__main__':
    graph()
