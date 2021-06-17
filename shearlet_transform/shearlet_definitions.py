import numpy as np


def v(x):
    if x < 0:
        return 0
    elif 0 <= x <= 1:
        return 35 * x ** 4 - 84 * x ** 5 + 70 * x ** 6 - 20 * x ** 7
    else:
        return 1


def b(w):
    if 1 <= abs(w) <= 2:
        return np.sin(np.pi / 2 * v(abs(w) - 1))
    elif 2 < abs(w) <= 4:
        return np.cos(np.pi / 2 * v(1 / 2 * abs(w) - 1))
    else:
        return 0


def phi(w):
    if abs(w) <= 1 / 2:
        return 1
    elif 1 / 2 < abs(w) < 1:
        return np.cos(np.pi / 2 * v(2 * abs(w) - 1))
    else:
        return 0


def phiHat(w1, w2):
    if abs(w1) <= 1 / 2 and abs(w2) <= 1 / 2:
        return 1
    elif 1 / 2 < abs(w1) < 1 and abs(w2) <= abs(w1):
        return np.cos(np.pi / 2 * v(2 * abs(w1) - 1))
    elif 1 / 2 < abs(w2) < 1 and abs(w1) < abs(w2):
        return np.cos(np.pi / 2 * v(2 * abs(w2) - 1))
    else:
        return 0


def psiHat1(w):
    return np.sqrt(b(2 * w) ** 2 + b(w) ** 2)


def psiHat2(w):
    if w <= 0:
        return np.sqrt(v(1 + w))
    else:
        return np.sqrt(v(1 - w))


def psiHat(w1, w2):  # separable generating function?
    # AFAIK this is not explicitly defined on w1 == 0, but psiHat1(0) is 0 so simply return 0
    if w1 == 0:
        return 0
    return psiHat1(w1) * psiHat2(w2 / w1)
