from math import exp


def get_couplings(z1, z2, x1):
    y1_mda = (-0.1 + ((z1 - 0.1) ** 2 + 0.8 * z2 + x1) ** (0.5)) ** 2
    y2_mda = z1 + z2 - 0.1 + ((z1 - 0.1) ** 2 + 0.8 * z2 + x1) ** (0.5)
    return y1_mda, y2_mda


def get_objective(x1, z2, y1, y2):
    return x1 ** 2. + z2 + y1 + exp(-y2)


def get_g1(y1):
    return y1/3.16 - 1.


def get_g2(y2):
    return 1. - y2/24.