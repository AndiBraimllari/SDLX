import numpy as np
from scipy.sparse import identity
from scipy.sparse.linalg import cg
from matplotlib import pyplot as plt
from ttictoc import tic, toc

from shearlet_admm.shearlet_admm_definitions import calc_J_from
from shearlet_transform.shearlet_transform import applyShearletTransform as SH, applyInverseShearletTransform as SHt


# NB the parameter rho is never used
def shearlet_admm(max_iter, rho_zero, rho_one, rho_two, w, R_phi, y):  # try having everything as a numpy array
    tic()

    n = np.sqrt(R_phi.shape[1])
    J = calc_J_from(n)
    I_n_squared = identity(n ** 2)
    R_phi_t = R_phi.transpose()
    f = np.dot(R_phi_t, y)
    P1z = np.zeros(J * n ** 2)
    P2z = np.zeros_like(f)
    P1u = np.zeros_like(P1z)
    P2u = np.zeros_like(f)

    f_norms = []
    iterations = 0
    while iterations < max_iter:
        A = rho_zero * np.dot(R_phi_t, R_phi) + (rho_one + rho_two) * I_n_squared
        b = rho_zero * np.dot(R_phi_t, y) + rho_one * SHt(P1z - P1u) + rho_two * (P2z - P2u)
        f = cg(A, b)

        P1z = shrink(SH(np.reshape(f, (n, n))) + P1u, rho_zero * w / rho_one)
        P2z = max(f + P2u, np.zeros(n ** 2))
        P1u = P1u + SH(np.reshape(f, (n, n))) - P1z
        P2u = P2u + f - P2z

        f_norms.append(np.linalg.norm(f, ord=2))
        iterations += 1

    print('Finished custom shearlet admm solver in ', toc())
    plt.imshow(f_norms)
    plt.show()
    return f

# ========== shapes of relevant objects ==========

# SH: J n^2 x n^2
# SH(f): J n x n
# Rphi: m x n^2
# y: m
# f: n^2
# eta: m

# note that I've used here J (decomposition subbands) and L from shearlet_transform.py interchangeably, they're
# probably not identically the same


# ========== deprecated stuff ==========

# SHt =  # I think the inverse is implied here, not entirely sure how the transpose and inverse are connected
