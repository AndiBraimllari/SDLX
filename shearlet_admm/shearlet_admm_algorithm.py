import numpy as np
from scipy.sparse import identity
from scipy.sparse.linalg import cg
from matplotlib import pyplot as plt
from ttictoc import tic, toc

from shearlet_admm_definitions import calc_J_from, shrink

from shearlet_transform.shearlet_transform_algorithm import applyShearletTransform as SH, \
    applyInverseShearletTransform as SHt


# NB the parameter rho is never used
def shearlet_admm(rho_zero, rho_one, rho_two, w, R_phi, y, max_iter=20):  # try having everything as a numpy array
    tic()

    n = int(np.sqrt(R_phi.shape[1]))
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
    # note here that the spectra (which takes the most time) of SH and SHt need only be computed only once
    while iterations < max_iter:
        A = rho_zero * np.dot(R_phi_t, R_phi) + (rho_one + rho_two) * I_n_squared
        b = rho_zero * np.dot(R_phi_t, y) + np.reshape(rho_one * SHt(np.reshape(P1z - P1u, (J, n, n))),
                                                       (n * n)) + rho_two * (P2z - P2u)
        f = cg(A, b)[0]  # TODO cg returns a tuple, cg(A, b)[1] is 0, what does that represent?

        # note that SH(...) returns both (SHf, spectra), therefore simply get the first one by [0]
        P1z = shrink(np.reshape(SH(np.reshape(f, (n, n)))[0], (J * n * n)) + P1u, rho_zero * w / rho_one)
        P2z = np.maximum(f + P2u, np.zeros(n ** 2))
        P1u = P1u + np.reshape(SH(np.reshape(f, (n, n)))[0], (J * n * n)) - P1z
        P2u = P2u + f - P2z

        f_norms.append(np.linalg.norm(f, ord=2))
        print(f_norms)
        iterations += 1

    print('Finished custom shearlet admm solver in ', toc())
    plt.plot(f_norms)
    plt.xlabel("number of iterations")
    plt.ylabel("L2 norm of solution")
    plt.show()
    return f

# ========== shapes of relevant objects ==========

# SH: J n^2 x n^2
# SH(f): J n x n
# Rphi: m x n^2
# y: m
# f: n^2
# eta: m

# note that I've used here J (decomposition subbands) and L from shearlet_transform_algorithm.py interchangeably, they're
# probably not identically the same


# ========== deprecated stuff ==========

# SHt =  # I think the inverse is implied here, not entirely sure how the transpose and inverse are connected
