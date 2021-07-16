import numpy as np
from scipy.sparse import identity
from scipy.sparse.linalg import cg
from matplotlib import pyplot as plt
from ttictoc import tic, toc

from shearlet_admm.shearlet_admm_definitions import shrink
from shearlet_transform.shearlet_transform_algorithm import applyShearletTransform as SH, \
    applyInverseShearletTransform as SHt, calc_L_from_shape


# the parameter rho is never used
# try having everything as a numpy array
def shearlet_admm(rho_zero, rho_one, rho_two, w, R_phi, y, max_iter=20, L=None):
    """
    TODO reword
    Calculates the spectra for a given shape. One can also specify the parabolic scaling (dilation) and shearing, as
    well as the number scales

    Parameters:
    rho_zero (int): .
    rho_one (int): .
    rho_two (int): .
    w (int): L1 weights.
    R_phi (int): .
    y (int): .
    max_iter (int): Maximum number of iterations.
    L (int): Number of scales.

    Returns:
    numpy.ndarray: 2D object of shape (M, N) containing an image
    """
    tic()

    n = int(np.sqrt(R_phi.shape[1]))
    if L is None:
        L = calc_L_from_shape(n)
    I_n_squared = identity(n ** 2)
    R_phi_t = R_phi.transpose()
    f = np.dot(R_phi_t, y)
    P1z = np.zeros(L * n ** 2)
    P2z = np.zeros_like(f)
    P1u = np.zeros_like(P1z)
    P2u = np.zeros_like(f)

    f_norms = []
    iterations = 0
    # note here that the spectra (which takes the most time) of SH and SHt need only be computed only once
    while iterations < max_iter:
        A = rho_zero * np.dot(R_phi_t, R_phi) + (rho_one + rho_two) * I_n_squared
        b = rho_zero * np.dot(R_phi_t, y) + np.reshape(rho_one * SHt(np.reshape(P1z - P1u, (L, n, n))),
                                                       (n * n)) + rho_two * (P2z - P2u)
        f = cg(A, b)[0]  # TODO cg returns a tuple, cg(A, b)[1] is 0, what does that represent?

        # note that SH(...) returns both (SHf, spectra), therefore simply get the first one by [0]
        P1z = shrink(np.reshape(SH(np.reshape(f, (n, n)))[0], (L * n * n)) + P1u, rho_zero * w / rho_one)
        P2z = np.maximum(f + P2u, np.zeros(n ** 2))
        P1u = P1u + np.reshape(SH(np.reshape(f, (n, n)))[0], (L * n * n)) - P1z
        P2u = P2u + f - P2z

        f_norms.append(np.linalg.norm(f, ord=2))
        print(f_norms)
        iterations += 1

    print('Finished custom shearlet admm solver in ', toc())
    plt.plot(f_norms)
    plt.xlabel("number of iterations")
    plt.ylabel("L2 norm of solution")
    plt.show()
    return np.reshape(f, (50, 50))

# ========== shapes of relevant objects ==========

# SH: L n^2 x n^2
# SH(f): L n x n
# Rphi: m x n^2
# Rphi(x): m
# y: m (eta: m)
# f: n^2
# z: R ^ (L+1)n^2
# u: R ^ (L+1)n^2

# J <-> L

# note that I've used here J (decomposition subbands) and L from shearlet_transform_algorithm.py interchangeably,
# they're probably not identically the same


# ========== deprecated stuff ==========

# SHt =  # I think the inverse is implied here, not entirely sure how the transpose and inverse are connected
