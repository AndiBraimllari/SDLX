import numpy as np
from scipy.sparse import identity
from scipy.sparse.linalg import cg
from matplotlib import pyplot as plt
from skimage import color
from skimage import io
from ttictoc import tic, toc
from scipy import misc
from skimage.transform import radon, rescale
from shearlet_transform.shearlet_transform import applyShearletTransform as SH, applyInverseShearletTransform as SHt


# ========== definitions start here ==========
def run_sanity_check(image):
    if image.shape[0] != image.shape[1]:
        raise ValueError('We currently only support square images')


def max(a, b):  # element wise max
    if a.shape[0] != b.shape[0]:
        raise ValueError('Unmatching dimensions in inputs of element wise max')
    res = np.zeros(a.shape[0])
    for i in range(a.shape[0]):
        res[i] = max(a[i], b[i])
    return res


def shrink(a, b):  # element wise shrink
    if a.shape[0] != b.shape[0]:
        raise ValueError('Unmatching dimensions in inputs of element wise shrink')
    res = np.zeros(a.shape[0])
    for i in range(a.shape[0]):
        res[i] = max(abs(a[i]) - b[i], 0) * (a[i] / abs(a[i])) if a[i] != 0 else 0
    return res


# NB eta and J are used interchangeably
def calc_J_from(n):
    return 59
    # TODO add logic


# TODO is 30 a good default value here? what value makes sense?
def generate_R_y(image, phi=30):
    theta = np.linspace(0., 180., max(image.shape), endpoint=False)  # TODO use phi here
    sinogram = radon(image, theta=theta)
    # TODO sinogram is y? R projector?
    return sinogram, np.array([])
    # TODO note that based on this current code, sinogram.shape == image.shape


# ===== shearlet admm starts here =====
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


def admm_demo(imagePath):
    image = color.rgb2gray(io.imread(imagePath))  # R ^ M x N, but actually we only currently accept R ^ n x n
    run_sanity_check(image)

    # TODO generate R_phi and y from image? check example2d.cpp
    #  AFAIK radon package in MATLAB offers something similar, equivalent for Python is?
    R_phi, y = generate_R_y(image)

    # EVERYTHING(except R_phi and y) is a hyper-parameter
    # f = shearlet_admm(max_iter=50, rho_zero=, rho_one=, rho_two=1, w=, R_phi=R_phi, y=y)


admm_demo('slice_511.jpg')

# ========== shapes of relevant objects ==========

# SH: J n^2 x n^2
# SH(f): J n x n
# Rphi: m x n^2
# y: m
# f: n^2
# eta: m # this eta is unrelated to the eta in shearlet_transform.py

# note that I've used here J (decomposition subbands) and eta from shearlet_transform.py interchangeably, they're
# probably not identically the same


# ========== deprecated stuff ==========

# SHt =  # I think the inverse is implied here, not entirely sure how the transpose and inverse are connected
