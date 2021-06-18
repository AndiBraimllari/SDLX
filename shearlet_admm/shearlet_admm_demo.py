from matplotlib import pyplot as plt
from skimage import color
from skimage import io
import numpy as np

from shearlet_admm_algorithm import shearlet_admm
from shearlet_admm_definitions import run_sanity_check, generate_R_y


def admm_demo(imagePath):
    image = color.rgb2gray(io.imread(imagePath))  # R ^ M x N, but actually we only currently accept R ^ n x n
    run_sanity_check(image)

    # TODO generate R_phi and y from image? check example2d.cpp
    #  AFAIK radon package in MATLAB offers something similar, equivalent for Python is?
    R_phi, y = generate_R_y(image)

    # EVERYTHING (except R_phi and y) is a hyper-parameter
    # f = shearlet_admm(rho_zero=, rho_one=, rho_two=1, w=, R_phi=R_phi, y=y)
    f = shearlet_admm(max_iter=50, rho_zero=1, rho_one=1, rho_two=1, w=np.ones(13 * 50 * 50), R_phi=R_phi, y=y)
    plt.imshow(np.reshape(f, (50, 50)))
    plt.show()


admm_demo('../slice_511.jpg')
