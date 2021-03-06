import numpy as np
from skimage.transform import radon


def run_sanity_check(image):
    if image.shape[0] != image.shape[1]:
        raise ValueError('We currently only support square images')


def max(a, b):  # element wise max
    if a.shape[0] != b.shape[0]:
        raise ValueError('Non-matching dimensions in inputs of element wise max')
    res = np.zeros(a.shape[0])
    for i in range(a.shape[0]):
        res[i] = max(a[i], b[i])
    return res


def shrink(a, b):  # element wise shrink (soft-thresholding)
    if a.shape[0] != b.shape[0]:
        raise ValueError('Non-matching dimensions in inputs of element wise shrink')
    res = np.zeros(a.shape[0])
    for i in range(a.shape[0]):
        res[i] = np.max(abs(a[i]) - b[i], 0) * (a[i] / abs(a[i])) if a[i] != 0 else 0
    return res


# TODO is 30 a good default value here? what value makes sense?
def generate_R_y(image, phi=30):
    # TODO note that based on this following code, sinogram.shape == image.shape
    # theta = np.linspace(0., 180., max(image.shape), endpoint=False)  # TODO use phi here
    # sinogram = radon(image, theta=theta)
    # TODO sinogram is y? R projector?
    return np.random.uniform(size=(20, 50 * 50)), np.random.uniform(size=20)
