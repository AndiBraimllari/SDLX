import numpy as np
import os
from pathlib import Path
from tqdm import tqdm
from os.path import splitext


# from shearlet_transform.shearlet_transform_algorithm import applyShearletTransform


def generate_shearleted_npy_images(src_dir, out_dir=None, scales=None, limit=None):
    """
    Generate shearleted images from one directory to the other. These images are contained in NumPy
    files.
    """
    if not os.path.exists(src_dir):
        raise ValueError('The provided src_dir directory does not exist')

    if out_dir is not None and not os.path.exists(out_dir):
        os.mkdir(out_dir)

    if out_dir is None:
        parent_dir = str(Path(src_dir).parent.absolute())
        out_dir = parent_dir + '/shearleted_scales_' + str(scales)
        os.mkdir(out_dir)

    paths = os.listdir(src_dir)

    if limit is not None:
        paths = paths[:limit]

    for file_name in tqdm(paths):
        image = np.load(src_dir + '/' + file_name)

        sh_image, spectra = applyShearletTransform(image, jZero=scales)

        file_name_no_ex, extension = splitext(file_name)

        np.save(out_dir + '/' + file_name_no_ex, sh_image)


##### REMOVE ME AND REPLACE WITH THE CORRECT IMPORT #####

import numpy as np
from ttictoc import tic, toc


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


def applyShearletTransform(img, spectra=None, jZero=None):
    """
    Calculates the Cone-Adapted discrete shearlet transform of a given image.

    Parameters:
    img (numpy.ndarray): Image of shape (W, H).
    spectra (numpy.ndarray): Shearlet spectra of shape (L, W, H). Providing this object avoids its recalculation and
    drastically increases performance.
    jZero (int): Number of scales.

    Returns:
    numpy.ndarray: 3D object of shape (L, W, H) containing its calculated shearlet transform.
   """
    tic()
    W = img.shape[0]
    H = img.shape[1]

    print('Shape of the input image in the shearlet transform is:', img.shape)

    if spectra is None:
        spectra = calculateSpectra(W, H, jZero=jZero)

    fftImg = np.fft.fft2(img)
    SHf = np.real(np.fft.ifft2(spectra * fftImg))

    print('Finished shearlet transform in: ', toc())
    return SHf, spectra


def applyInverseShearletTransform(SHf, spectra=None, real=True):
    """
    Calculates the Cone-Adapted discrete shearlet inverse transform of a given image.

    Parameters:
    SHf (numpy.ndarray): Shearlet coefficients of shape (L, W, H)
    spectra (numpy.ndarray): Shearlet spectra of shape (L, W, H). Providing this object avoids its recalculation and
    drastically increases performance.
    real (bool):

    Returns:
    numpy.ndarray: 3D object of shape (L, M, N) containing its calculated shearlet transform
   """
    tic()
    W = SHf.shape[1]
    H = SHf.shape[2]

    if spectra is None:
        spectra = calculateSpectra(W, H)

    print('Finished inverse shearlet transform in: ', toc())  # AFAIK SH is orthogonal, therefore inverse == transpose
    if real:
        return np.real(np.sum(np.fft.ifft2(np.fft.fft2(SHf) * spectra), axis=0))
    else:
        return np.sum(np.fft.ifft2(np.fft.fft2(SHf) * spectra), axis=0)


def calculateSpectra(W, H, a=lambda j: 2 ** (-2 * j), s=lambda j, k: k * 2 ** (-j), jZero=None):
    """
    Calculates the spectra for a given shape. One can also specify the parabolic scaling (dilation) and shearing, as
    well as the number scales

    Parameters:
    W (int): Width.
    H (int): Length.
    a (lambda): The parabolic scaling (dilation) parameter.
    s (lambda): The shearing parameter.
    jZero (int): Number of scales.

    The parameters a and s are currently lambdas that have a set number of inputs, ideally should take any

    Returns:
    numpy.ndarray: 3D object of shape (L, W, H) containing the calculated spectra
    """
    print('Shape required for constructing this spectra is:({}, {})'.format(W, H))

    if jZero is None:
        jZero = int(np.floor(1 / 2 * np.log2(max(W, H))))

    L = calc_L_from_scales(jZero)
    spectra = np.zeros([L, W, H])

    i = 0

    sectionZero = np.zeros([W, H])
    for w1 in range(int(-np.floor(W / 2)), int(np.ceil(W / 2))):
        for w2 in range(int(-np.floor(H / 2)), int(np.ceil(H / 2))):
            sectionZero[w1, w2] = phiHat(w1, w2)
    spectra[i] = sectionZero
    i += 1

    for j in range(jZero):
        for k in range(-2 ** j, 2 ** j + 1):
            sectionh = np.zeros([W, H])
            sectionv = np.zeros([W, H])
            sectionhxv = np.zeros([W, H])
            for w1 in range(int(-np.floor(W / 2)), int(np.ceil(W / 2))):
                for w2 in range(int(-np.floor(H / 2)), int(np.ceil(H / 2))):
                    horiz = 0
                    vertic = 0
                    if abs(w2) <= abs(w1):
                        horiz = psiHat(a(j) * w1, np.sqrt(a(j)) * s(j, k) * w1 + np.sqrt(a(j)) * w2)
                    else:
                        vertic = psiHat(a(j) * w2, np.sqrt(a(j)) * s(j, k) * w2 + np.sqrt(a(j)) * w1)
                    if abs(k) <= 2 ** j - 1:
                        sectionh[w1, w2] = horiz
                        sectionv[w1, w2] = vertic
                    elif abs(k) == 2 ** j:
                        sectionhxv[w1, w2] = horiz + vertic
            if abs(k) <= 2 ** j - 1:
                spectra[i] = sectionh
                i += 1
                spectra[i] = sectionv
                i += 1
            elif abs(k) == 2 ** j:
                spectra[i] = sectionhxv
                i += 1

    return spectra


def calc_L_from_scales(jZero):
    return 2 ** (jZero + 2) - 3


def calc_L_from_shape(W, H=0):
    jZero = int(np.floor(1 / 2 * np.log2(max(W, H))))
    return 2 ** (jZero + 2) - 3
