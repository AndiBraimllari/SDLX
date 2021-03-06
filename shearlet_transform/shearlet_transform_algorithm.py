import numpy as np
from ttictoc import tic, toc

from shearlet_transform.shearlet_definitions import psiHat, phiHat


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
    SHf = np.fft.ifft2(spectra * fftImg)

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
