import numpy as np
from matplotlib import pyplot as plt
from skimage import color
from skimage import io
from ttictoc import tic, toc
from scipy import misc


# ========== definitions start here ==========
def v(x):
    if x < 0:
        return 0
    elif 0 <= x <= 1:
        return 35 * x ** 4 - 84 * x ** 5 + 70 * x ** 6 - 20 * x ** 7
    else:
        return 1


def b(w):  # R -> R
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


def phiHat(w1, w2):  # R^2 -> R
    if abs(w1) <= 1 / 2 and abs(w2) <= 1 / 2:
        return 1
    elif 1 / 2 < abs(w1) < 1 and abs(w2) <= abs(w1):
        return np.cos(np.pi / 2 * v(2 * abs(w1) - 1))
    elif 1 / 2 < abs(w2) < 1 and abs(w1) < abs(w2):
        return np.cos(np.pi / 2 * v(2 * abs(w2) - 1))
    else:
        return 0


def psiHat1(w):  # R -> R
    return np.sqrt(b(2 * w) ** 2 + b(w) ** 2)


def psiHat2(w):  # R -> R
    if w <= 0:
        return np.sqrt(v(1 + w))
    else:
        return np.sqrt(v(1 - w))


def psiHat(w1, w2):  # separable generating function?
    # TODO how is this function defined on w1 := 0? maybe psiHat1(w1) * psiHat2(w2) ? if psiHat1(w1) returns 0 then ???
    if w1 == 0:
        return 0
    return psiHat1(w1) * psiHat2(w2 / w1)


# ===== shearlet transforms starts here =====
def applyShearletTransform(img):
    tic()
    M = img.shape[0]
    N = img.shape[1]

    print('Shape of the input image in the shearlet transform is:', img.shape)

    # currently the values of the image vary from [0, 1]
    # img /= M # TODO should I rescale the image? might be required for the low-frequency part

    spectra = calculateSpectra(M, N)

    fftImg = np.fft.fft2(img)
    SHf = np.fft.ifft2(spectra * fftImg)

    print('Finished shearlet transform in: ', toc())
    return SHf, spectra


def applyInverseShearletTransform(SHf, spectra=None, real=True):
    tic()
    # here I'm assuming SHf is of shape (eta, M, N)
    M = SHf.shape[1]
    N = SHf.shape[2]

    if spectra is None:
        spectra = calculateSpectra(M, N)

    print('Finished inverse shearlet transform in: ', toc())  # AFAIK SH is orthogonal, therefore inverse == transpose
    if real:
        return np.real(np.sum(np.fft.ifft2(np.fft.fft2(SHf) * spectra), axis=0))
    else:
        return np.sum(np.fft.ifft2(np.fft.fft2(SHf) * spectra), axis=0)


def calculateSpectra(M, N, eta=None):
    jZero = int(np.floor(1 / 2 * np.log2(max(M, N))))
    if eta is None:
        eta = 2 ** (jZero + 2) - 3

    spectra = np.zeros([eta, M, N])

    i = 0

    tempSHtcSectionZero = np.zeros([M, N])
    for w1 in range(int(-np.floor(M / 2)), int(np.ceil(M / 2))):
        for w2 in range(int(-np.floor(N / 2)), int(np.ceil(N / 2))):
            tempSHtcSectionZero[w1, w2] = phiHat(w1, w2)
    spectra[i] = tempSHtcSectionZero
    i += 1

    for j in range(jZero):
        for k in range(-2 ** j, 2 ** j + 1):
            tempSHSectionh = np.zeros([M, N])
            tempSHSectionv = np.zeros([M, N])
            tempSHSectionhxv = np.zeros([M, N])
            for w1 in range(int(-np.floor(M / 2)), int(np.ceil(M / 2))):
                for w2 in range(int(-np.floor(N / 2)), int(np.ceil(N / 2))):
                    horiz = 0
                    vertic = 0
                    if abs(w2) <= abs(w1):
                        horiz = psiHat(4 ** (-j) * w1, 4 ** (-j) * k * w1 + 2 ** (-j) * w2)
                    else:
                        vertic = psiHat(4 ** (-j) * w2, 4 ** (-j) * k * w2 + 2 ** (-j) * w1)
                    if abs(k) <= 2 ** j - 1:
                        tempSHSectionh[w1, w2] = horiz
                        tempSHSectionv[w1, w2] = vertic
                    elif abs(k) == 2 ** j:
                        tempSHSectionhxv[w1, w2] = horiz + vertic
            if abs(k) <= 2 ** j - 1:
                spectra[i] = tempSHSectionh
                i += 1
                spectra[i] = tempSHSectionv
                i += 1
            elif abs(k) == 2 ** j:
                spectra[i] = tempSHSectionhxv
                i += 1

    return spectra


# ========== demo starts here ==========
def shearlet_demo(imagePath):
    image = color.rgb2gray(io.imread(imagePath))  # R ^ M x N
    # image = misc.face(gray=True)
    plt.imshow(image, cmap='gray')
    plt.colorbar()
    plt.show()

    # takes about 36 seconds to run
    shearletCoeffs, spectra = applyShearletTransform(image)

    # # TODO this returns imaginary numbers, debug, fftshift?
    # takes about <1 second to run
    reconstruction = applyInverseShearletTransform(shearletCoeffs, spectra=spectra)
    # takes about 36 seconds to run
    # reconstruction = applyInverseShearletTransform(shearletCoeffs)

    plt.imshow(reconstruction, cmap='gray')
    plt.colorbar()
    plt.show()


shearlet_demo('slice_511.jpg')
