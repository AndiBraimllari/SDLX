import numpy as np
from matplotlib import pyplot as plt
from skimage import color
from skimage import io
from ttictoc import tic, toc
from scipy import misc


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


def applyShearletTransform(img, spectra=None):
    """
    Calculates the Cone-Adapted discrete shearlet transform of a given image.

    Parameters:
    img (numpy.ndarray): Image of shape (M, N).

    Returns:
    numpy.ndarray: 3D object of shape (eta, M, N) containing its calculated shearlet transform.
   """
    tic()
    M = img.shape[0]
    N = img.shape[1]

    print('Shape of the input image in the shearlet transform is:', img.shape)

    if spectra is None:
        spectra = calculateSpectra(M, N)

    fftImg = np.fft.fft2(img)
    SHf = np.fft.ifft2(spectra * fftImg)

    print('Finished shearlet transform in: ', toc())
    return SHf, spectra


def applyInverseShearletTransform(SHf, spectra=None, real=True):
    """
    Calculates the Cone-Adapted discrete shearlet inverse transform of a given image.

    Parameters:
    SHf (numpy.ndarray): Shearlet coefficients of shape (eta, M, N)
    spectra (numpy.ndarray): Shearlet spectra of shape (eta, M, N). Providing this object avoids its recalculation and
    drastically increases performance.

    Returns:
    numpy.ndarray: 3D object of shape (eta, M, N) containing its calculated shearlet transform
   """
    tic()
    M = SHf.shape[1]
    N = SHf.shape[2]

    if spectra is None:
        spectra = calculateSpectra(M, N)

    print('Finished inverse shearlet transform in: ', toc())  # AFAIK SH is orthogonal, therefore inverse == transpose
    if real:
        return np.real(np.sum(np.fft.ifft2(np.fft.fft2(SHf) * spectra), axis=0))
    else:
        return np.sum(np.fft.ifft2(np.fft.fft2(SHf) * spectra), axis=0)


def calculateSpectra(M, N, a=lambda j: 2 ** (-2 * j), s=lambda j, k: k * 2 ** (-j), jZero=None):
    """
    Calculates the spectra for a given shape. One can also specify the parabolic scaling (dilation) and shearing, as
    well as the number scales

    Parameters:
    M (int): Width
    N (int): Length
    a (lambda): The parabolic scaling (dilation) parameter.
    s (lambda): The shearing parameter
    jZero (int): Number of scales

    The parameters a and s are currently lambdas that have a set number of inputs, ideally should take any

    Returns:
    numpy.ndarray: 3D object of shape (eta, M, N) containing the calculated spectra
   """
    print('Shape required for constructing this spectra is:({}, {})'.format(M, N))

    if jZero is None:
        jZero = int(np.floor(1 / 2 * np.log2(max(M, N))))

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
                        horiz = psiHat(a(j) * w1, np.sqrt(a(j)) * s(j, k) * w1 + np.sqrt(a(j)) * w2)
                    else:
                        vertic = psiHat(a(j) * w2, np.sqrt(a(j)) * s(j, k) * w2 + np.sqrt(a(j)) * w1)
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


def shearlet_demo(imagePath):
    image = color.rgb2gray(io.imread(imagePath))  # R ^ M x N
    # image = misc.face(gray=True)
    plt.imshow(image, cmap='gray')
    plt.title('ground truth')
    plt.colorbar()
    plt.show()

    # takes about 36 seconds to run
    spectra = calculateSpectra(image.shape[0], image.shape[1])

    shearletCoeffs, _ = applyShearletTransform(image, spectra=spectra)

    # takes about <1 second to run
    reconstruction = applyInverseShearletTransform(shearletCoeffs, spectra=spectra)

    # takes about 36 seconds to run
    # reconstruction = applyInverseShearletTransform(shearletCoeffs)

    plt.imshow(reconstruction, cmap='gray')
    plt.title('recon.')
    plt.colorbar()
    plt.show()

    reconGtDiff = reconstruction - image
    plt.imshow(reconGtDiff, cmap='gray')
    plt.title('diff.')
    plt.colorbar()
    plt.show()

    # fig, axes = plt.subplots(1, 3)
    #
    # axes[0].imshow(image, cmap='gray')
    # axes[0].set_axis_off()
    # axes[0].set_title('gt')
    #
    # axes[1].imshow(reconstruction, cmap='gray')
    # axes[1].set_axis_off()
    # axes[1].set_title('recon.')
    #
    # axes[2].imshow(reconGtDiff, cmap='gray')
    # axes[2].set_axis_off()
    # axes[2].set_title('diff (sum of squares {})'.format(np.sum(np.square(np.concatenate(image - reconstruction)))))
    #
    # plt.show()


shearlet_demo('slice_511.jpg')
