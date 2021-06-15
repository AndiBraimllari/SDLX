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


# phiHat is equivalent to
# if abs(w2) <= abs(w1):
#     return phi(w1)
# else:
#     return phi(w2)


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

    # currently the values of the image vary from [0, 1]
    # img /= M # TODO should I rescale the image? might be required for the low-frequency part

    jZero = np.floor(1 / 2 * np.log2(max(M, N))).astype('int')
    eta = 2 ** (jZero + 2) - 3

    SHf = np.zeros([eta, M, N], dtype=complex)

    fftImg = np.fft.fft2(img)
    print('Shape of the input image in the shearlet transform is:', fftImg.shape)

    # sampling from is [-255, 255] x [-255, 255]
    i = 0
    for j in range(jZero):
        for k in range(-2 ** j, 2 ** j + 1):
            tempSHSectionh = np.zeros([M, N], dtype=complex)
            tempSHSectionv = np.zeros([M, N], dtype=complex)
            tempSHSectionhxv = np.zeros([M, N], dtype=complex)
            # [-floor(M / 2), ..., ceil(M / 2) - 1]
            for w1 in range(-np.floor(M / 2).astype('int'), np.ceil(M / 2).astype('int')):
                # [-floor(N / 2), ..., ceil(N / 2) - 1]
                for w2 in range(-np.floor(N / 2).astype('int'), np.ceil(N / 2).astype('int')):
                    horiz = 0
                    vertic = 0
                    if abs(w2) <= abs(w1):
                        horiz = psiHat(4 ** (-j) * w1, 4 ** (-j) * k * w1 + 2 ** (-j) * w2)
                    else:
                        vertic = psiHat(4 ** (-j) * w2, 4 ** (-j) * k * w2 + 2 ** (-j) * w1)
                    if abs(k) <= 2 ** j - 1:
                        # section of horizontal cone
                        tempSHSectionh[w1, w2] = horiz * fftImg[w1][w2]
                        # section of vertical cone
                        tempSHSectionv[w1, w2] = vertic * fftImg[w1][w2]
                    elif abs(k) == 2 ** j:
                        # section of the seam lines
                        tempSHSectionhxv[w1, w2] = (horiz + vertic) * fftImg[w1][w2]
            if abs(k) <= 2 ** j - 1:
                SHSectionh = np.fft.ifft2(tempSHSectionh)
                if SHSectionh.all() != np.real(SHSectionh).all():
                    raise ValueError('Unexpected state! Non-zero imaginary parts found in SHSectionh!')
                SHf[i] = SHSectionh
                i += 1

                SHSectionv = np.fft.ifft2(tempSHSectionv)
                if SHSectionv.all() != np.real(SHSectionv).all():
                    raise ValueError('Unexpected state! Non-zero imaginary parts found in SHSectionv!')
                SHf[i] = SHSectionv
                i += 1
            elif abs(k) == 2 ** j:
                SHSectionhxv = np.fft.ifft2(tempSHSectionhxv)
                if SHSectionhxv.all() != np.real(SHSectionhxv).all():
                    raise ValueError('Unexpected state! Non-zero imaginary parts found in SHSectionhxv!')
                SHf[i] = SHSectionhxv
                i += 1
                # plt.imshow(np.real(SHSectionhxv), cmap='gray')  # cmap='gray'  # np.imag
                # plt.show()

    # TODO current problem, phiHat returns non-zero for input around [-1, 1] but our w1 and w2 vary from [-255, 255]
    #  is it okay that most of the low-frequency output is 0? or should I scale w1 an w2?
    # section of the low frequency
    tempSHSectionZero = np.zeros([M, N], dtype=complex)

    # NB currently only [0, 0] outputs non-zero value
    for w1 in range(-np.floor(M / 2).astype('int'), np.ceil(M / 2).astype('int')):
        for w2 in range(-np.floor(N / 2).astype('int'), np.ceil(N / 2).astype('int')):
            tempSHSectionZero[w1, w2] = phiHat(w1, w2) * fftImg[w1, w2]

    SHSectionZero = np.fft.ifft2(tempSHSectionZero)
    SHf[i] = SHSectionZero

    print('Finished shearlet transform in', toc())
    return SHf


def applyInverseShearletTransform(SHf):
    tic()
    # here I'm assuming SHf is of shape (eta, M, N)
    M = SHf.shape[1]
    N = SHf.shape[2]
    jZero = np.floor(1 / 2 * np.log2(max(M, N))).astype('int')

    fHat = np.zeros([M, N], dtype=complex)

    # sampling from is [-255, 255] x [-255, 255]
    i = 0
    for j in range(jZero):
        for k in range(-2 ** j, 2 ** j + 1):
            tempSHSectionh = np.zeros([M, N])
            tempSHSectionv = np.zeros([M, N])
            tempSHSectionhxv = np.zeros([M, N])
            for w1 in range(-np.floor(M / 2).astype('int'), np.ceil(M / 2).astype('int')):
                # [-floor(N / 2), ..., ceil(N / 2) - 1]
                for w2 in range(-np.floor(N / 2).astype('int'), np.ceil(N / 2).astype('int')):
                    horiz = 0
                    vertic = 0
                    if abs(w2) <= abs(w1):
                        horiz = psiHat(4 ** (-j) * w1, 4 ** (-j) * k * w1 + 2 ** (-j) * w2)
                    else:
                        vertic = psiHat(4 ** (-j) * w2, 4 ** (-j) * k * w2 + 2 ** (-j) * w1)
                    if abs(k) <= 2 ** j - 1:
                        # section of horizontal cone
                        tempSHSectionh[w1, w2] = horiz
                        # section of vertical cone
                        tempSHSectionv[w1, w2] = vertic
                    elif abs(k) == 2 ** j:
                        # section of the seam lines
                        tempSHSectionhxv[w1, w2] = horiz + vertic
            if abs(k) <= 2 ** j - 1:
                SHtSectionh = np.multiply(np.fft.fft2(SHf[i]), tempSHSectionh)
                i += 1
                SHtSectionv = np.multiply(np.fft.fft2(SHf[i]), tempSHSectionv)
                i += 1
                fHat += SHtSectionh + SHtSectionv
            elif abs(k) == 2 ** j:
                SHtSectionhxv = np.multiply(np.fft.fft2(SHf[i]), tempSHSectionhxv)
                i += 1
                fHat += SHtSectionhxv

            # if fHat.all() != np.real(fHat).all():  # TODO do I need to do some check here for fHat?
            #     print(?)

    # section of the low frequency
    tempSHtcSectionZero = np.zeros([M, N])

    for w1 in range(-np.floor(M / 2).astype('int'), np.ceil(M / 2).astype('int')):
        for w2 in range(-np.floor(N / 2).astype('int'), np.ceil(N / 2).astype('int')):
            tempSHtcSectionZero[w1, w2] = phiHat(w1, w2)

    SHtSectionZero = np.multiply(np.fft.fft2(SHf[i]), tempSHtcSectionZero)
    fHat += SHtSectionZero

    print('Finished inverse shearlet transform in', toc())  # AFAIK SH is orthogonal, therefore inverse == transpose
    return np.fft.ifft2(fHat)


# ========== demo starts here ==========
def shearlet_demo(imagePath):
    image = color.rgb2gray(io.imread(imagePath))  # R ^ M x N
    # image = misc.face(gray=True)
    # NB takes about 86 seconds to run
    shearletCoeffs = applyShearletTransform(image)

    # # TODO this returns imaginary numbers, debug, fftshift?
    # NB takes about 26 seconds to run
    reconstruction = applyInverseShearletTransform(shearletCoeffs)

    plt.imshow(np.real(reconstruction), cmap='gray')
    plt.show()


shearlet_demo('slice_511.jpg')
