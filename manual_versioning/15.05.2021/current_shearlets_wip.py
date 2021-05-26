import numpy as np
import torch
from matplotlib import pyplot as plt
from skimage import color
from skimage import io


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


def phiHat(w1Param, w2Param):  # R^2 -> R
    if abs(w1Param) <= 1 / 2 and abs(w2Param) <= 1 / 2:
        return 1
    elif 1 / 2 < abs(w1Param) < 1 and abs(w2Param) < abs(w1Param):
        return np.cos(np.pi / 2 * v(2 * abs(w1Param) - 1))
    elif 1 / 2 < abs(w2Param) < 1 and abs(w1Param) < abs(w2Param):
        return np.cos(np.pi / 2 * v(2 * abs(w2Param) - 1))
    else:
        return 0


def psiHat1(w):  # R -> R
    return np.sqrt(b(2 * w) ** 2 + b(w) ** 2)


def psiHat2(w):  # R -> R
    if w <= 0:
        return np.sqrt(v(1 + w))
    else:
        return np.sqrt(v(1 - w))


def psiHat(w1Param, w2Param):  # separable generating function?
    # TODO how is this function defined on w1 := 0?
    if w1Param == 0:
        return 0
    return psiHat1(w1Param) * psiHat2(w2Param / w1Param)


# ===== shearlet transform starts here =====
def applyShearletTransform(img):
    M = img.shape[0]
    N = img.shape[1]

    jZero = np.floor(1 / 2 * np.log2(max(M, N))).astype('int')  # floor(1/2 log_2(max(M, N)))
    eta = 2 ** (jZero + 2) - 3

    SHf = np.zeros([M, N, eta])

    fftImg = np.fft.fft2(img)
    print(fftImg.shape)

    # TODO implement proper indexing, generate by j and k
    # for i in range(eta):
    #     pass

    # TODO NB KSI is [-255, 255] x [-255, 255]
    # TODO NB OMEGA is [-255, 255] x [-255, 255]

    # TODO kappa = 0
    tempSHKappaZero = np.zeros([M, N], dtype=complex)

    for w1 in range(-np.floor(M / 2).astype('int'), np.ceil(M / 2).astype('int')):
        for w2 in range(-np.floor(N / 2).astype('int'), np.ceil(N / 2).astype('int')):
            # print(phiHat(w1, w2) * fftImg[w1, w2])
            tempSHKappaZero[w1, w2] = phiHat(w1, w2) * fftImg[w1, w2]

    SHKappaZero = np.fft.ifft2(tempSHKappaZero)

    # plt.imshow(np.real(SHKappaZero))
    # plt.show()

    # 0, ..., j0 - 1
    for j in range(jZero):
        # -2^j + 1, ..., 2^j - 1  # TODO or is it -2^j, ..., 2^j?
        for k in range(-2 ** j + 1, 2 ** j):
            tempSHKappah = np.zeros([M, N], dtype=complex)
            tempSHKappav = np.zeros([M, N], dtype=complex)
            tempSHKappahxv = np.zeros([M, N], dtype=complex)
            # -floor(M / 2), ..., ceil(M / 2) - 1
            for w1 in range(-np.floor(M / 2).astype('int'), np.ceil(M / 2).astype('int')):
                # -floor(N / 2), ..., ceil(N / 2) - 1
                for w2 in range(-np.floor(N / 2).astype('int'), np.ceil(N / 2).astype('int')):
                    if abs(k) <= 2 ** j - 1:
                        # TODO kappa == h
                        tempSHKappah[w1, w2] = psiHat(4 ** (-j) * w1, 4 ** (-j) * k * w1 + 2 ** (-j) * w2) * \
                                               fftImg[w1][w2]
                        # TODO kappa == v
                        tempSHKappav[w1, w2] = psiHat(4 ** (-j) * w2, 4 ** (-j) * k * w2 + 2 ** (-j) * w1) * \
                                               fftImg[w1][w2]
                    elif abs(k) == 2 ** j:
                        # TODO kappa != 0
                        a = 1 / 0
                        print('here yo')
                        tempSHKappahxv[w1, w2] = psiHat(4 ** (-j) * w1, 4 ** (-j) * k * w1 + 2 ** (-j) * w2) * \
                                                 fftImg[w1][w2]
            SHKappah = np.fft.ifft2(tempSHKappah)
            SHKappav = np.fft.ifft2(tempSHKappav)
            SHKappahxv = np.fft.ifft2(tempSHKappahxv)

            # if SHKappah.all() == np.real(SHKappah).all():  # TODO check if all imaginary parts are 0 in all 3 of them
            #     print("equal obv")

            plt.imshow(np.real(SHKappahxv))  # np.imag
            plt.show()
    # plt.imshow(cs)
    # plt.show()

    print("done")


# for kappa = 0
# SHfl = np.fft.ifft2(phiHat(w1, w2) * fftImg(w1, w2))

# for kappa = h, |k| <= 2^j - 1
# SHfh = np.fft.ifft2(psiHat(4 ** (-j) * w1, 4 ** (-j) * k * w1 + 2 ** (-j) * w2) * fftImg(w1, w2))

# for kappa = h, |k| <= 2^j - 1
# SHfv = np.fft.ifft2(psiHat(4 ** (-j) * w2, 4 ** (-j) * k * w2 + 2 ** (-j) * w1) * fftImg(w1, w2))

# TODO different psi hat used here (psiHathxv)
# for kappa != 0, |k| = 2^j
# SHfhxv = np.fft.ifft2(psiHat(4 ** (-j) * w1, 4 ** (-j) * k * w1 + 2 ** (-j) * w2) * fftImg(w1, w2))

# ========== demo starts here ==========
img = color.rgb2gray(io.imread('grs_511.jpg'))  # R ^ M x N

applyShearletTransform(img)

# NB potential plan:
# fft2
# fftshift
# math here
# ifftshift
# ifft2


# NB torch can also be used:
# npFft = np.fft.fft([0, 1, 2, 3, 4])
# print(npFft)
# torchFft = torch.fft.fft2('grs_512.jpg')
# torchFft = torch.fft.fft(torch.IntTensor([0, 1, 2, 3, 4]))
# print(torchFft)

# NB different maps can be chosen
# plt.imshow(img, cmap='gray')
# plt.show()


# ========== deprecated helpers ==========

# # NB this is a quick indexing generator of [-255, 255] x [-255, 255]
# for i in range(-M // 2 + 1, M // 2 + 1):
#     for j in range(-N // 2 + 1, N // 2 + 1):

# # NB parabolic scaling matrix and shearing matrix, currently infused in the methods
# def Aa(aParam):  # aParam has to be positive
#     return np.array([
#         [aParam, 0],
#         [0, np.sqrt(aParam)]
#     ])
# def Ss(sParam):
#     return np.array([
#         [1, sParam],
#         [0, 1]
#     ])

# # NB negative indexing in Python may replace the need for this
# periodicImage = np.zeros([2 * M, 2 * N])
# for i in range(-M, M):
#     for j in range(-N, N):
#         # Q2
#         if i < 0 and j > 0:
#             periodicImage[i, j] = img[i + M, j]
#         # Q3
#         elif i < 0 and j < 0:
#             periodicImage[i, j] = img[i + M, j + N]
#         # Q4
#         elif i > 0 and j < 0:
#             periodicImage[i, j] = img[i, j + N]
#         # Q1
#         else:
#             periodicImage[i, j] = img[i, j]
