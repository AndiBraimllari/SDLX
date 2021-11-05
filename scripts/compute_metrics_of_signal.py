import pyelsa as elsa

import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity
import numpy as np

# code from http://www.math.uni-bremen.de/cda/HaarPSI/software/haarPsi.py
from scripts.haarpsi import haar_psi


def relative_error(ground_truth, reconstruction):
    return np.sum(np.power((ground_truth - reconstruction), 2)) / np.sum(np.power(ground_truth, 2))


def compute_metrics(gt, recon, data_range=255):
    if gt.shape != recon.shape:
        raise ValueError('different shapes of signals!')

    a = relative_error(gt, recon)
    print('Relative error (the lower the better): [' + str(a) + '] for a signal of shape: [' + str(
        recon.shape) + ']')

    b = peak_signal_noise_ratio(gt, recon, data_range=data_range)
    print('Peak signal-to-noise ratio error (the higher the better): [' + str(
        b) + '] for a signal of shape: [' + str(recon.shape) + ']')

    c = structural_similarity(gt, recon)
    print('Structural similarity (the higher the better): [' + str(
        c) + '] for a signal of shape: [' + str(recon.shape) + ']')

    d = haar_psi(gt, recon)  # note that haar_psi returns 3 elements, the first is the metric
    print('Haar wavelet-based perceptual similarity index error (the higher the better): [' + str(
        d[0]) + '] for a signal of shape: [' + str(recon.shape) + ']')


def read_edf(signal_path):
    signal = elsa.EDF.readf(signal_path)
    signal = np.array(signal)

    return signal


def read_image(signal_path):
    signal = plt.imread(signal_path)

    return signal


ground_truth = None
reconstruction = None

compute_metrics(ground_truth, reconstruction)
