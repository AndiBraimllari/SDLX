import pyelsa as elsa

from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity
import numpy as np

# code from http://www.math.uni-bremen.de/cda/HaarPSI/software/haarPsi.py
from scripts.haarpsi import haar_psi


def relative_error(ground_truth, reconstruction):
    return np.sum(np.power((ground_truth - reconstruction), 2)) / np.sum(np.power(ground_truth, 2))


def compute_metrics(ground_truth, reconstruction):
    if ground_truth.shape != reconstruction.shape:
        raise ValueError('different shapes of signals!')

    a = relative_error(ground_truth, reconstruction)
    print('Relative error (the lower the better): [' + str(a) + '] for a signal of shape: [' + str(
        reconstruction.shape) + ']')

    b = peak_signal_noise_ratio(ground_truth, reconstruction, data_range=1)  # TODO correct data_range
    print('Peak signal-to-noise ratio error (the higher the better): [' + str(
        b) + '] for a signal of shape: [' + str(reconstruction.shape) + ']')

    c = structural_similarity(ground_truth, reconstruction)
    print('Structural similarity (the higher the better): [' + str(
        c) + '] for a signal of shape: [' + str(reconstruction.shape) + ']')

    d = haar_psi(ground_truth, reconstruction)  # note that haar_psi returns 3 elements, the first is the metric
    print('Haar wavelet-based perceptual similarity index error (the higher the better): [' + str(
        d[0]) + '] for a signal of shape: [' + str(reconstruction.shape) + ']')


def read_edf(signal_path):
    signal = elsa.EDF.readf(signal_path)
    signal = np.array(signal)

    return signal


gt_path = None
recon_path = None

compute_metrics(read_edf(gt_path), read_edf(recon_path))
