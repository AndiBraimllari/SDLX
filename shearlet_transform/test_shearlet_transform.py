import unittest
import numpy as np
from skimage import io
from skimage import color
from numpy.testing import *
from scipy import misc

from shearlet_transform import calculateSpectra, applyShearletTransform, applyInverseShearletTransform


class TestShearletTransform(unittest.TestCase):
    def test_recon_with_gt(self):
        """
        Test how close the reconstruction to the ground truth really is. Increasing the number of scales lowers the L2
        norm of the difference between these two.
        """
        image = misc.face(gray=True)

        SHf, spectra = applyShearletTransform(image, jZero=5)
        recon = applyInverseShearletTransform(SHf, spectra=spectra)
        squaredSumDiff = np.sum(np.square(np.concatenate(image - recon)))

        self.assertIsInstance(squaredSumDiff, float)
        self.assertAlmostEqual(squaredSumDiff, 0)

    def test_parseval_frame(self):
        """
        If a matrix mxn A has rows that constitute Parseval frame, then AtA = I (Corollary 1.4.7 from An Introduction to
        Frames and Riesz Bases). Given that our spectra constitute a Parseval frame, we can utilize this property to
        check if they've been generated correctly.
        """
        image = misc.face(gray=True)

        spectra = calculateSpectra(image.shape[0], image.shape[1], jZero=5)

        frameCorrectness = np.sum(np.square(spectra), axis=0) - 1

        assert_array_almost_equal(frameCorrectness, np.zeros_like(frameCorrectness))


if __name__ == '__main__':
    unittest.main()
