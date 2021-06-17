import unittest
import numpy as np
from skimage import io
from skimage import color
from numpy.testing import *
from scipy import misc

from shearlet_transform import calculateSpectra, applyShearletTransform, applyInverseShearletTransform


class TestShearletTransform(unittest.TestCase):
    """
    Test how close the reconstruction to the ground truth really is. Increasing the number of scales lowers the L2 norm
    of the difference between these two.
    """

    def test_recon_with_gt(self):
        image = misc.face(gray=True)

        SHf, spectra = applyShearletTransform(image, jZero=5)
        recon = applyInverseShearletTransform(SHf, spectra=spectra)
        squaredSumDiff = np.sum(np.square(np.concatenate(image - recon)))

        self.assertIsInstance(squaredSumDiff, float)
        self.assertAlmostEqual(squaredSumDiff, 0)

    def test_parseval_frame(self):
        """
        Given that our spectra constitute a Parseval frame, if a matrix mxn A has rows then AtA = I (Corollary 1.4.7
        from An Introduction to Frames and Riesz Bases). This ensures that the spectra we generated exhibits desired
        behaviour.
        """
        image = misc.face(gray=True)

        spectra = calculateSpectra(image.shape[0], image.shape[1], jZero=5)

        frameCorrectness = np.sum(np.square(spectra), axis=0) - 1

        assert_array_almost_equal(frameCorrectness, np.zeros_like(frameCorrectness))


if __name__ == '__main__':
    unittest.main()
