from matplotlib import pyplot as plt
from skimage import color
from skimage import io
from scipy import misc

from shearlet_transform import calculateSpectra, applyShearletTransform, applyInverseShearletTransform


def shearlet_demo(imagePath):  # expecting image of shape MxN
    # image = color.rgb2gray(io.imread(imagePath))
    image = misc.face(gray=True)
    plt.imshow(image, cmap='gray')
    plt.title('ground truth')
    plt.colorbar()
    plt.show()

    # takes about 36 seconds to run a 511x511 image with default arguments
    spectra = calculateSpectra(image.shape[0], image.shape[1])

    shearletCoeffs, _ = applyShearletTransform(image, spectra=spectra)

    # takes about <1 second to run a 511x511 image with default arguments
    reconstruction = applyInverseShearletTransform(shearletCoeffs, spectra=spectra)

    # takes about 36 seconds to run a 511x511 image with default arguments
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


shearlet_demo('../slice_511.jpg')
