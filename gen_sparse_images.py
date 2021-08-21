import numpy as np
import matplotlib.pyplot as plt
from skimage import color
from skimage import io
import os
from tqdm import tqdm

import pyelsa as elsa


def generate_sparse_ct_images(src_dir, out_dir, num_angles=50, limit=None):
    """
    Generate sparsely sampled images from one directory to the other.
    """
    if not os.path.exists(src_dir):
        raise ValueError('The provided src_dir directory does not exist')

    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    paths = os.listdir(src_dir)

    if limit is not None:
        paths = paths[:limit]

    for file_name in tqdm(paths):
        image = elsa.DataContainer(color.rgb2gray(io.imread(src_dir + '/' + file_name)))
        size = np.array([image.getDataDescriptor().getNumberOfCoefficientsPerDimension()[0],
                         image.getDataDescriptor().getNumberOfCoefficientsPerDimension()[1]])

        volume_descriptor = image.getDataDescriptor()

        # generate circular trajectory
        arc = 360
        sino_descriptor = elsa.CircleTrajectoryGenerator.createTrajectory(num_angles, volume_descriptor, arc,
                                                                          size[0] * 100, size[0])

        # setup operator for 2D X-ray transform
        projector = elsa.SiddonsMethod(volume_descriptor, sino_descriptor)

        # simulate the sinogram
        sinogram = projector.apply(image)

        # setup reconstruction problem
        wls_problem = elsa.WLSProblem(projector, sinogram)

        # solve the reconstruction problem
        cg_solver = elsa.CG(wls_problem)

        no_iterations = 20
        cg_reconstruction = cg_solver.solve(no_iterations)

        new_image_name = out_dir + '/cg_recon_i_' + str(no_iterations) + '_p_' + str(num_angles) + '_' + file_name
        plt.imsave(new_image_name, cg_reconstruction)
