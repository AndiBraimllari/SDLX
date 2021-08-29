from tqdm import tqdm
import numpy as np
import os
from pathlib import Path

# this script is to be run in an environment with the Python bindings of elsa installed
import pyelsa as elsa


def generate_sparse_npy_images(src_dir, out_dir=None, num_angles=50, no_iterations=20, limit=None):
    """
    Generate sparsely sampled images from one directory to the other, through elsa. These images are contained in NumPy
    files.
    """
    if not os.path.exists(src_dir):
        raise ValueError('The provided src_dir directory does not exist')

    if out_dir is not None and not os.path.exists(out_dir):
        os.mkdir(out_dir)

    if out_dir is None:
        parent_dir = str(Path(src_dir).parent.absolute())
        out_dir = parent_dir + '/cg_recon_iters_' + str(no_iterations) + '_poses_' + str(num_angles)
        os.mkdir(out_dir)

    paths = os.listdir(src_dir)

    if limit is not None:
        paths = paths[:limit]

    for file_name in tqdm(paths):
        image = elsa.DataContainer(np.load(src_dir + '/' + file_name))
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

        cg_reconstruction = cg_solver.solve(no_iterations)

        np.save(out_dir + '/' + file_name, cg_reconstruction)
