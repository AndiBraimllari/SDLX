import numpy as np
import os
from pathlib import Path
from tqdm import tqdm
from os.path import splitext
import sys

# add the parent of the parent of this file to the system path
sys.path.append(str(Path(os.path.abspath(Path(__file__).parent)).parent))

from shearlet_transform.shearlet_transform_algorithm import applyShearletTransform, calculateSpectra


def generate_shearleted_npy_images(src_dir, out_dir=None, scales=None, limit=None):
    """
    Generate shearleted images from one directory to the other. These images are contained in NumPy
    files.
    """
    if not os.path.exists(src_dir):
        raise ValueError('The provided src_dir directory does not exist')

    if out_dir is not None and not os.path.exists(out_dir):
        os.mkdir(out_dir)

    if out_dir is None:
        parent_dir = str(Path(src_dir).parent.absolute())
        if scales is None:
            out_dir = parent_dir + '/shearleted_'
        else:
            out_dir = parent_dir + '/shearleted_scales_' + str(scales)
        os.mkdir(out_dir)

    paths = os.listdir(src_dir)

    if limit is not None:
        paths = paths[:limit]

    first_image = np.load(src_dir + '/' + paths[0])
    init_width = first_image.shape[0]
    init_height = first_image.shape[1]
    spectra = calculateSpectra(init_width, init_height, jZero=scales)

    for file_name in tqdm(paths):
        image = np.load(src_dir + '/' + file_name)

        if image.shape[0] != init_width or image.shape[1] != init_height:
            raise ValueError('Encountered different shapes of images! We only precomputed for one shape, too bad!')

        sh_image, spectra = applyShearletTransform(image, spectra)

        file_name_no_ex, extension = splitext(file_name)

        np.save(out_dir + '/' + file_name_no_ex, sh_image)
