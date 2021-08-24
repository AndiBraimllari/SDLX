import numpy as np
import os
from pathlib import Path
from tqdm import tqdm
from os.path import splitext

from shearlet_transform.shearlet_transform_algorithm import applyShearletTransform


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
        out_dir = parent_dir + '/shearleted_scales_' + str(scales)
        os.mkdir(out_dir)

    paths = os.listdir(src_dir)

    if limit is not None:
        paths = paths[:limit]

    for file_name in tqdm(paths):
        image = np.load(src_dir + '/' + file_name)

        sh_image, spectra = applyShearletTransform(image, jZero=scales)

        file_name_no_ex, extension = splitext(file_name)

        np.save(out_dir + '/' + file_name_no_ex, sh_image)
