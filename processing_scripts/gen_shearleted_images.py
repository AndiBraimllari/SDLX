import os
from os.path import splitext
import numpy as np
import matplotlib.pyplot as plt
from skimage import color
from skimage import io
from os.path import splitext

from shearlet_transform.shearlet_transform_algorithm import applyShearletTransform


def generate_shearleted_images(src_dir, out_dir, scales=50, limit=None):
    """
    Generate shearleted images from one directory to the other.
    """
    if not os.path.exists(src_dir):
        raise ValueError('The provided src_dir directory does not exist')

    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    paths = os.listdir(src_dir)

    if limit is not None:
        paths = paths[:limit]

    for file_name in tqdm(paths):
        image = color.rgb2gray(io.imread(src_dir + '/' + file_name))

        sh_image = applyShearletTransform(image)

        new_image_name = out_dir + '/shearleted_s_' + str(scales) + '_' + file_name
        file_name, extension = splitext(new_image_name)
        new_image_name = file_name + '.npy'

        plt.imsave(new_image_name, sh_image)
