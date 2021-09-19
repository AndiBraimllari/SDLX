import pyelsa as elsa

from skimage import color, io
import numpy as np
from os.path import splitext


def image_to_edf(image_path):
    image = color.rgb2gray(io.imread(image_path))

    image = np.array(image)

    image_path_wo_suffix, extension = splitext(image_path)

    elsa.EDF.write(elsa.DataContainer(image), image_path_wo_suffix + '.edf')


image_to_edf('/home/andibraimllari/Desktop/playground/elsa/cmake-build-debug/slice_511.jpg')
