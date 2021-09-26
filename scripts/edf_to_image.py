import pyelsa as elsa

import matplotlib.pyplot as plt
import numpy as np
from os.path import splitext


def edf_to_image(edf_path):
    edf = elsa.EDF.readf(edf_path)

    image = np.array(edf)

    edf_path_wo_suffix, extension = splitext(edf_path)

    plt.imsave(edf_path_wo_suffix + '.png', image)


edf_to_image(None)
