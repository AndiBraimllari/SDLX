import pyelsa
import numpy as np
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('-d', '--direct', dest="direct", action='store_true', help='direct shearlet transform or not')
args = vars(parser.parse_args())


def delegate_fourier_transform(direct):
    """
    THIS SCRIPT SHOULD NOT EXIST.
    With that being said, as a workaround, it is meant to be executed while running the shearlet transform in elsa. An
    EDF array is created and persisted by elsa. Then, this functionality is called through the 'system' command, the
    respective Fourier transform is performed, and the output written in an EDF file. This is then picked up by elsa,
    and used accordingly.
    """
    spectra = pyelsa.EDF.readf('spectra.edf')
    spectra = np.array(spectra)

    if direct:
        img = pyelsa.EDF.readf('f.edf')

        img = np.array(img)

        v = np.zeros_like(spectra, dtype=complex)

        for i in range(spectra.shape[2]):
            v[:, :, i] = spectra[:, :, i] * np.fft.fft2(img)

        tempo = np.real(np.fft.ifft2(v, axes=(0, 1)))
        pyelsa.EDF.write(pyelsa.DataContainer(tempo), 'shearTrf.edf')
    else:
        y = pyelsa.EDF.readf('y.edf')

        y = np.array(y)

        tempo = np.real(np.sum(np.fft.ifft2(np.fft.fft2(y, axes=(0, 1)) * spectra, axes=(0, 1)), axis=2))
        pyelsa.EDF.write(pyelsa.DataContainer(tempo), 'invShearTrf.edf')


delegate_fourier_transform(args['direct'])
