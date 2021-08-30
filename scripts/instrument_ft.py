import pyelsa
import numpy as np
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('-i', '--inverse', dest="inverse", action='store_true',
                    help='compute IFFT if this flag is specified, else use FFT by default')
parser.add_argument('-s', '--source', dest='source', action='store', required=True, help='read EDF array from source')
parser.add_argument('-d', '--destination', dest='destination', action='store',
                    help='write transformed EDF array to destination')
args = vars(parser.parse_args())


def instrument_fourier_transform(source, destination, inverse):
    """
    THIS SCRIPT SHOULD NOT EXIST.
    With that being said, as a workaround, it is meant to be executed while running the shearlet transform in elsa. An
    EDF array is created and persisted by elsa. Then, this functionality is called through the 'system' command, the
    respective Fourier transform is performed, and the output written in an EDF file. This is then picked up by elsa,
    and used accordingly.
    """
    obj = pyelsa.EDF.readf(source)

    arr = np.array(obj)

    if not inverse:
        output = np.fft.fft2(arr)
    else:
        output = np.fft.ifft2(arr)

    output = np.real(output)

    if destination is None:
        destination = source

    pyelsa.EDF.write(pyelsa.DataContainer(output), destination)


inv = args['inverse']
src = args['source']
dest = args['destination']

instrument_fourier_transform(src, dest, inv)
