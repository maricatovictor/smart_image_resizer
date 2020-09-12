import numpy as np
from scipy import ndimage as ndi


def backward_energy(im):
    """
    Simple gradient magnitude energy map.
    """
    xgrad = ndi.convolve1d(im, np.array([1, 0, -1]), axis=1, mode="wrap")
    ygrad = ndi.convolve1d(im, np.array([1, 0, -1]), axis=0, mode="wrap")

    grad_mag = np.sqrt(np.sum(xgrad ** 2, axis=2) + np.sum(ygrad ** 2, axis=2))

    return grad_mag
