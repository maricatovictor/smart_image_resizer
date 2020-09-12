import numpy as np
from shrink.energy import get_energy_fn
from numba import jit

ENERGY_MASK_CONST = 100000.0  # large energy value for protective masking
MASK_THRESHOLD = 10  # minimum pixel intensity for binary mask


@jit
def add_seam(im, seam_idx):
    """
    Add a vertical seam to a 3-channel color image at the indices provided
    by averaging the pixels values to the left and right of the seam.
    Code adapted from https://github.com/vivianhylee/seam-carving.
    """
    h, w = im.shape[:2]
    output = np.zeros((h, w + 1, 3))
    for row in range(h):
        col = seam_idx[row]
        for ch in range(3):
            if col == 0:
                p = np.mean(im[row, col : col + 2, ch])
                output[row, col, ch] = im[row, col, ch]
                output[row, col + 1, ch] = p
                output[row, col + 1 :, ch] = im[row, col:, ch]
            else:
                p = np.mean(im[row, col - 1 : col + 1, ch])
                output[row, :col, ch] = im[row, :col, ch]
                output[row, col, ch] = p
                output[row, col + 1 :, ch] = im[row, col:, ch]

    return output


@jit
def add_seam_grayscale(im, seam_idx):
    """
    Add a vertical seam to a grayscale image at the indices provided
    by averaging the pixels values to the left and right of the seam.
    """
    h, w = im.shape[:2]
    output = np.zeros((h, w + 1))
    for row in range(h):
        col = seam_idx[row]
        if col == 0:
            p = np.average(im[row, col : col + 2])
            output[row, col] = im[row, col]
            output[row, col + 1] = p
            output[row, col + 1 :] = im[row, col:]
        else:
            p = np.average(im[row, col - 1 : col + 1])
            output[row, :col] = im[row, :col]
            output[row, col] = p
            output[row, col + 1 :] = im[row, col:]

    return output


@jit(forceobj=True)
def remove_seam(im, boolmask):
    h, w = im.shape[:2]
    boolmask3c = np.stack([boolmask] * 3, axis=2)
    return im[boolmask3c].reshape((h, w - 1, 3))


@jit
def remove_seam_grayscale(im, boolmask):
    h, w = im.shape[:2]
    return im[boolmask].reshape((h, w - 1))


def get_minimum_seam(im, energy="backward"):
    """
    DP algorithm for finding the seam of minimum energy. Code adapted from
    https://karthikkaranth.me/blog/implementing-seam-carving-with-python/
    """
    h, w = im.shape[:2]
    energyfn = get_energy_fn(energy)
    M = energyfn(im)

    backtrack = np.zeros_like(M, dtype=np.int)

    # populate DP matrix
    for i in range(1, h):
        for j in range(0, w):
            if j == 0:
                idx = np.argmin(M[i - 1, j : j + 2])
                backtrack[i, j] = idx + j
                min_energy = M[i - 1, idx + j]
            else:
                idx = np.argmin(M[i - 1, j - 1 : j + 2])
                backtrack[i, j] = idx + j - 1
                min_energy = M[i - 1, idx + j - 1]

            M[i, j] += min_energy

    # backtrack to find path
    seam_idx = []
    boolmask = np.ones((h, w), dtype=np.bool)
    j = np.argmin(M[-1])
    for i in range(h - 1, -1, -1):
        boolmask[i, j] = False
        seam_idx.append(j)
        j = backtrack[i, j]

    seam_idx.reverse()
    return np.array(seam_idx), boolmask
