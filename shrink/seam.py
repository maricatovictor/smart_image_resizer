import numpy as np
from shrink.energy import get_energy_fn
from numba import jit


@jit(forceobj=True)
def remove_seam(im, boolmask):
    h, w = im.shape[:2]
    boolmask3c = np.stack([boolmask] * 3, axis=2)
    return im[boolmask3c].reshape((h, w - 1, 3))


def get_minimum_seam(im, energy="backward"):
    """
    DP algorithm for finding the seam of minimum energy. Code adapted from
    https://karthikkaranth.me/blog/implementing-seam-carving-with-python/
    """
    h, w = im.shape[:2]
    energyfn = get_energy_fn(energy)
    M = energyfn(im)

    backtrack = np.zeros_like(M, dtype=np.int)

    M = populate_matrix(M, backtrack, h, w)

    seam_idx, boolmask = find_path_backtracking(M, backtrack, h, w)

    return np.array(seam_idx), boolmask


@jit
def populate_matrix(M, backtrack, h, w):
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
    return M


def find_path_backtracking(M, backtrack, h, w):
    seam_idx = []
    boolmask = np.ones((h, w), dtype=np.bool)
    j = np.argmin(M[-1])
    for i in range(h - 1, -1, -1):
        boolmask[i, j] = False
        seam_idx.append(j)
        j = backtrack[i, j]

    seam_idx.reverse()
    return seam_idx, boolmask
