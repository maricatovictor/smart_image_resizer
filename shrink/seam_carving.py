import numpy as np

from shrink.load import visualize
from shrink.seam import (
    remove_seam,
    remove_seam_grayscale,
    get_minimum_seam,
    add_seam,
    add_seam_grayscale,
)


def seam_carve(im: np.ndarray, dx: float, mask=None, vis=False, energy="backward"):
    if not isinstance(im, np.ndarray):
        raise TypeError("Image should be of type np.ndarray, got %s" % type(im))
    im = im.astype(np.float64)
    _, w = im.shape[:2]
    assert 0 < dx <= w

    if mask is not None:
        mask = mask.astype(np.float64)

    output = im

    output, mask = seams_insertion(output, dx, mask, vis, energy)

    return output


def seams_insertion(im, num_add, mask=None, vis=False, energy="backward"):
    seams_record = []
    temp_im = im.copy()
    temp_mask = mask.copy() if mask is not None else None

    for _ in range(num_add):
        seam_idx, boolmask = get_minimum_seam(temp_im, temp_mask, energy=energy)
        if vis:
            visualize(temp_im)

        seams_record.append(seam_idx)
        temp_im = remove_seam(temp_im, boolmask)
        if temp_mask is not None:
            temp_mask = remove_seam_grayscale(temp_mask, boolmask)

    seams_record.reverse()

    for _ in range(num_add):
        seam = seams_record.pop()
        im = add_seam(im, seam)
        if vis:
            visualize(im)
        if mask is not None:
            mask = add_seam_grayscale(mask, seam)

        # update the remaining seam indices
        for remaining_seam in seams_record:
            remaining_seam[np.where(remaining_seam >= seam)] += 2

    return im, mask
