import numpy as np
from shrink.visualization import visualize, create_gif_from_images
from shrink.utils import resize
from shrink.seam import (
    remove_seam,
    get_minimum_seam,
)


def seam_carve(
    im: np.ndarray, dx: float, vis=False, energy="backward", downsize=500, gif=False
):
    if not isinstance(im, np.ndarray):
        raise TypeError("Image should be of type np.ndarray, got %s" % type(im))

    im = im.astype(np.float64)
    _, w = im.shape[:2]
    assert 0 < dx <= w

    if downsize:
        im = resize(im, downsize)

    output = seams_removal(im, dx, vis, energy, gif)

    return output


def seams_removal(im, num_remove, vis=False, energy="backward", gif=False):
    images = []
    for _ in range(num_remove):
        _, boolmask = get_minimum_seam(im, energy=energy)
        if vis:
            visualize(im, interactive=True)
        im = remove_seam(im, boolmask)
        images.append(im)

    if gif:
        create_gif_from_images(images)

    return im
