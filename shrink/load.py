from PIL import Image
import numpy as np
import cv2

from shrink.utils import rotate_image

SEAM_COLOR = np.array([255, 200, 200])  # seam visualization color (BGR)


def load_image(filename: str) -> Image:
    return Image.open(filename)


def visualize(im: np.ndarray, boolmask=None, rotate=False):
    vis = im.astype(np.uint8)
    if boolmask is not None:
        vis[np.where(boolmask == False)] = SEAM_COLOR
    if rotate:
        vis = rotate_image(vis, False)
    return Image.fromarray(im)
