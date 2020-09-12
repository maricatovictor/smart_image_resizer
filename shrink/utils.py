import numpy as np
import cv2


def resize(image, width):
    dim = None
    h, w = image.shape[:2]
    dim = (width, int(h * width / float(w)))
    return cv2.resize(image, dim)


def rotate_image(image, clockwise):
    k = 1 if clockwise else 3
    return np.rot90(image, k)
