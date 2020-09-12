from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt

SEAM_COLOR = np.array([255, 200, 200])  # seam visualization color (BGR)


def load_image(filename: str) -> Image:
    return Image.open(filename)


def visualize(im):
    vis = im.astype(np.uint8)
    plt.imshow(vis)


def save(image: np.ndarray, name="img.jpg"):
    visualize(image)
    cv2.imwrite(name, image)
