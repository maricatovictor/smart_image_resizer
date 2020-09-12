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
    img_rgb = cv2.cvtColor(image.astype(np.float32), cv2.COLOR_BGR2RGB)
    cv2.imwrite(name, img_rgb)


def resize(image, width):
    dim = None
    h, w = image.shape[:2]
    dim = (width, int(h * width / float(w)))
    return cv2.resize(image, dim)
