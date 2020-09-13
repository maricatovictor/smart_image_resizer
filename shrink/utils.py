from PIL import Image
import numpy as np
import cv2
from shrink.visualization import visualize


def load_image(filename: str) -> Image:
    return Image.open(filename)


def save(image: np.ndarray, name="img.jpg"):
    visualize(image)
    img_rgb = cv2.cvtColor(image.astype(np.float32), cv2.COLOR_BGR2RGB)
    cv2.imwrite(name, img_rgb)


def resize(image, width):
    dim = None
    h, w = image.shape[:2]
    dim = (width, int(h * width / float(w)))
    return cv2.resize(image, dim)
