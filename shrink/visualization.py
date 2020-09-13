import matplotlib.pyplot as plt
from IPython.display import clear_output
import numpy as np
from typing import List
import imageio
from PIL import Image


def visualize(im, interactive=True):
    if interactive:
        clear_output(wait=True)
    vis = im.astype(np.uint8)
    plt.imshow(vis)
    plt.show()


def create_gif_from_images(images: List[np.ndarray], path="data/process.gif"):
    frames = []
    for img in images:
        frames.append(Image.fromarray(img.astype(np.uint8)))
    imageio.mimsave(path, frames)
    print(f"Successfully generated gif at {path}")
