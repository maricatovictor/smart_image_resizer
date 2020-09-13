import streamlit as st
import numpy as np

from shrink.seam_carving import seam_carve
from image_upload import image_upload

st.header("Smart Image Shrinker")

image = st.empty()
original_image = image_upload()


@st.cache()
def update_with_original_image(original_image):
    image.image(original_image, width=640)


if original_image:
    update_with_original_image(original_image)
    shrink_dx = st.slider("Shrinker Width", 1, 100)
    shrinked_image = seam_carve(np.asarray(original_image), shrink_dx,)
    image.image(shrinked_image.astype(np.uint8), width=640)
