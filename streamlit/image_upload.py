from PIL import Image
import streamlit as st


def image_upload():
    st.set_option("deprecation.showfileUploaderEncoding", False)
    uploaded_file = st.file_uploader(
        "Choose an image to upload", type=["png", "jpg", "jpeg"]
    )
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        return image
