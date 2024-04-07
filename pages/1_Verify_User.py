import os.path

import streamlit as st
from PIL import Image
import numpy as np
from verification_system import VerificationSystem


st.set_page_config(page_title="Verify User", page_icon="👁️")


def verify_user(user_name: str, user_image: bytes):
    column_left, column_right = st.columns(2)

    with column_left:
        st.image(image=user_image, use_column_width="auto")

    try:
        image = Image.open(user_image)
        image_array = np.array(image)

        is_verified = face_verification_system.verify_user(
            user_name=user_name, user_photo_path=image_array
        )

        with column_right:
            if is_verified:
                st.success("Verified")
            else:
                st.error("Not Verified")
    except Exception as e:
        with column_right:
            st.exception(e)


st.title("Verify User")

face_verification_system = VerificationSystem(
    database_path=os.path.join("data", "database")
)

uploaded_name = st.text_input(label="Username")

if uploaded_name:
    uploaded_image = st.file_uploader(
        label="Choose an image...",
        type=["jpg", "jpeg", "png"],
        label_visibility="hidden",
    )

    if uploaded_image is not None:
        verify_user(user_name=uploaded_name, user_image=uploaded_image)
