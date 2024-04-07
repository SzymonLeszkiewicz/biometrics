import os.path

import streamlit as st
import numpy as np
from verification_system import VerificationSystem


st.set_page_config(page_title="Verify Multiple User", page_icon="👁️")


@st.cache_data
def verify_multiple_users(incoming_users_path: str):
    try:
        return face_verification_system.verify_multiple_users(
            incoming_users_path=incoming_users_path
        )
    except Exception as e:
        st.exception(e)


st.title("Verify Multiple Users")

face_verification_system = VerificationSystem(
    database_path=os.path.join("data", "database")
)

uploaded_folder_path = st.text_input(
    "Enter path to the folder containing users profiles:"
)

if os.path.isdir(uploaded_folder_path):
    df_multiple_users = verify_multiple_users(uploaded_folder_path)
    access_granted_rate = face_verification_system.calculate_access_granted_rate(
        df_multiple_users
    )

    st.write("Access Granted Rate:", np.round(access_granted_rate, 3))
    st.dataframe(df_multiple_users)
