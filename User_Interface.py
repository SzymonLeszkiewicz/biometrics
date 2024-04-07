import streamlit as st
import pandas as pd
import numpy as np


st.set_page_config(
    page_title="System Overview",
    page_icon="📰",
)

st.write("# Welcome to Face Verification System! 👋")

st.markdown(
    """
    👈 **Select page from the sidebar** to see some examples of what our system can do!
    
    ### Available methods
    - Verify User
    - Facial Analysis
    """
)

st.write("## Results Overview")
st.dataframe(
    pd.DataFrame(
        np.random.randn(1000, 2) / [50, 50] + [37.76, -122.4], columns=["lat", "lon"]
    ).transpose()
)
