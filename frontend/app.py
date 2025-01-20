import sys
import os

print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES", "Not Set"))
# Add the project root to the Python path
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# frontend/app.py
import streamlit as st

# from pages import home, upload

# Configure the app
st.set_page_config(
    page_title="Glioma Segmentation",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Create sidebar navigation
page = st.sidebar.success("Select a page to get started")

st.markdown(
    """ # Glioma Segmentation App 🧠
            This app performs glioma segmentation on multi-modal MRI scans using a 3D U-Net model.
            Check out the [wandb dashboard](https://api.wandb.ai/links/duongmaixa1207-university-of-south-florida/ncle7vv1) for model training details and interesting visualizations.
            """
)
# Add footer
st.sidebar.markdown("---")
st.sidebar.markdown("Made with ❤️ by Duong Mai")
