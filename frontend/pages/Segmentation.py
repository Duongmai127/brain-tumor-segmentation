import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.append(project_root)

import os

# import tempfile
import streamlit as st
import wandb
from dotenv import load_dotenv

# from models.model_inference import SingleFileInference
# import torch
from frontend.components.results import ResultsDisplay

import requests
import numpy as np

# Load environment variables
load_dotenv()

api_url = os.getenv("API_URL")


def initialize_wandb():
    """Initialize wandb with API key from environment"""
    api_key = os.getenv("WANDB_API_KEY")
    if not api_key:
        st.error("WANDB API key not found in environment variables!")
        st.stop()

    wandb.login(key=api_key)
    return wandb.init(project="glioma-brain-tumor-segmentation", job_type="inference")


def check_api_health():
    """Check if the API is running and healthy"""
    try:
        response = requests.get(f"{api_url}/health")
        return response.status_code == 200 and response.json()["model_loaded"]
    except requests.RequestException:
        return False


def main():
    st.set_page_config(page_title="Brain Tumor Segmentation", layout="wide")
    st.title("Brain Tumor Segmentation")

    # Initialize W&B
    run = initialize_wandb()

    try:
        # Check API health
        if not check_api_health():
            st.error(
                "API is not available or model is not loaded. Please ensure the API is running."
            )
            st.stop()

        st.success("Connected to API successfully!")

        # File upload
        uploaded_file = st.file_uploader("Choose a NIfTI file", type="nii.gz")

        if uploaded_file:
            with st.spinner("Processing..."):
                # Send file to API
                files = {
                    "file": (
                        "image.nii.gz",
                        uploaded_file.getvalue(),
                        "application/octet-stream",
                    )
                }
                response = requests.post(f"{api_url}/predict", files=files)

                if response.status_code != 200:
                    st.error(f"API Error: {response.json()['detail']}")
                    st.stop()

                # Parse response
                result = response.json()
                original_img = np.array(result["original_img"])
                predictions = np.array(result["predictions"])

                # Initialize and display results
                results_display = ResultsDisplay()
                results_display.show(original_img=original_img, predictions=predictions)
                st.success("Processing complete!")

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

    finally:
        # Cleanup button in sidebar
        if st.sidebar.button("End Session"):
            if wandb.run is not None:
                wandb.finish()
            st.success("Session ended successfully!")


if __name__ == "__main__":
    main()
