import sys
import os
import streamlit as st
import wandb
from dotenv import load_dotenv
from frontend.components.results import ResultsDisplay
import requests
import numpy as np
import json
from typing import Dict, Any
import time

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


def process_streaming_response(response, progress_bar):
    """Process streaming response and update progress"""
    metadata = None
    original_img_data = []
    predictions_data = []
    original_total = 0
    predictions_total = 0
    current_array = original_img_data  # Start with original image

    for line in response.iter_lines():
        if not line:
            continue

        data = json.loads(line)

        if "metadata" in data:
            metadata = data["metadata"]
            continue

        if "total_elements" in data:
            if not original_total:
                original_total = data["total_elements"]
            else:
                predictions_total = data["total_elements"]
                current_array = predictions_data
            continue

        if "chunk_start" in data:
            current_array.extend(data["data"])
            if current_array is original_img_data:
                progress = len(original_img_data) / original_total
            else:
                progress = 0.5 + (len(predictions_data) / predictions_total) * 0.5
            progress_bar.progress(progress)

    # Reshape arrays according to metadata
    original_img = np.array(original_img_data).reshape(metadata["original_shape"])
    predictions = np.array(predictions_data).reshape(metadata["predictions_shape"])

    return original_img, predictions


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
            # Create progress containers
            progress_container = st.empty()
            progress_bar = st.progress(0)
            status_container = st.empty()

            with st.spinner("Processing..."):
                # Send file to API
                files = {
                    "file": (
                        "image.nii.gz",
                        uploaded_file.getvalue(),
                        "application/octet-stream",
                    )
                }

                # Stream response
                with requests.post(
                    f"{api_url}/predict_stream", files=files, stream=True
                ) as response:
                    if response.status_code != 200:
                        st.error(f"API Error: {response.text}")
                        st.stop()

                    # Process streaming response
                    original_img, predictions = process_streaming_response(
                        response, progress_bar
                    )

                    # Clean up progress indicators
                    progress_container.empty()
                    progress_bar.empty()
                    status_container.empty()

                    # Display results
                    results_display = ResultsDisplay()
                    results_display.show(
                        original_img=original_img, predictions=predictions
                    )
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
