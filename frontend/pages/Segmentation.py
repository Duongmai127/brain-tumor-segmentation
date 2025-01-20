import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.append(project_root)

import os
import tempfile
import streamlit as st
import wandb
from dotenv import load_dotenv
from models.model_inference import SingleFileInference
import torch
from frontend.components.results import ResultsDisplay

# Load environment variables
load_dotenv()


def initialize_wandb():
    """Initialize wandb with API key from environment"""
    api_key = os.getenv("WANDB_API_KEY")
    if not api_key:
        st.error("WANDB API key not found in environment variables!")
        st.stop()

    wandb.login(key=api_key)
    return wandb.init(project="glioma-brain-tumor-segmentation", job_type="inference")


def main():
    st.set_page_config(page_title="Brain Tumor Segmentation", layout="wide")
    st.title("Brain Tumor Segmentation")

    # Initialize W&B
    run = initialize_wandb()
    # print(f"Cuda support is {torch.cuda.is_available()}")
    # print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES", "Not Set"))
    # print(
    #     "Current GPU device:",
    #     torch.cuda.current_device() if torch.cuda.is_available() else "None",
    # )
    try:
        # Initialize model
        with st.spinner("Loading model..."):
            wandb_artifact = os.getenv("WANDB_ARTIFACT_CHECKPOINT")
            segmentation = SingleFileInference()
            segmentation.load_model(wandb_artifact=wandb_artifact)

        st.success("Model loaded successfully!")

        # File upload
        uploaded_file = st.file_uploader("Choose a NIfTI file", type="nii.gz")

        if uploaded_file:
            with st.spinner("Processing..."):
                # Create temporary file
                with tempfile.NamedTemporaryFile(
                    delete=False, suffix=".nii.gz"
                ) as tmp_file:
                    tmp_file.write(uploaded_file.read())
                    tmp_file.flush()

                    # Load original image
                    data = segmentation.transforms({"image": tmp_file.name})
                    original_img = data["image"].numpy()  # Shape: (4, H, W, D)

                    # Run inference
                    predictions = segmentation.predict(
                        tmp_file.name
                    )  # Shape: (3, H, W, D)

                    # Initialize and display results
                    results_display = ResultsDisplay()
                    results_display.show(
                        original_img=original_img, predictions=predictions
                    )

                    # if predictions is not None:
                    #     # Display results in columns
                    #     st.success("Segmentation complete!")

                    #     col1, col2, col3 = st.columns(3)

                    #     slice_index = 77

                    #     with col1:
                    #         st.subheader("Tumor Core (TC)")
                    #         st.image(
                    #             predictions[0].numpy()[:, :, slice_index],
                    #             caption="TC Prediction",
                    #             clamp=True,
                    #         )

                    #     with col2:
                    #         st.subheader("Whole Tumor (WT)")
                    #         st.image(
                    #             predictions[1].numpy()[:, :, slice_index],
                    #             caption="WT Prediction",
                    #             clamp=True,
                    #         )

                    #     with col3:
                    #         st.subheader("Enhancing Tumor (ET)")
                    #         st.image(
                    #             predictions[2].numpy()[:, :, slice_index],
                    #             caption="ET Prediction",
                    #             clamp=True,
                    #         )

    except Exception as e:
        st.error(f"Tumor error occurred: {e}")

    finally:
        # Cleanup button in sidebar
        if st.sidebar.button("End Session"):
            if wandb.run is not None:
                wandb.finish()
            st.success("Session ended successfully!")


if __name__ == "__main__":
    main()
