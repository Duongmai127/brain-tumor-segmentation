import sys
import os

# print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES", "Not Set"))
# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import sys
import os
import time
import logging
import streamlit as st
import wandb
import torch
from dotenv import load_dotenv
from components.results import ResultsDisplay
import numpy as np
from models.model_inference import SingleFileInference
import tempfile

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

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


def initialize_model():
    """Initialize the segmentation model"""
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {device}")

        model = SingleFileInference(device=device)
        wandb_artifact = os.getenv("WANDB_ARTIFACT_CHECKPOINT")
        if not wandb_artifact:
            st.error("WANDB_MODEL_ARTIFACT not found in environment variables!")
            st.stop()

        model.load_model(wandb_artifact)
        return model
    except Exception as e:
        logger.error(f"Model initialization failed: {str(e)}")
        st.error(f"Failed to initialize model: {str(e)}")
        st.stop()


def main():
    logger.info("Starting Streamlit application")
    st.set_page_config(page_title="Brain Tumor Segmentation", layout="wide")
    st.title("Brain Tumor Segmentation")

    # Initialize W&B
    start_time = time.time()
    run = initialize_wandb()
    logger.info(f"W&B initialization took {time.time() - start_time:.2f} seconds")

    try:
        # Initialize model
        model_init_start = time.time()
        model = initialize_model()
        logger.info(
            f"Model initialization took {time.time() - model_init_start:.2f} seconds"
        )
        st.success("Model loaded successfully!")

        # File upload
        uploaded_file = st.file_uploader("Choose a NIfTI file", type="nii.gz")

        if uploaded_file:
            logger.info(f"File uploaded: {uploaded_file.name}")
            file_size = len(uploaded_file.getvalue()) / (1024 * 1024)  # Size in MB
            logger.info(f"File size: {file_size:.2f} MB")
            st.info(f"File size: {file_size:.2f} MB")

            with st.spinner("Processing..."):
                # Save uploaded file temporarily
                with tempfile.NamedTemporaryFile(
                    suffix=".nii.gz", delete=False
                ) as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_path = tmp_file.name

                try:
                    # Run inference
                    inference_start = time.time()
                    predictions = model.predict(tmp_path)
                    logger.info(
                        f"Inference took {time.time() - inference_start:.2f} seconds"
                    )

                    # Get original image data
                    data = model.transforms({"image": tmp_path})
                    original_img = data["image"].numpy()

                    # Convert predictions to numpy for visualization
                    predictions = predictions.numpy()

                    logger.info(f"Original image shape: {original_img.shape}")
                    logger.info(f"Predictions shape: {predictions.shape}")

                    # Display results
                    display_start = time.time()
                    results_display = ResultsDisplay()
                    results_display.show(
                        original_img=original_img, predictions=predictions
                    )
                    logger.info(
                        f"Results display took {time.time() - display_start:.2f} seconds"
                    )
                    st.success("Processing complete!")

                finally:
                    # Clean up temporary file
                    os.unlink(tmp_path)

    except Exception as e:
        logger.error(f"An error occurred: {str(e)}", exc_info=True)
        st.error(f"An error occurred: {str(e)}")

    finally:
        # Cleanup button in sidebar
        if st.sidebar.button("End Session"):
            if wandb.run is not None:
                wandb.finish()
            logger.info("Session ended")
            st.success("Session ended successfully!")


if __name__ == "__main__":
    main()

# import sys
# import os
# import time
# import logging
# import streamlit as st
# import wandb
# from dotenv import load_dotenv
# from components.results import ResultsDisplay
# import requests
# import numpy as np

# # Set up logging
# logging.basicConfig(
#     level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
# )
# logger = logging.getLogger(__name__)

# # Load environment variables
# load_dotenv()

# api_url = os.getenv("API_URL")


# def initialize_wandb():
#     """Initialize wandb with API key from environment"""
#     api_key = os.getenv("WANDB_API_KEY")
#     if not api_key:
#         st.error("WANDB API key not found in environment variables!")
#         st.stop()

#     wandb.login(key=api_key)
#     return wandb.init(project="glioma-brain-tumor-segmentation", job_type="inference")


# def check_api_health():
#     """Check if the API is running and healthy"""
#     try:
#         response = requests.get(f"{api_url}/health")
#         return response.status_code == 200 and response.json()["model_loaded"]
#     except requests.RequestException as e:
#         logger.error(f"API health check failed: {str(e)}")
#         return False


# def main():
#     logger.info("Starting Streamlit application")
#     st.set_page_config(page_title="Brain Tumor Segmentation", layout="wide")
#     st.title("Brain Tumor Segmentation")

#     # Initialize W&B
#     start_time = time.time()
#     run = initialize_wandb()
#     logger.info(f"W&B initialization took {time.time() - start_time:.2f} seconds")

#     try:
#         # Check API health
#         api_check_start = time.time()
#         if not check_api_health():
#             st.error(
#                 "API is not available or model is not loaded. Please ensure the API is running."
#             )
#             st.stop()
#         logger.info(
#             f"API health check took {time.time() - api_check_start:.2f} seconds"
#         )

#         st.success("Connected to API successfully!")

#         # File upload
#         uploaded_file = st.file_uploader("Choose a NIfTI file", type="nii.gz")

#         if uploaded_file:
#             logger.info(f"File uploaded: {uploaded_file.name}")
#             file_size = len(uploaded_file.getvalue()) / (1024 * 1024)  # Size in MB
#             logger.info(f"File size: {file_size:.2f} MB")
#             st.info(f"File size: {file_size:.2f} MB")

#             with st.spinner("Processing..."):
#                 # Send file to API
#                 request_start = time.time()
#                 files = {
#                     "file": (
#                         "image.nii.gz",
#                         uploaded_file.getvalue(),
#                         "application/octet-stream",
#                     )
#                 }
#                 logger.info("Sending request to API...")
#                 response = requests.post(f"{api_url}/predict", files=files)
#                 logger.info(
#                     f"API request took {time.time() - request_start:.2f} seconds"
#                 )

#                 if response.status_code != 200:
#                     error_msg = response.json().get("detail", "Unknown error")
#                     logger.error(f"API Error: {error_msg}")
#                     st.error(f"API Error: {error_msg}")
#                     st.stop()

#                 # Parse response
#                 parse_start = time.time()
#                 result = response.json()
#                 original_img = np.array(result["original_img"])
#                 predictions = np.array(result["predictions"])
#                 logger.info(
#                     f"Response parsing took {time.time() - parse_start:.2f} seconds"
#                 )
#                 logger.info(f"Original image shape: {original_img.shape}")
#                 logger.info(f"Predictions shape: {predictions.shape}")

#                 # Display results
#                 display_start = time.time()
#                 results_display = ResultsDisplay()
#                 results_display.show(original_img=original_img, predictions=predictions)
#                 logger.info(
#                     f"Results display took {time.time() - display_start:.2f} seconds"
#                 )
#                 st.success("Processing complete!")

#     except Exception as e:
#         logger.error(f"An error occurred: {str(e)}", exc_info=True)
#         st.error(f"An error occurred: {str(e)}")

#     finally:
#         # Cleanup button in sidebar
#         if st.sidebar.button("End Session"):
#             if wandb.run is not None:
#                 wandb.finish()
#             logger.info("Session ended")
#             st.success("Session ended successfully!")


# if __name__ == "__main__":
#     main()
