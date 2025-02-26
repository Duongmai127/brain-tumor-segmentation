import os
import tempfile
import time
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, HTTPException
import numpy as np
from models.model_inference import SingleFileInference
import wandb
from dotenv import load_dotenv
import torch

# Load environment variables
load_dotenv()

# Global model variable
model = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    try:
        print("\n=== API Initialization ===")
        print("Starting API initialization...")

        if torch.cuda.is_available():
            print(f"CUDA Device: {torch.cuda.get_device_name(0)}")
            print(f"Initial GPU Memory: {torch.cuda.memory_allocated()/1024**2:.2f} MB")
        else:
            print("CUDA not available")

        api_key = os.getenv("WANDB_API_KEY")
        if not api_key:
            raise HTTPException(status_code=500, detail="WANDB API key not found")

        wandb.login(key=api_key)
        wandb.init(project="glioma-brain-tumor-segmentation", job_type="api-inference")

        wandb_artifact = os.getenv("WANDB_ARTIFACT_CHECKPOINT")
        if not wandb_artifact:
            raise HTTPException(status_code=500, detail="WANDB artifact not configured")

        print("Loading model...")
        model = SingleFileInference()
        model.load_model(wandb_artifact=wandb_artifact)
        print("Model loaded successfully")

        yield

    finally:
        if wandb.run is not None:
            wandb.finish()


app = FastAPI(title="Brain Tumor Segmentation API", lifespan=lifespan)


@app.post("/predict")
async def predict(file: UploadFile):
    print("\n=== Starting New Prediction ===")
    overall_start = time.time()
    print(f"Processing file: {file.filename}")

    if not file.filename.endswith(".nii.gz"):
        raise HTTPException(status_code=400, detail="Only .nii.gz files are supported")

    if not model:
        raise HTTPException(status_code=500, detail="Model not initialized")

    if torch.cuda.is_available():
        print(f"GPU Memory at start: {torch.cuda.memory_allocated()/1024**2:.2f} MB")

    temp_file_path = None
    try:
        # Read file
        read_start = time.time()
        content = await file.read()
        print(f"File read time: {time.time() - read_start:.2f}s")
        print(f"File size: {len(content)/1024**2:.2f} MB")

        # Save to temp file
        write_start = time.time()
        with tempfile.NamedTemporaryFile(delete=False, suffix=".nii.gz") as tmp_file:
            temp_file_path = tmp_file.name
            tmp_file.write(content)
        print(f"File write time: {time.time() - write_start:.2f}s")

        # Transform
        transform_start = time.time()
        print("Starting image transformation...")
        data = model.transforms({"image": temp_file_path})
        original_img = data["image"]
        print(f"Transform time: {time.time() - transform_start:.2f}s")

        if torch.cuda.is_available():
            print(
                f"GPU Memory after transform: {torch.cuda.memory_allocated()/1024**2:.2f} MB"
            )

        # Predict
        predict_start = time.time()
        print("Starting model prediction...")
        predictions = model.predict(temp_file_path)
        predict_time = time.time() - predict_start
        print(f"Prediction time: {predict_time:.2f}s")

        if torch.cuda.is_available():
            print(
                f"GPU Memory after prediction: {torch.cuda.memory_allocated()/1024**2:.2f} MB"
            )

        # Convert and serialize
        convert_start = time.time()
        print("Converting to numpy...")
        original_img_np = original_img.numpy()
        predictions_np = predictions.numpy()
        print(f"Numpy conversion time: {time.time() - convert_start:.2f}s")
        print(f"Original shape: {original_img_np.shape}")
        print(f"Predictions shape: {predictions_np.shape}")

        serialize_start = time.time()
        print("Serializing response...")
        response = {
            "original_img": original_img_np.tolist(),
            "predictions": predictions_np.tolist(),
        }
        print(f"Serialization time: {time.time() - serialize_start:.2f}s")

        total_time = time.time() - overall_start
        print(f"\nTotal processing time: {total_time:.2f}s")

        if torch.cuda.is_available():
            print(f"Final GPU Memory: {torch.cuda.memory_allocated()/1024**2:.2f} MB")

        return response

    except Exception as e:
        print(f"Error occurred: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                cleanup_start = time.time()
                os.unlink(temp_file_path)
                print(f"Cleanup time: {time.time() - cleanup_start:.2f}s")
            except Exception as e:
                print(f"Failed to delete temporary file: {e}")


@app.get("/health")
async def health_check():
    info = {"status": "healthy", "model_loaded": model is not None}
    if torch.cuda.is_available():
        info["gpu_memory"] = f"{torch.cuda.memory_allocated()/1024**2:.2f} MB"
    return info
