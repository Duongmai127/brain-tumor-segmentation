import os
import tempfile
import hashlib
from contextlib import asynccontextmanager
from typing import Dict, Any
from fastapi import FastAPI, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.middleware.gzip import GZipMiddleware
import numpy as np
from models.model_inference import SingleFileInference
import wandb
from dotenv import load_dotenv
from functools import lru_cache
import asyncio
from concurrent.futures import ThreadPoolExecutor

# Load environment variables
load_dotenv()

# Global variables
model = None
# In-memory cache for predictions
prediction_cache: Dict[str, Any] = {}
# Thread pool for parallel processing
thread_pool = ThreadPoolExecutor(max_workers=3)


def compute_file_hash(content: bytes) -> str:
    """Compute SHA-256 hash of file content"""
    return hashlib.sha256(content).hexdigest()


def process_image_in_thread(temp_file_path: str):
    """Process image in a separate thread"""
    data = model.transforms({"image": temp_file_path})
    original_img = data["image"].numpy()

    # Convert to float16 to reduce memory usage and transfer size
    original_img = original_img.astype(np.float16)

    # Get predictions
    predictions = model.predict(temp_file_path)
    predictions_np = predictions.numpy().astype(np.float16)

    return original_img, predictions_np


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown events"""
    global model
    try:
        # Initialize wandb
        api_key = os.getenv("WANDB_API_KEY")
        if not api_key:
            raise HTTPException(status_code=500, detail="WANDB API key not found")

        wandb.login(key=api_key)
        wandb.init(project="glioma-brain-tumor-segmentation", job_type="api-inference")

        # Initialize model
        wandb_artifact = os.getenv("WANDB_ARTIFACT_CHECKPOINT")
        if not wandb_artifact:
            raise HTTPException(status_code=500, detail="WANDB artifact not configured")

        model = SingleFileInference()
        model.load_model(wandb_artifact=wandb_artifact)

        yield

    finally:
        thread_pool.shutdown(wait=False)
        if wandb.run is not None:
            wandb.finish()


# Initialize FastAPI with lifespan
app = FastAPI(title="Brain Tumor Segmentation API", lifespan=lifespan)

# Add GZip compression middleware
app.add_middleware(GZipMiddleware, minimum_size=1000)


def cleanup_temp_file(temp_file_path: str):
    """Background task to clean up temporary file"""
    if os.path.exists(temp_file_path):
        try:
            os.unlink(temp_file_path)
        except Exception as e:
            print(f"Warning: Failed to delete temporary file: {e}")


@app.post("/predict")
async def predict(file: UploadFile, background_tasks: BackgroundTasks):
    """Process a NIfTI file and return segmentation predictions"""
    if not file.filename.endswith(".nii.gz"):
        raise HTTPException(status_code=400, detail="Only .nii.gz files are supported")

    if not model:
        raise HTTPException(status_code=500, detail="Model not initialized")

    content = await file.read()
    file_hash = compute_file_hash(content)

    # Check cache
    if file_hash in prediction_cache:
        return prediction_cache[file_hash]

    temp_file_path = None
    try:
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".nii.gz") as tmp_file:
            temp_file_path = tmp_file.name
            tmp_file.write(content)

        # Process image in thread pool
        loop = asyncio.get_event_loop()
        original_img, predictions_np = await loop.run_in_executor(
            thread_pool, process_image_in_thread, temp_file_path
        )

        # Prepare response
        response = {
            "original_img": original_img.tolist(),
            "predictions": predictions_np.tolist(),
        }

        # Cache the result
        prediction_cache[file_hash] = response

        # Schedule cleanup as a background task
        background_tasks.add_task(cleanup_temp_file, temp_file_path)

        return response

    except Exception as e:
        if temp_file_path:
            background_tasks.add_task(cleanup_temp_file, temp_file_path)
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "name": "Brain Tumor Segmentation API",
        "version": "1.0",
        "endpoints": {
            "/": "This information",
            "/health": "Health check endpoint",
            "/predict": "Upload and process NIfTI files (POST)",
        },
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "cache_size": len(prediction_cache),
    }


# Cache cleanup endpoint
@app.post("/clear-cache")
async def clear_cache():
    """Clear the prediction cache"""
    prediction_cache.clear()
    return {"status": "Cache cleared"}
