import os
import tempfile
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import numpy as np
from models.model_inference import SingleFileInference
import wandb
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Global model variable
model = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for startup and shutdown events
    """
    # Startup
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

        yield  # Server is running and ready to accept requests

    finally:
        # Shutdown
        if wandb.run is not None:
            wandb.finish()


# Initialize FastAPI with lifespan
app = FastAPI(title="Brain Tumor Segmentation API", lifespan=lifespan)


@app.post("/predict")
async def predict(file: UploadFile):
    """
    Process a NIfTI file and return segmentation predictions
    """
    if not file.filename.endswith(".nii.gz"):
        raise HTTPException(status_code=400, detail="Only .nii.gz files are supported")

    if not model:
        raise HTTPException(status_code=500, detail="Model not initialized")

    temp_file_path = None
    try:
        # Create a unique temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".nii.gz") as tmp_file:
            temp_file_path = tmp_file.name
            content = await file.read()
            tmp_file.write(content)

        # Process the image
        data = model.transforms({"image": temp_file_path})
        original_img = data["image"].numpy()

        # Get predictions
        predictions = model.predict(temp_file_path)
        predictions_np = predictions.numpy()

        # Convert numpy arrays to lists for JSON serialization
        return {
            "original_img": original_img.tolist(),
            "predictions": predictions_np.tolist(),
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

    finally:
        # Clean up temporary file in finally block
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.unlink(temp_file_path)
            except Exception as e:
                print(f"Warning: Failed to delete temporary file: {e}")


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
    return {"status": "healthy", "model_loaded": model is not None}
