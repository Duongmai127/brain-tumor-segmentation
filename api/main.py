import os
import tempfile
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse, JSONResponse
import numpy as np
from models.model_inference import SingleFileInference
import wandb
from dotenv import load_dotenv
import json
import asyncio
from typing import AsyncGenerator

# Load environment variables
load_dotenv()

# Global model variable
model = None


@asynccontextmanager
async def lifespan(app: FastAPI):
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
        if wandb.run is not None:
            wandb.finish()


app = FastAPI(title="Brain Tumor Segmentation API", lifespan=lifespan)


async def process_chunk(array: np.ndarray) -> AsyncGenerator[str, None]:
    """Process numpy array in chunks"""
    CHUNK_SIZE = 1000  # Adjust based on your needs
    total_elements = array.size

    # Send total size first
    yield json.dumps({"total_elements": total_elements}) + "\n"

    # Process array in chunks
    for i in range(0, total_elements, CHUNK_SIZE):
        chunk = array.flat[i : i + CHUNK_SIZE].tolist()
        yield json.dumps({"chunk_start": i, "data": chunk}) + "\n"
        await asyncio.sleep(0.01)  # Prevent blocking


@app.post("/predict_stream")
async def predict_stream(file: UploadFile, background_tasks: BackgroundTasks):
    """Stream predictions chunk by chunk"""
    if not file.filename.endswith(".nii.gz"):
        raise HTTPException(status_code=400, detail="Only .nii.gz files are supported")

    if not model:
        raise HTTPException(status_code=500, detail="Model not initialized")

    temp_file_path = None
    try:
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".nii.gz") as tmp_file:
            temp_file_path = tmp_file.name
            content = await file.read()
            tmp_file.write(content)

        # Process the image
        data = model.transforms({"image": temp_file_path})
        original_img = data["image"].numpy()
        predictions = model.predict(temp_file_path)
        predictions_np = predictions.numpy()

        # Create streaming response
        async def generate():
            # Send metadata first
            yield json.dumps(
                {
                    "metadata": {
                        "original_shape": original_img.shape,
                        "predictions_shape": predictions_np.shape,
                    }
                }
            ) + "\n"

            # Stream original image
            async for chunk in process_chunk(original_img):
                yield chunk

            # Stream predictions
            async for chunk in process_chunk(predictions_np):
                yield chunk

        return StreamingResponse(generate(), media_type="application/x-ndjson")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.unlink(temp_file_path)
            except Exception as e:
                print(f"Warning: Failed to delete temporary file: {e}")


# Keep existing endpoints
@app.get("/")
async def root():
    return {
        "name": "Brain Tumor Segmentation API",
        "version": "1.0",
        "endpoints": {
            "/": "This information",
            "/health": "Health check endpoint",
            "/predict_stream": "Stream predictions for NIfTI files (POST)",
        },
    }


@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": model is not None}
