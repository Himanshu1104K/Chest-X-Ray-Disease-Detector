import os
import torch
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
from torchvision import transforms
import json
from typing import List, Dict
import uvicorn
import sys
from pathlib import Path
import importlib

# Add the parent directory to the Python path to import the model
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Import the model using the correct path
model_path = (
    project_root / "Deep Learning Model" / "Training" / "chest_xray_model_torch.py"
)
spec = importlib.util.spec_from_file_location("chest_xray_model_torch", str(model_path))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
ChestXRayModel = module.ChestXRayModel

# Initialize FastAPI app
app = FastAPI(title="Chest X-Ray Classification API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Constants
IMAGE_SIZE = (224, 224)
MODEL_PATH = "Model/chest_xray_model.pth"
CLASS_NAMES_PATH = "Model/class_names.npy"

# Define image transformation pipeline
transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load model
try:
    # Load class names first to get the number of classes
    class_names = np.load(CLASS_NAMES_PATH, allow_pickle=True)
    num_classes = len(class_names)

    # Initialize the model with the correct number of classes
    model = ChestXRayModel(num_classes)
    # Load the state dictionary
    state_dict = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
except FileNotFoundError:
    raise Exception("Model file not found. Please ensure chest_xray_model.pth exists.")


@app.get("/")
async def root():
    return {"message": "Chest X-Ray Classification API"}


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Read and validate image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")

        # Preprocess image
        image_tensor = transform(image).unsqueeze(0).to(device)

        # Make prediction
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.sigmoid(outputs)
            predictions = (probabilities > 0.5).float()

        # Convert to numpy
        predictions = predictions.cpu().numpy()[0]
        probabilities = probabilities.cpu().numpy()[0]

        # Get predicted classes and their probabilities
        predicted_classes = []
        for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
            if pred == 1:
                predicted_classes.append(
                    {"class": class_names[i], "probability": float(prob)}
                )

        # Sort by probability
        predicted_classes.sort(key=lambda x: x["probability"], reverse=True)

        return {
            "predictions": predicted_classes,
            "message": "Successfully processed image",
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/classes")
async def get_classes():
    return {"classes": class_names.tolist()}


if __name__ == "__main__":
    uvicorn.run("main:app", reload=True)
