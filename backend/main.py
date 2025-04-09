import os
import torch
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
from torchvision import transforms
from typing import List
import uvicorn
from pathlib import Path
import torch.nn as nn


# Define the model class directly in the file
class ChestXRayModel(nn.Module):
    def __init__(self, num_classes):
        super(ChestXRayModel, self).__init__()
        # ResNet-like architecture
        self.features = nn.Sequential(
            # Initial conv layer
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            # Block 1
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Block 2
            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Block 3
            nn.Conv2d(256, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Adaptive pooling to ensure fixed size output regardless of input size
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Fully connected layer for classification
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


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

# Get the absolute path to the backend directory
BACKEND_DIR = Path(__file__).parent.absolute()
MODEL_DIR = BACKEND_DIR / "Model"
MODEL_PATH = MODEL_DIR / "chest_xray_model.pth"
CLASS_NAMES_PATH = MODEL_DIR / "class_names.npy"

# Define image transformation pipeline
transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Flag to indicate if model is loaded
model_loaded = False
class_names = []

try:
    if MODEL_PATH.exists() and CLASS_NAMES_PATH.exists():
        # Load class names
        class_names = np.load(CLASS_NAMES_PATH, allow_pickle=True)
        num_classes = len(class_names)
        
        # Initialize model
        model = ChestXRayModel(num_classes)
        
        # Load model weights
        state_dict = torch.load(MODEL_PATH, map_location=device)
        model.load_state_dict(state_dict, strict=False)
        
        model.to(device)
        model.eval()
        model_loaded = True
        print("Model loaded successfully")
    else:
        print(f"Model files not found, using dummy predictions")
        class_names = ["Pneumonia", "COVID-19", "Tuberculosis", "Normal"]
except Exception as e:
    print(f"Error loading model: {str(e)}")
    class_names = ["Pneumonia", "COVID-19", "Tuberculosis", "Normal"]


@app.get("/")
async def root():
    return {"message": "Chest X-Ray Classification API"}


@app.get("/health")
async def health_check():
    return {
        "status": "healthy" if model_loaded else "model_not_loaded",
        "model_status": "loaded" if model_loaded else "not_loaded",
        "device": str(device),
        "model_path_exists": MODEL_PATH.exists(),
        "class_names_path_exists": CLASS_NAMES_PATH.exists(),
        "classes": len(class_names) if isinstance(class_names, (list, np.ndarray)) else 0
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not model_loaded:
        return {
            "predictions": [
                {"class": "Normal", "probability": 0.85},
                {"class": "Pneumonia", "probability": 0.15},
            ],
            "message": "Test prediction (model not loaded)",
        }

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        image_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.sigmoid(outputs)
            predictions = (probabilities > 0.5).float()

        predictions = predictions.cpu().numpy()[0]
        probabilities = probabilities.cpu().numpy()[0]

        predicted_classes = []
        for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
            if pred == 1:
                predicted_classes.append(
                    {"class": class_names[i], "probability": float(prob)}
                )

        predicted_classes.sort(key=lambda x: x["probability"], reverse=True)

        return {
            "predictions": predicted_classes,
            "message": "Successfully processed image",
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/classes")
async def get_classes():
    return {
        "classes": (
            class_names if isinstance(class_names, list) else class_names.tolist()
        )
    }


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
