import React, { useState, useEffect } from "react";
import "./App.css";
import About from "./About";
import { API_URL } from "./config";

interface PredictionResult {
  disease: string;
  confidence: number;
}

function App() {
  const [selectedImage, setSelectedImage] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string>("");
  const [prediction, setPrediction] = useState<PredictionResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [showAbout, setShowAbout] = useState(false);
  const [apiStatus, setApiStatus] = useState<string>("");
  const [availableClasses, setAvailableClasses] = useState<string[]>([]);
  const [error, setError] = useState<string>("");

  // Check API health on component mount
  useEffect(() => {
    const checkApiStatus = async () => {
      try {
        console.log(`Checking API health at ${API_URL}/health`);
        const response = await fetch(`${API_URL}/health`);
        
        if (!response.ok) {
          throw new Error(`API health check failed with status: ${response.status}`);
        }
        
        const data = await response.json();
        console.log("API health response:", data);
        setApiStatus(data.status);
        
        // Fetch available disease classes
        console.log(`Fetching classes from ${API_URL}/classes`);
        const classesResponse = await fetch(`${API_URL}/classes`);
        
        if (!classesResponse.ok) {
          throw new Error(`Classes fetch failed with status: ${classesResponse.status}`);
        }
        
        const classesData = await classesResponse.json();
        console.log("API classes response:", classesData);
        setAvailableClasses(classesData.classes);
      } catch (error) {
        console.error("API connection error:", error);
        setApiStatus("disconnected");
      }
    };
    
    checkApiStatus();
  }, []);

  const validateXRayImage = (file: File): Promise<boolean> => {
    return new Promise((resolve) => {
      const img = new Image();
      img.onload = () => {
        // Basic validation criteria for X-ray images:
        // 1. Grayscale or limited color palette
        // 2. Relatively high aspect ratio (chest X-rays are typically portrait or square)
        // 3. Reasonable dimensions for medical images
        
        const canvas = document.createElement('canvas');
        const ctx = canvas.getContext('2d');
        canvas.width = img.width;
        canvas.height = img.height;
        
        if (!ctx) {
          resolve(false);
          return;
        }
        
        ctx.drawImage(img, 0, 0);
        const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
        const data = imageData.data;
        
        // Sample pixels to check if it's likely grayscale
        let colorVariation = 0;
        let sampleSize = 100;
        let sampledPixels = 0;
        
        for (let i = 0; i < data.length; i += 4 * Math.floor(data.length / (4 * sampleSize))) {
          if (sampledPixels >= sampleSize) break;
          
          const r = data[i];
          const g = data[i + 1];
          const b = data[i + 2];
          
          // In grayscale images, R, G, and B values are very similar
          const variation = Math.max(Math.abs(r - g), Math.abs(r - b), Math.abs(g - b));
          colorVariation += variation;
          sampledPixels++;
        }
        
        const avgColorVariation = colorVariation / sampledPixels;
        const aspectRatio = img.width / img.height;
        const minDimension = Math.min(img.width, img.height);
        
        // Typical X-ray images:
        // - Have low color variation (mostly grayscale)
        // - Have aspect ratios close to 1:1 or portrait
        // - Have sufficiently high resolution
        const isLikelyXRay = 
          avgColorVariation < 20 && // Low color variation
          aspectRatio >= 0.5 && aspectRatio <= 1.5 && // Reasonable aspect ratio
          minDimension >= 300; // Reasonable image size
        
        resolve(isLikelyXRay);
      };
      
      img.onerror = () => {
        resolve(false);
      };
      
      img.src = URL.createObjectURL(file);
    });
  };

  const handleImageUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      // Clear previous state
      setError("");
      setPrediction(null);
      
      // Validate file type
      if (!file.type.startsWith('image/')) {
        setError("Please upload an image file.");
        return;
      }
      
      // Create preview
      const tempPreviewUrl = URL.createObjectURL(file);
      setPreviewUrl(tempPreviewUrl);
      
      // Validate if it's likely an X-ray image
      const isXRay = await validateXRayImage(file);
      
      if (!isXRay) {
        setError("The uploaded image doesn't appear to be a chest X-ray. Please upload a valid chest X-ray image.");
        return;
      }
      
      // If validation passes, set the selected image
      setSelectedImage(file);
    }
  };

  const handleSubmit = async () => {
    if (!selectedImage) return;

    setLoading(true);
    const formData = new FormData();
    formData.append("file", selectedImage);

    try {
      console.log(`Sending request to ${API_URL}/predict`);
      
      const response = await fetch(`${API_URL}/predict`, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`API responded with status: ${response.status}`);
      }

      const data = await response.json();
      console.log("API response:", data);
      
      if (data.predictions && data.predictions.length > 0) {
        setPrediction({
          disease: data.predictions[0].class,
          confidence: data.predictions[0].probability
        });
      } else {
        setPrediction({
          disease: "No disease detected",
          confidence: 0
        });
      }
    } catch (error) {
      console.error("Error:", error);
      alert(`Error processing the image: ${error instanceof Error ? error.message : 'Unknown error'}`);
    } finally {
      setLoading(false);
    }
  };

  const toggleAbout = () => {
    setShowAbout(!showAbout);
  };

  if (showAbout) {
    return (
      <div className="app-container">
        <button onClick={toggleAbout} className="nav-button">
          Back to Predictor
        </button>
        <About />
      </div>
    );
  }

  return (
    <div className="app-container">
      <header className="app-header">
        <h1>Chest X-ray Disease Predictor</h1>
        <p>Upload your chest X-ray image to get a prediction</p>
        <button onClick={toggleAbout} className="nav-button">
          About the Project
        </button>
        {apiStatus && (
          <div className={`api-status ${apiStatus === "healthy" ? "healthy" : "error"}`}>
            API Status: {apiStatus}
          </div>
        )}
      </header>

      <main className="main-content">
        <div className="upload-section">
          <div className="image-preview">
            {previewUrl ? (
              <img src={previewUrl} alt="Preview" />
            ) : (
              <div className="placeholder">
                <p>No image selected</p>
              </div>
            )}
          </div>

          <div className="upload-controls">
            <input
              type="file"
              accept="image/*"
              onChange={handleImageUpload}
              className="file-input"
            />
            <button
              onClick={handleSubmit}
              disabled={!selectedImage || loading}
              className="predict-button"
            >
              {loading ? "Processing..." : "Predict Disease"}
            </button>
          </div>
          
          {error && (
            <div className="error-message">
              {error}
            </div>
          )}
        </div>

        {prediction && (
          <div className="results-section">
            <h2>Prediction Results</h2>
            <div className="prediction-card">
              <p className="disease-name">{prediction.disease}</p>
              <p className="confidence">
                Confidence: {(prediction.confidence * 100).toFixed(2)}%
              </p>
            </div>
          </div>
        )}

        {availableClasses.length > 0 && (
          <div className="available-classes">
            <h3>Detectable Diseases</h3>
            <ul>
              {availableClasses.map((className, index) => (
                <li key={index}>{className}</li>
              ))}
            </ul>
          </div>
        )}
      </main>
    </div>
  );
}

export default App;
