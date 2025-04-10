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

  const handleImageUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      setSelectedImage(file);
      setPreviewUrl(URL.createObjectURL(file));
      setPrediction(null);
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
