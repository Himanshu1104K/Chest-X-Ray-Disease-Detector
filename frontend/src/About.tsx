import React from 'react';
import './About.css';

const About: React.FC = () => {
  return (
    <div className="about-container">
      <h1>About Chest X-ray Disease Predictor</h1>
      
      <section className="about-section">
        <h2>Project Overview</h2>
        <p>
          The Chest X-ray Disease Predictor is an AI-powered application that helps medical professionals 
          and researchers analyze chest X-ray images to detect potential diseases. Using advanced machine 
          learning algorithms, the system provides quick and accurate predictions to assist in medical diagnosis.
        </p>
      </section>

      <section className="about-section">
        <h2>Features</h2>
        <ul>
          <li>Upload and analyze chest X-ray images</li>
          <li>Get instant disease predictions</li>
          <li>View confidence scores for predictions</li>
          <li>User-friendly interface</li>
          <li>Responsive design for all devices</li>
        </ul>
      </section>

      <section className="about-section">
        <h2>Technology Stack</h2>
        <ul>
          <li>Frontend: React with TypeScript</li>
          <li>Backend: Python with FastAPI</li>
          <li>Machine Learning: TensorFlow/PyTorch</li>
          <li>Styling: CSS with responsive design</li>
        </ul>
      </section>

      <section className="about-section">
        <h2>How It Works</h2>
        <ol>
          <li>Upload a chest X-ray image through the interface</li>
          <li>The system processes the image using trained ML models</li>
          <li>Get instant predictions with confidence scores</li>
          <li>View detailed results and analysis</li>
        </ol>
      </section>

      <section className="about-section">
        <h2>Disclaimer</h2>
        <p>
          This application is designed to assist medical professionals and should not be used as a 
          sole diagnostic tool. Always consult with qualified healthcare providers for medical advice 
          and diagnosis.
        </p>
      </section>
    </div>
  );
};

export default About; 