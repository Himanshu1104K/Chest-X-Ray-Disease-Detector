# Deep Learning Project

This project is a full-stack application that combines a deep learning model with a modern web interface. It consists of three main components: a frontend web application, a backend API server, and a deep learning model.

## Project Structure

```
.
├── backend/              # FastAPI backend server
│   ├── main.py          # Main API endpoints
│   ├── requirements.txt # Python dependencies
│   └── Model/           # Deep learning model implementation
├── frontend/            # React + TypeScript frontend
│   ├── src/            # Source code
│   ├── public/         # Static assets
│   └── package.json    # Node.js dependencies
└── Deep Learning Model/ # Deep learning model training and evaluation
```

## Technologies Used

### Backend
- FastAPI - Modern, fast web framework for building APIs
- PyTorch - Deep learning framework
- Uvicorn - ASGI server
- Python-multipart - For handling file uploads
- Pillow - Image processing
- Matplotlib - Data visualization

### Frontend
- React - JavaScript library for building user interfaces
- TypeScript - Typed JavaScript
- Vite - Next generation frontend tooling

## Setup Instructions

### Backend Setup
1. Navigate to the backend directory:
   ```bash
   cd backend
   ```

2. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Start the server:
   ```bash
   uvicorn main:app --reload
   ```

### Frontend Setup
1. Navigate to the frontend directory:
   ```bash
   cd frontend
   ```

2. Install dependencies:
   ```bash
   npm install
   ```

3. Start the development server:
   ```bash
   npm run dev
   ```

## API Documentation

The backend API documentation is available at `http://localhost:8000/docs` when the backend server is running.

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details. 