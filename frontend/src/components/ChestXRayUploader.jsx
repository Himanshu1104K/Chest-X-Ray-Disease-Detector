import React, { useState } from 'react';
import axios from 'axios';
import { Box, Button, Typography, CircularProgress, Paper, List, ListItem, ListItemText } from '@mui/material';
import CloudUploadIcon from '@mui/icons-material/CloudUpload';

const ChestXRayUploader = () => {
  const [file, setFile] = useState(null);
  const [predictions, setPredictions] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleFileChange = (event) => {
    setFile(event.target.files[0]);
    setPredictions([]);
    setError(null);
  };

  const handleUpload = async () => {
    if (!file) {
      setError('Please select a file first');
      return;
    }

    setLoading(true);
    setError(null);

    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await axios.post('http://localhost:8000/predict', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      setPredictions(response.data.predictions);
    } catch (err) {
      setError(err.response?.data?.detail || 'An error occurred while processing the image');
    } finally {
      setLoading(false);
    }
  };

  return (
    <Box sx={{ maxWidth: 600, mx: 'auto', p: 3 }}>
      <Paper elevation={3} sx={{ p: 3 }}>
        <Typography variant="h5" gutterBottom>
          Chest X-Ray Classification
        </Typography>

        <Box sx={{ my: 2 }}>
          <input
            accept="image/*"
            style={{ display: 'none' }}
            id="raised-button-file"
            type="file"
            onChange={handleFileChange}
          />
          <label htmlFor="raised-button-file">
            <Button
              variant="contained"
              component="span"
              startIcon={<CloudUploadIcon />}
              sx={{ mr: 2 }}
            >
              Select Image
            </Button>
          </label>
          <Button
            variant="contained"
            color="primary"
            onClick={handleUpload}
            disabled={!file || loading}
          >
            {loading ? <CircularProgress size={24} /> : 'Upload and Predict'}
          </Button>
        </Box>

        {file && (
          <Box sx={{ my: 2 }}>
            <Typography variant="subtitle1">Selected file: {file.name}</Typography>
          </Box>
        )}

        {error && (
          <Typography color="error" sx={{ my: 2 }}>
            {error}
          </Typography>
        )}

        {predictions.length > 0 && (
          <Box sx={{ my: 2 }}>
            <Typography variant="h6" gutterBottom>
              Predictions:
            </Typography>
            <List>
              {predictions.map((prediction, index) => (
                <ListItem key={index}>
                  <ListItemText
                    primary={prediction.class}
                    secondary={`Probability: ${(prediction.probability * 100).toFixed(2)}%`}
                  />
                </ListItem>
              ))}
            </List>
          </Box>
        )}
      </Paper>
    </Box>
  );
};

export default ChestXRayUploader; 