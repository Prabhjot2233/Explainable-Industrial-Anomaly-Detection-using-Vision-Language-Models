import React, { useEffect, useState } from 'react';
import axios from 'axios';

function App() {
  const [health, setHealth] = useState('Checking...');

  useEffect(() => {
    axios.get('http://localhost:8000/api/health')
      .then(response => {
        setHealth(response.data.status);
      })
      .catch(error => {
        console.error("Error fetching health check:", error);
        setHealth('Backend Unreachable');
      });
  }, []);

  return (
    <div style={{ padding: '20px', fontFamily: 'sans-serif' }}>
      <h1>Explainable Industrial Anomaly Detection</h1>
      <div style={{ padding: '10px', border: '1px solid #ccc', borderRadius: '4px' }}>
        <strong>Backend Status:</strong> {health}
      </div>
    </div>
  );
}

export default App;