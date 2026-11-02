import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import { 
  ShieldAlert, 
  ShieldCheck, 
  Scan, 
  Cpu, 
  Layers, 
  Eye, 
  Sparkles, 
  Upload, 
  RefreshCw, 
  ChevronRight, 
  Info, 
  Sliders, 
  Zap, 
  FileCheck2, 
  Crosshair,
  Maximize2
} from 'lucide-react';
import confetti from 'canvas-confetti';

const API_BASE = 'http://localhost:8000/api';

const DETECTORS = [
  { id: 'winclip', name: 'WinCLIP+ (Few-Shot)', k_shots: 2, badge: 'VLM Few-Shot', desc: 'Harmonic mean sliding windows with normal reference memory' },
  { id: 'winclip_zero', name: 'WinCLIP (Zero-Shot)', k_shots: 0, badge: 'Zero-Shot', desc: 'Compositional prompt ensemble without any reference images' },
  { id: 'anomalyclip', name: 'AnomalyCLIP', k_shots: 0, badge: 'Object-Agnostic', desc: 'Generic prompt offsets over ViT dense patch tokens' },
  { id: 'patchcore', name: 'PatchCore Baseline', k_shots: 4, badge: 'Non-VLM', desc: 'WideResNet-50 layer2+layer3 neighborhood coreset memory bank' }
];

const EXPLAINERS = [
  { id: 'abigx', name: 'ABIGX (AFR + IG)', badge: 'Counterfactual XAI', desc: 'Adversarial Fault Reconstruction baseline + Integrated Gradients' },
  { id: 'vit_gradcam', name: 'ViT Grad-CAM', badge: 'Attention Relevance', desc: 'Gradients of last transformer block mapped to 15x15 patch tokens' },
  { id: 'saliency', name: 'Input Saliency', badge: 'Pixel Gradients', desc: 'Direct ∂score/∂pixel sensitivity map via Captum' },
  { id: 'ig_black', name: 'Integrated Gradients', badge: 'Black Baseline', desc: 'Path integration from zero baseline satisfying completeness' },
  { id: 'ig_expected', name: 'Expected-IG', badge: 'Normal Baselines', desc: 'Path integration averaged across k normal reference images' }
];

const CATEGORIES = [
  { id: 'bottle', name: 'Bottle', icon: '🍾' },
  { id: 'carpet', name: 'Carpet', icon: '🧶' },
  { id: 'leather', name: 'Leather', icon: '👞' },
  { id: 'tile', name: 'Tile', icon: '🧱' },
  { id: 'transistor', name: 'Transistor', icon: '⚡' }
];

export default function App() {
  const [backendStatus, setBackendStatus] = useState({ online: false, checking: true, latency: 0 });
  const [samples, setSamples] = useState([]);
  const [selectedSample, setSelectedSample] = useState(null);
  const [uploadedFile, setUploadedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);

  const [category, setCategory] = useState('bottle');
  const [detector, setDetector] = useState('winclip');
  const [explainer, setExplainer] = useState('vit_gradcam');
  const [threshold, setThreshold] = useState(0.50);
  const [heatmapAlpha, setHeatmapAlpha] = useState(0.48);

  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [history, setHistory] = useState([]);
  const [activeTab, setActiveTab] = useState('inspection'); // 'inspection' | 'history' | 'about'

  const fileInputRef = useRef(null);

  // Check backend health
  const checkHealth = async () => {
    const start = performance.now();
    try {
      const res = await axios.get(`${API_BASE}/health`, { timeout: 3000 });
      const dt = Math.round(performance.now() - start);
      setBackendStatus({ online: res.data.status === 'healthy', checking: false, latency: dt });
    } catch (err) {
      setBackendStatus({ online: false, checking: false, latency: 0 });
    }
  };

  // Fetch sample items
  const fetchSamples = async () => {
    try {
      const res = await axios.get(`${API_BASE}/detection/samples`);
      if (res.data?.samples) {
        setSamples(res.data.samples);
        // Default to first defective sample
        if (res.data.samples.length > 0 && !selectedSample) {
          handleSelectSample(res.data.samples[0]);
        }
      }
    } catch (err) {
      console.warn("Could not load sample list:", err);
    }
  };

  useEffect(() => {
    checkHealth();
    fetchSamples();
    const interval = setInterval(checkHealth, 15000);
    return () => clearInterval(interval);
  }, []);

  const handleSelectSample = (sample) => {
    setSelectedSample(sample);
    setUploadedFile(null);
    setPreviewUrl(sample.image_url);
    setCategory(sample.category);
  };

  const handleFileUpload = (e) => {
    const file = e.target.files?.[0];
    if (!file) return;
    setUploadedFile(file);
    setSelectedSample(null);
    setPreviewUrl(URL.createObjectURL(file));
  };

  const handleDrop = (e) => {
    e.preventDefault();
    const file = e.dataTransfer.files?.[0];
    if (file && file.type.startsWith('image/')) {
      setUploadedFile(file);
      setSelectedSample(null);
      setPreviewUrl(URL.createObjectURL(file));
    }
  };

  const runAnalysis = async () => {
    if (!previewUrl && !uploadedFile && !selectedSample) return;
    setLoading(true);
    try {
      const formData = new FormData();
      if (uploadedFile) {
        formData.append('file', uploadedFile);
      } else if (selectedSample) {
        formData.append('sample_id', selectedSample.id);
      }

      const detObj = DETECTORS.find(d => d.id === detector);
      formData.append('category', category);
      formData.append('detector', detector === 'winclip_zero' ? 'winclip' : detector);
      formData.append('k_shots', detObj?.k_shots ?? 2);
      formData.append('explainer', explainer);

      const res = await axios.post(`${API_BASE}/detection/analyze`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });

      const data = res.data;
      setResult(data);
      setHistory(prev => [data, ...prev.slice(0, 7)]);

      if (!data.is_anomaly) {
        confetti({
          particleCount: 40,
          spread: 60,
          origin: { y: 0.8 },
          colors: ['#10b981', '#38bdf8', '#3b82f6']
        });
      }
    } catch (err) {
      console.error("Analysis failed:", err);
      alert("Analysis failed. Ensure the backend server is running on port 8000.");
    } finally {
      setLoading(false);
    }
  };

  // Auto-run on first sample load
  useEffect(() => {
    if (samples.length > 0 && !result && !loading) {
      runAnalysis();
    }
  }, [samples]);

  return (
    <div style={{ minHeight: '100vh', display: 'flex', flexDirection: 'column' }}>
      
      {/* Top Navbar */}
      <header style={{
        borderBottom: '1px solid var(--border-subtle)',
        background: 'rgba(8, 11, 18, 0.85)',
        backdropFilter: 'blur(16px)',
        position: 'sticky',
        top: 0,
        zIndex: 50,
        padding: '14px 28px',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between'
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '14px' }}>
          <div style={{
            width: '42px',
            height: '42px',
            borderRadius: '12px',
            background: 'linear-gradient(135deg, #0ea5e9, #6366f1)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            boxShadow: '0 0 20px rgba(14, 165, 233, 0.45)'
          }}>
            <Scan size={24} color="#ffffff" />
          </div>
          <div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
              <h1 style={{ fontSize: '18px', fontWeight: '700', letterSpacing: '-0.02em' }}>
                Explainable Vision-AI
              </h1>
              <span className="chip chip-active" style={{ fontSize: '11px', padding: '2px 8px' }}>
                Team 12 • COMP 8977
              </span>
            </div>
            <p style={{ fontSize: '12px', color: 'var(--text-muted)' }}>
              Industrial Defect Localization with Vision-Language Models & Pixel Attributions
            </p>
          </div>
        </div>

        {/* Right Status Badges */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
          <div style={{
            display: 'flex',
            alignItems: 'center',
            gap: '8px',
            padding: '6px 14px',
            borderRadius: 'var(--radius-full)',
            background: 'rgba(255, 255, 255, 0.04)',
            border: '1px solid var(--border-subtle)',
            fontSize: '12px'
          }}>
            <span style={{
              width: '8px',
              height: '8px',
              borderRadius: '50%',
              backgroundColor: backendStatus.online ? '#10b981' : '#f43f5e',
              boxShadow: backendStatus.online ? '0 0 8px #10b981' : '0 0 8px #f43f5e'
            }} />
            <span style={{ color: backendStatus.online ? 'var(--text-primary)' : '#f43f5e', fontWeight: 500 }}>
              {backendStatus.checking ? 'Connecting...' : backendStatus.online ? `FastAPI Online (${backendStatus.latency}ms)` : 'Backend Offline'}
            </span>
          </div>

          <div className="chip" style={{ background: 'rgba(56, 189, 248, 0.08)', borderColor: 'rgba(56, 189, 248, 0.25)' }}>
            <Cpu size={14} color="#38bdf8" />
            <span style={{ color: '#38bdf8', fontFamily: 'var(--font-mono)' }}>ViT-B-16-plus-240</span>
          </div>
        </div>
      </header>

      {/* Main Workspace Layout */}
      <main style={{ flex: 1, padding: '24px 28px', maxWidth: '1600px', margin: '0 auto', width: '100%' }}>
        
        {/* Preset Samples Bar */}
        <section style={{ marginBottom: '24px' }}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '12px' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
              <Sparkles size={16} color="#38bdf8" />
              <h2 style={{ fontSize: '14px', fontWeight: '600', textTransform: 'uppercase', letterSpacing: '0.05em', color: 'var(--text-secondary)' }}>
                Quick Test Demo Specimens (MVTec AD)
              </h2>
            </div>
            <span style={{ fontSize: '12px', color: 'var(--text-muted)' }}>
              Click any specimen to load and inspect immediately
            </span>
          </div>

          <div style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fit, minmax(160px, 1fr))',
            gap: '12px'
          }}>
            {samples.map((s) => {
              const isSelected = selectedSample?.id === s.id;
              const isDefect = s.type === 'defective';
              return (
                <div
                  key={s.id}
                  onClick={() => handleSelectSample(s)}
                  style={{
                    display: 'flex',
                    alignItems: 'center',
                    gap: '10px',
                    padding: '8px 10px',
                    borderRadius: 'var(--radius-sm)',
                    background: isSelected ? 'rgba(56, 189, 248, 0.12)' : 'var(--bg-card)',
                    border: `1px solid ${isSelected ? 'var(--accent-cyan)' : 'var(--border-subtle)'}`,
                    cursor: 'pointer',
                    transition: 'all 0.2s ease',
                    boxShadow: isSelected ? '0 0 15px rgba(56, 189, 248, 0.2)' : 'none'
                  }}
                >
                  <img
                    src={s.image_url}
                    alt={s.name}
                    style={{
                      width: '42px',
                      height: '42px',
                      borderRadius: '6px',
                      objectFit: 'cover',
                      border: '1px solid rgba(255, 255, 255, 0.1)'
                    }}
                  />
                  <div style={{ overflow: 'hidden' }}>
                    <div style={{
                      fontSize: '12px',
                      fontWeight: '600',
                      whiteSpace: 'nowrap',
                      textOverflow: 'ellipsis',
                      overflow: 'hidden'
                    }}>
                      {s.name}
                    </div>
                    <span style={{
                      fontSize: '10px',
                      fontWeight: '700',
                      textTransform: 'uppercase',
                      color: isDefect ? '#f43f5e' : '#10b981'
                    }}>
                      {isDefect ? '● Defective' : '● Normal'}
                    </span>
                  </div>
                </div>
              );
            })}
          </div>
        </section>

        {/* 2-Column Grid: Controls (Left) & Results (Right) */}
        <div style={{
          display: 'grid',
          gridTemplateColumns: '380px 1fr',
          gap: '24px',
          alignItems: 'start'
        }}>
          
          {/* Left Column: Inspection Configuration */}
          <aside className="glass-panel" style={{ padding: '22px' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '18px' }}>
              <Sliders size={18} color="#38bdf8" />
              <h3 style={{ fontSize: '15px', fontWeight: '600' }}>Inspection Parameters</h3>
            </div>

            {/* Input Selection Zone */}
            <div style={{ marginBottom: '20px' }}>
              <label style={{ fontSize: '12px', fontWeight: '600', color: 'var(--text-secondary)', display: 'block', marginBottom: '8px' }}>
                Target Specimen Image
              </label>
              
              <div
                onDragOver={(e) => e.preventDefault()}
                onDrop={handleDrop}
                onClick={() => fileInputRef.current?.click()}
                style={{
                  border: '2px dashed var(--border-medium)',
                  borderRadius: 'var(--radius-md)',
                  padding: '20px',
                  textAlign: 'center',
                  background: 'rgba(255, 255, 255, 0.02)',
                  cursor: 'pointer',
                  transition: 'all 0.2s ease',
                  position: 'relative'
                }}
              >
                <input
                  type="file"
                  ref={fileInputRef}
                  onChange={handleFileUpload}
                  accept="image/png, image/jpeg, image/webp"
                  style={{ display: 'none' }}
                />
                
                {previewUrl ? (
                  <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '10px' }}>
                    <img
                      src={previewUrl}
                      alt="Preview"
                      style={{
                        width: '120px',
                        height: '120px',
                        objectFit: 'cover',
                        borderRadius: '10px',
                        border: '1px solid var(--accent-cyan)',
                        boxShadow: '0 0 16px rgba(56, 189, 248, 0.25)'
                      }}
                    />
                    <span style={{ fontSize: '12px', color: 'var(--accent-cyan)', fontWeight: 500 }}>
                      {uploadedFile ? uploadedFile.name : selectedSample ? selectedSample.name : 'Selected'}
                    </span>
                    <span style={{ fontSize: '11px', color: 'var(--text-muted)' }}>Click or drop to replace</span>
                  </div>
                ) : (
                  <div>
                    <Upload size={28} color="#94a3b8" style={{ margin: '0 auto 8px' }} />
                    <p style={{ fontSize: '13px', fontWeight: 500 }}>Drag & drop image here</p>
                    <p style={{ fontSize: '11px', color: 'var(--text-muted)', marginTop: '4px' }}>PNG, JPG or WEBP (240x240)</p>
                  </div>
                )}
              </div>
            </div>

            {/* Category Selector */}
            <div style={{ marginBottom: '20px' }}>
              <label style={{ fontSize: '12px', fontWeight: '600', color: 'var(--text-secondary)', display: 'block', marginBottom: '8px' }}>
                Industrial Category
              </label>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '8px' }}>
                {CATEGORIES.map(c => (
                  <button
                    key={c.id}
                    type="button"
                    onClick={() => setCategory(c.id)}
                    className={`btn btn-secondary ${category === c.id ? 'chip-active' : ''}`}
                    style={{ fontSize: '12px', padding: '8px 10px', justifyContent: 'flex-start' }}
                  >
                    <span>{c.icon}</span>
                    <span>{c.name}</span>
                  </button>
                ))}
              </div>
            </div>

            {/* Model Architecture */}
            <div style={{ marginBottom: '20px' }}>
              <label style={{ fontSize: '12px', fontWeight: '600', color: 'var(--text-secondary)', display: 'block', marginBottom: '8px' }}>
                Detection Model (Week 3)
              </label>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
                {DETECTORS.map(d => {
                  const isSel = detector === d.id;
                  return (
                    <div
                      key={d.id}
                      onClick={() => setDetector(d.id)}
                      className={`selector-card ${isSel ? 'selected' : ''}`}
                    >
                      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                        <span style={{ fontSize: '13px', fontWeight: '600' }}>{d.name}</span>
                        <span className="chip" style={{ fontSize: '10px', padding: '2px 8px' }}>{d.badge}</span>
                      </div>
                      <p style={{ fontSize: '11px', color: 'var(--text-muted)', marginTop: '4px' }}>{d.desc}</p>
                    </div>
                  );
                })}
              </div>
            </div>

            {/* Visual Explainer Method */}
            <div style={{ marginBottom: '20px' }}>
              <label style={{ fontSize: '12px', fontWeight: '600', color: 'var(--text-secondary)', display: 'block', marginBottom: '8px' }}>
                Visual Explainer (Week 4)
              </label>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
                {EXPLAINERS.map(e => {
                  const isSel = explainer === e.id;
                  return (
                    <div
                      key={e.id}
                      onClick={() => setExplainer(e.id)}
                      className={`selector-card ${isSel ? 'selected' : ''}`}
                    >
                      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                        <span style={{ fontSize: '13px', fontWeight: '600' }}>{e.name}</span>
                        <span className="chip" style={{ fontSize: '10px', padding: '2px 8px' }}>{e.badge}</span>
                      </div>
                      <p style={{ fontSize: '11px', color: 'var(--text-muted)', marginTop: '4px' }}>{e.desc}</p>
                    </div>
                  );
                })}
              </div>
            </div>

            {/* Sensitivity Threshold Slider */}
            <div style={{ marginBottom: '24px' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '6px' }}>
                <label style={{ fontSize: '12px', fontWeight: '600', color: 'var(--text-secondary)' }}>
                  Decision Threshold (τ)
                </label>
                <span style={{ fontFamily: 'var(--font-mono)', fontSize: '12px', color: 'var(--accent-cyan)' }}>
                  {threshold.toFixed(2)}
                </span>
              </div>
              <input
                type="range"
                min="0.10"
                max="0.90"
                step="0.05"
                value={threshold}
                onChange={(e) => setThreshold(parseFloat(e.target.value))}
                style={{ width: '100%', accentColor: '#38bdf8', cursor: 'pointer' }}
              />
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '10px', color: 'var(--text-muted)', marginTop: '4px' }}>
                <span>Strict (0.10)</span>
                <span>Default (0.50)</span>
                <span>Permissive (0.90)</span>
              </div>
            </div>

            {/* Run Button */}
            <button
              type="button"
              onClick={runAnalysis}
              disabled={loading || !backendStatus.online}
              className="btn btn-primary"
              style={{ width: '100%', padding: '14px', fontSize: '15px' }}
            >
              {loading ? (
                <>
                  <RefreshCw className="animate-spin" size={18} />
                  <span>Computing Multi-Scale Gradients...</span>
                </>
              ) : (
                <>
                  <Zap size={18} />
                  <span>Run Anomaly Inspection</span>
                </>
              )}
            </button>
          </aside>

          {/* Right Column: Visual Inspection Results */}
          <section style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
            
            {result ? (
              <>
                {/* Big Verdict Header Card */}
                {(() => {
                  const isAnomaly = result.anomaly_score >= threshold;
                  return (
                    <div
                      className="glass-panel"
                      style={{
                        padding: '24px',
                        borderLeft: `6px solid ${isAnomaly ? 'var(--accent-rose)' : 'var(--accent-emerald)'}`,
                        boxShadow: isAnomaly ? 'var(--shadow-glow-rose)' : 'var(--shadow-glow-emerald)',
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'space-between',
                        flexWrap: 'wrap',
                        gap: '16px'
                      }}
                    >
                      <div style={{ display: 'flex', alignItems: 'center', gap: '18px' }}>
                        <div style={{
                          width: '56px',
                          height: '56px',
                          borderRadius: '16px',
                          background: isAnomaly ? 'rgba(244, 63, 94, 0.15)' : 'rgba(16, 185, 129, 0.15)',
                          display: 'flex',
                          alignItems: 'center',
                          justifyContent: 'center',
                          border: `1px solid ${isAnomaly ? 'rgba(244, 63, 94, 0.4)' : 'rgba(16, 185, 129, 0.4)'}`
                        }}>
                          {isAnomaly ? <ShieldAlert size={32} color="#f43f5e" /> : <ShieldCheck size={32} color="#10b981" />}
                        </div>
                        <div>
                          <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                            <span style={{
                              fontSize: '22px',
                              fontWeight: '800',
                              letterSpacing: '-0.02em',
                              color: isAnomaly ? '#f43f5e' : '#10b981'
                            }}>
                              {isAnomaly ? 'DEFECT DETECTED — REJECT' : 'NOMINAL SURFACE — PASS'}
                            </span>
                            <span className="chip" style={{ fontFamily: 'var(--font-mono)' }}>
                              τ = {threshold.toFixed(2)}
                            </span>
                          </div>
                          <p style={{ fontSize: '13px', color: 'var(--text-secondary)', marginTop: '4px' }}>
                            {result.narrative}
                          </p>
                        </div>
                      </div>

                      {/* Score Gauge */}
                      <div style={{ minWidth: '220px', display: 'flex', flexDirection: 'column', gap: '6px' }}>
                        <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '12px' }}>
                          <span style={{ color: 'var(--text-secondary)' }}>Anomaly Probability</span>
                          <span style={{ fontFamily: 'var(--font-mono)', fontWeight: '700', color: isAnomaly ? '#f43f5e' : '#10b981' }}>
                            {(result.anomaly_score * 100).toFixed(1)}%
                          </span>
                        </div>
                        <div className="gauge-container">
                          <div
                            className={`gauge-fill ${isAnomaly ? 'gauge-fill-anomaly' : 'gauge-fill-normal'}`}
                            style={{ width: `${Math.min(100, Math.max(3, result.anomaly_score * 100))}%` }}
                          />
                        </div>
                        <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '11px', color: 'var(--text-muted)' }}>
                          <span>Confidence: {(result.confidence * 100).toFixed(1)}%</span>
                          <span>Latency: {result.runtime_ms}ms</span>
                        </div>
                      </div>
                    </div>
                  );
                })()}

                {/* Heatmap Overlay Alpha Slider */}
                <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: '0 4px' }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                    <Layers size={16} color="#38bdf8" />
                    <span style={{ fontSize: '13px', fontWeight: '600' }}>Multi-Panel Visual Explanations</span>
                  </div>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
                    <span style={{ fontSize: '12px', color: 'var(--text-muted)' }}>Heatmap Blend (Alpha):</span>
                    <input
                      type="range"
                      min="0.1"
                      max="0.9"
                      step="0.05"
                      value={heatmapAlpha}
                      onChange={(e) => setHeatmapAlpha(parseFloat(e.target.value))}
                      style={{ width: '100px', accentColor: '#38bdf8' }}
                    />
                    <span style={{ fontFamily: 'var(--font-mono)', fontSize: '12px', color: 'var(--accent-cyan)' }}>
                      {heatmapAlpha.toFixed(2)}
                    </span>
                  </div>
                </div>

                {/* Visual Panels Grid */}
                <div className="preview-grid">
                  
                  {/* Panel 1: Original Image */}
                  <div className="preview-card">
                    <div className="preview-card-img-wrap">
                      <img src={result.visuals.original} alt="Original specimen" className="preview-card-img" />
                    </div>
                    <div className="preview-card-label">
                      <span>1. Original Specimen (RGB)</span>
                      <span className="chip" style={{ fontSize: '10px' }}>240×240</span>
                    </div>
                  </div>

                  {/* Panel 2: Detector Anomaly Heatmap */}
                  <div className="preview-card">
                    <div className="preview-card-img-wrap" style={{ position: 'relative' }}>
                      <img src={result.visuals.original} alt="Original base" className="preview-card-img" />
                      <img
                        src={result.visuals.heatmap_overlay}
                        alt="Anomaly Heatmap"
                        className="preview-card-img"
                        style={{ position: 'absolute', inset: 0, opacity: heatmapAlpha }}
                      />
                    </div>
                    <div className="preview-card-label">
                      <span>2. Detector Anomaly Map</span>
                      <span className="chip chip-active" style={{ fontSize: '10px' }}>{result.detector.toUpperCase()}</span>
                    </div>
                  </div>

                  {/* Panel 3: Explainer Attribution Map */}
                  <div className="preview-card">
                    <div className="preview-card-img-wrap" style={{ position: 'relative' }}>
                      <img src={result.visuals.original} alt="Original base" className="preview-card-img" />
                      <img
                        src={result.visuals.attribution_overlay}
                        alt="Attribution Map"
                        className="preview-card-img"
                        style={{ position: 'absolute', inset: 0, opacity: heatmapAlpha }}
                      />
                    </div>
                    <div className="preview-card-label">
                      <span>3. Visual Attribution Overlay</span>
                      <span className="chip" style={{ fontSize: '10px', color: '#8b5cf6', borderColor: 'rgba(139, 92, 246, 0.3)' }}>
                        {result.explainer}
                      </span>
                    </div>
                  </div>

                  {/* Panel 4: Standalone Heatmap (Raw) */}
                  <div className="preview-card">
                    <div className="preview-card-img-wrap">
                      <img src={result.visuals.attribution_standalone} alt="Pure Heatmap" className="preview-card-img" />
                    </div>
                    <div className="preview-card-label">
                      <span>4. Jet Colormap (Raw Attribution)</span>
                      <span className="chip" style={{ fontSize: '10px' }}>[0.0, 1.0]</span>
                    </div>
                  </div>

                </div>

                {/* Hotspot & Telemetry Breakdown */}
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 340px', gap: '20px' }}>
                  
                  {/* Hotspots Card */}
                  <div className="glass-panel" style={{ padding: '20px' }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '14px' }}>
                      <Crosshair size={18} color="#f43f5e" />
                      <h4 style={{ fontSize: '14px', fontWeight: '600' }}>
                        Detected Defect Hotspots & Localization
                      </h4>
                    </div>

                    {result.hotspots && result.hotspots.length > 0 ? (
                      <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
                        {result.hotspots.map((h, idx) => (
                          <div
                            key={h.region_id}
                            style={{
                              display: 'flex',
                              alignItems: 'center',
                              justifyContent: 'space-between',
                              padding: '10px 14px',
                              borderRadius: 'var(--radius-sm)',
                              background: 'rgba(255, 255, 255, 0.03)',
                              border: '1px solid var(--border-subtle)'
                            }}
                          >
                            <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
                              <span style={{
                                width: '24px',
                                height: '24px',
                                borderRadius: '50%',
                                background: 'rgba(244, 63, 94, 0.2)',
                                color: '#f43f5e',
                                display: 'flex',
                                alignItems: 'center',
                                justifyContent: 'center',
                                fontSize: '12px',
                                fontWeight: '700'
                              }}>
                                {idx + 1}
                              </span>
                              <div>
                                <div style={{ fontSize: '13px', fontWeight: '600' }}>
                                  Region at ({h.x}, {h.y})
                                </div>
                                <div style={{ fontSize: '11px', color: 'var(--text-muted)' }}>
                                  Bounding Box: {h.width}×{h.height}px • Center: ({(h.relative_center[0]*100).toFixed(0)}%, {(h.relative_center[1]*100).toFixed(0)}%)
                                </div>
                              </div>
                            </div>
                            <div style={{ textAlign: 'right' }}>
                              <span style={{ fontSize: '13px', fontWeight: '700', color: '#f43f5e', fontFamily: 'var(--font-mono)' }}>
                                {(h.peak_anomaly * 100).toFixed(1)}%
                              </span>
                              <div style={{ fontSize: '10px', color: 'var(--text-muted)' }}>Peak Intensity</div>
                            </div>
                          </div>
                        ))}
                      </div>
                    ) : (
                      <div style={{ padding: '20px', textAlign: 'center', color: 'var(--text-muted)', fontSize: '13px' }}>
                        No anomalous hotspots detected exceeding spatial clustering thresholds. Surface is uniform.
                      </div>
                    )}
                  </div>

                  {/* Technical Telemetry Card */}
                  <div className="glass-panel" style={{ padding: '20px' }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '14px' }}>
                      <FileCheck2 size={18} color="#38bdf8" />
                      <h4 style={{ fontSize: '14px', fontWeight: '600' }}>Inspection Telemetry</h4>
                    </div>

                    <div style={{ display: 'flex', flexDirection: 'column', gap: '12px', fontSize: '13px' }}>
                      <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                        <span style={{ color: 'var(--text-secondary)' }}>Model Backbone:</span>
                        <span style={{ fontFamily: 'var(--font-mono)', fontWeight: '600' }}>ViT-B-16-plus-240</span>
                      </div>
                      <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                        <span style={{ color: 'var(--text-secondary)' }}>Explainer Method:</span>
                        <span style={{ color: 'var(--accent-cyan)', fontWeight: '600' }}>{result.explainer_name}</span>
                      </div>
                      <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                        <span style={{ color: 'var(--text-secondary)' }}>Inference Hardware:</span>
                        <span style={{ fontFamily: 'var(--font-mono)' }}>{result.device}</span>
                      </div>
                      <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                        <span style={{ color: 'var(--text-secondary)' }}>Reference Images (k):</span>
                        <span style={{ fontFamily: 'var(--font-mono)' }}>{result.k_shots} shots</span>
                      </div>
                      <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                        <span style={{ color: 'var(--text-secondary)' }}>Execution Latency:</span>
                        <span style={{ fontFamily: 'var(--font-mono)', color: '#10b981', fontWeight: '600' }}>{result.runtime_ms} ms</span>
                      </div>
                    </div>
                  </div>

                </div>

              </>
            ) : (
              <div
                className="glass-panel"
                style={{
                  padding: '60px 40px',
                  textAlign: 'center',
                  display: 'flex',
                  flexDirection: 'column',
                  alignItems: 'center',
                  gap: '16px'
                }}
              >
                <Scan size={48} color="#38bdf8" />
                <h3 style={{ fontSize: '18px', fontWeight: '700' }}>Ready for Industrial Inspection</h3>
                <p style={{ maxWidth: '480px', color: 'var(--text-secondary)', fontSize: '14px' }}>
                  Select one of the demo specimens above or upload your own industrial image on the left, then click <strong>Run Anomaly Inspection</strong>.
                </p>
              </div>
            )}

          </section>

        </div>

      </main>

      {/* Footer */}
      <footer style={{
        marginTop: 'auto',
        borderTop: '1px solid var(--border-subtle)',
        padding: '16px 28px',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        fontSize: '12px',
        color: 'var(--text-muted)'
      }}>
        <div>
          COMP 8977 • Explainable Industrial Anomaly Detection using Vision-Language Models
        </div>
        <div>
          Prabhjot Singh & Sarhan Kapadiya • OpenCLIP & Captum
        </div>
      </footer>

    </div>
  );
}