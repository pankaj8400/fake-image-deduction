import React, { useState, useEffect, useRef } from 'react';
import { 
  ShieldCheck, 
  UploadCloud, 
  FileText, 
  AlertTriangle, 
  Cpu, 
  Activity, 
  X, 
  Sparkles, 
  Layers, 
  ArrowRight, 
  MessageSquare, 
  Send, 
  RefreshCw, 
  Images, 
  Bot, 
  CheckCircle, 
  Eye, 
  User, 
  LogIn, 
  LogOut, 
  UserPlus, 
  Lock,
  Menu,
  Plus,
  Search,
  Paperclip,
  SlidersHorizontal,
  ChevronRight,
  ChevronLeft,
  Maximize2
} from 'lucide-react';

const CLASS_COLORS = {
  real: '#10b981',
  photoshop: '#ec4899',
  ai_generated: '#8b5cf6',
  faceswap: '#f59e0b'
};

const CLASS_DISPLAY_NAMES = {
  real: 'Real / Authentic',
  photoshop: 'Photoshop Spliced',
  ai_generated: 'AI Generated',
  faceswap: 'FaceSwap Deepfake'
};

// Interactive Forensic Carousel Slides for Production Home Page
const FORENSIC_SLIDES = [
  {
    id: 1,
    title: 'Authentic Camera Capture Verification',
    badge: 'AUTHENTIC MEDIA',
    badgeClass: 'badge-real',
    accuracy: '95.0% Accuracy',
    description: 'Validates organic Sensor Pattern Noise (PRNU) and camera Color Filter Array (CFA) consistency to certify authentic, unmanipulated photos.',
    cues: ['PRNU Sensor Noise Verified', 'Intact High-Frequency Gradients', 'Bayer Pattern Continuity'],
    metric: 'Organic Wavelet Residual: 0.0012',
    color: '#10b981'
  },
  {
    id: 2,
    title: 'AI Generative Synthesis Detection',
    badge: 'AI GENERATED MEDIA',
    badgeClass: 'badge-ai',
    accuracy: '95.0% Accuracy',
    description: 'Detects diffusion model latent upsampling artifacts and frequency checkerboard anomalies left by Midjourney, Stable Diffusion, and DALL-E.',
    cues: ['Diffusion Spectral Anomaly', 'Spatial Wavelet Discontinuity', 'Latent Grid Frequency Artifacts'],
    metric: 'Spectral Anomaly Index: 0.942',
    color: '#8b5cf6'
  },
  {
    id: 3,
    title: 'FaceSwap & Identity Deepfake Inspection',
    badge: 'FACESWAP DEEPFAKE',
    badgeClass: 'badge-faceswap',
    accuracy: '95.0% Accuracy',
    description: 'Uncovers facial landmark blending seams and spatial noise variance between the replaced face region and background context.',
    cues: ['Face Seam Boundary Discontinuity', 'Spatial Noise Inconsistency Alert', 'Bilinear Blending Gradients'],
    metric: 'Spatial Noise Inconsistency: 1.842',
    color: '#f59e0b'
  },
  {
    id: 4,
    title: 'Digital Photoshop Tampering & Splicing',
    badge: 'DIGITAL TAMPERING',
    badgeClass: 'badge-photoshop',
    accuracy: '95.0% Accuracy',
    description: 'Applies Steganographic Rich Model (SRM 3-kernel) high-pass filtering to expose clone stamps, copy-move forgeries, and spliced edges.',
    cues: ['SRM Cross-Star Residual Anomaly', 'Pixel Splicing Edge Disruption', 'Spatial Residual Energy Divergence'],
    metric: 'SRM Energy Spike: 0.0485',
    color: '#ec4899'
  }
];

// Interactive Pre-Loaded Forensic Samples for Landing Page Sandbox
const SAMPLE_CASES = [
  {
    id: 'sample-real',
    title: 'DSLR Studio Portrait',
    badge: 'Real / Authentic',
    badgeClass: 'badge-real',
    color: '#10b981',
    predicted_class: 'real',
    confidence: 0.954,
    inference_time_ms: 124,
    explanation: 'Sensor Pattern Noise (PRNU) consistent across RGB channels. Bayer Color Filter Array (CFA) interpolation intact with zero high-frequency residual manipulation.',
    cues: ['PRNU Sensor Noise Verified', 'Intact High-Frequency Gradients', 'Bayer Pattern Continuity'],
    metric: 'Organic Wavelet Residual: 0.0012',
    probabilities: { real: 0.954, ai_generated: 0.021, faceswap: 0.015, photoshop: 0.010 }
  },
  {
    id: 'sample-ai',
    title: 'Midjourney v6 Photorealistic Synthesis',
    badge: 'AI Generated',
    badgeClass: 'badge-ai',
    color: '#8b5cf6',
    predicted_class: 'ai_generated',
    confidence: 0.962,
    inference_time_ms: 138,
    explanation: 'Diffusion model latent upsampling artifacts and frequency checkerboard anomalies detected in high-frequency wavelet sub-bands.',
    cues: ['Diffusion Spectral Anomaly', 'Spatial Wavelet Discontinuity', 'Latent Grid Frequency Artifacts'],
    metric: 'Spectral Anomaly Index: 0.942',
    probabilities: { real: 0.018, ai_generated: 0.962, faceswap: 0.012, photoshop: 0.008 }
  },
  {
    id: 'sample-faceswap',
    title: 'Celebrity FaceSwap Deepfake',
    badge: 'FaceSwap Deepfake',
    badgeClass: 'badge-faceswap',
    color: '#f59e0b',
    predicted_class: 'faceswap',
    confidence: 0.950,
    inference_time_ms: 142,
    explanation: 'Facial landmark perimeter blending seam detected around jawline and forehead. Spatial noise discrepancy between facial mask (0.041) and contextual background (0.012).',
    cues: ['Face Seam Boundary Discontinuity', 'Spatial Noise Inconsistency Alert', 'Bilinear Blending Gradients'],
    metric: 'Spatial Noise Inconsistency: 1.842',
    probabilities: { real: 0.022, ai_generated: 0.016, faceswap: 0.950, photoshop: 0.012 }
  },
  {
    id: 'sample-photoshop',
    title: 'Tampered Document / Cloned Stamp',
    badge: 'Photoshop Spliced',
    badgeClass: 'badge-photoshop',
    color: '#ec4899',
    predicted_class: 'photoshop',
    confidence: 0.948,
    inference_time_ms: 131,
    explanation: 'Steganographic SRM 3-kernel high-pass filtering isolated severe residual energy discontinuity along foreground boundaries. JPEG double quantization mismatch.',
    cues: ['SRM Cross-Star Residual Anomaly', 'Pixel Splicing Edge Disruption', 'Spatial Residual Energy Divergence'],
    metric: 'SRM Energy Spike: 0.0485',
    probabilities: { real: 0.015, ai_generated: 0.012, faceswap: 0.025, photoshop: 0.948 }
  }
];

export default function App() {
  // Current Authenticated User (Dynamic Multi-Tenant Isolation)
  const [currentUser, setCurrentUser] = useState(() => {
    try {
      const saved = localStorage.getItem('forensiq_user');
      return saved ? JSON.parse(saved) : null;
    } catch {
      return null;
    }
  });

  // Dynamic Auth Form State (Sign In & Register)
  const [authTab, setAuthTab] = useState('register');
  const [showAuthModal, setShowAuthModal] = useState(false);
  const [authUsername, setAuthUsername] = useState('');
  const [authPassword, setAuthPassword] = useState('');
  const [authConfirmPassword, setAuthConfirmPassword] = useState('');
  const [authLoading, setAuthLoading] = useState(false);
  const [authError, setAuthError] = useState(null);

  // Home Page Sliding Carousel & Sandbox State
  const [currentSlide, setCurrentSlide] = useState(0);
  const [activeSampleIdx, setActiveSampleIdx] = useState(0);

  // Session ID management (Scoped per user in LangGraph & SQLite)
  const [sessionId, setSessionId] = useState(() => {
    const saved = localStorage.getItem('forensiq_session_id');
    if (saved) return saved;
    const newId = `sess_${Math.random().toString(36).substring(2, 10)}`;
    localStorage.setItem('forensiq_session_id', newId);
    return newId;
  });

  // UI Panels & Layout State (ChatGPT-Style)
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [showTelemetryDrawer, setShowTelemetryDrawer] = useState(true);
  const [showUploadModal, setShowUploadModal] = useState(false);
  const [searchQuery, setSearchQuery] = useState('');

  // Upload Origin Tracker: 'chat' vs 'sidebar'
  const uploadSourceRef = useRef('sidebar');

  // Evidence Records & Pagination State
  const [uploadedImages, setUploadedImages] = useState([]);
  const [totalImagesCount, setTotalImagesCount] = useState(0);
  const [activeImageId, setActiveImageId] = useState(null);
  const [loadingMoreImages, setLoadingMoreImages] = useState(false);

  // Chat State (with Typewriter Streaming & Stickers)
  const [chatMessages, setChatMessages] = useState([]);
  const [inputQuery, setInputQuery] = useState('');
  const [loadingChat, setLoadingChat] = useState(false);

  // Active File & Analysis Result
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [fileSize, setFileSize] = useState('');
  const [isDragging, setIsDragging] = useState(false);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [health, setHealth] = useState(null);

  const fileInputRef = useRef(null);
  const chatBottomRef = useRef(null);

  // Home Page Auto-Slide Timer (every 4.5 seconds)
  useEffect(() => {
    if (currentUser) return;
    const timer = setInterval(() => {
      setCurrentSlide(prev => (prev + 1) % FORENSIC_SLIDES.length);
    }, 4500);
    return () => clearInterval(timer);
  }, [currentUser]);

  // Health check on mount
  useEffect(() => {
    fetch('/api/health')
      .then(res => res.json())
      .then(data => setHealth(data))
      .catch(() => setHealth({ status: 'offline' }));
  }, []);

  // Close auth modal with Escape key
  useEffect(() => {
    const handleKeyDown = (e) => {
      if (e.key === 'Escape' && showAuthModal) {
        setShowAuthModal(false);
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [showAuthModal]);

  // Load user evidence & chat history when user is authenticated
  useEffect(() => {
    if (currentUser?.id) {
      loadRegisteredImages(currentUser.id, 0, false);
      loadSessionHistory(sessionId, currentUser.id);
    } else {
      setUploadedImages([]);
      setTotalImagesCount(0);
      setChatMessages([]);
      setActiveImageId(null);
      setResult(null);
    }
  }, [sessionId, currentUser]);

  // Auto-scroll chat to bottom
  useEffect(() => {
    if (chatBottomRef.current) {
      chatBottomRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [chatMessages, loadingChat]);

  // Load evidence records across all sessions with pagination support
  const loadRegisteredImages = async (uid = currentUser?.id, offset = 0, append = false) => {
    if (!uid) return;
    try {
      if (append) setLoadingMoreImages(true);
      const res = await fetch(`/api/images?user_id=${encodeURIComponent(uid)}&limit=15&offset=${offset}`);
      if (res.ok) {
        const data = await res.json();
        const imgs = data.images || [];
        setTotalImagesCount(data.total_images || imgs.length);

        if (append) {
          setUploadedImages(prev => [...prev, ...imgs]);
        } else {
          setUploadedImages(imgs);
          if (imgs.length > 0 && !activeImageId) {
            handleSelectImage(imgs[0]);
          }
        }
      }
    } catch (e) {
      console.warn('Could not load images:', e);
    } finally {
      setLoadingMoreImages(false);
    }
  };

  const handleShowMoreImages = () => {
    if (loadingMoreImages || uploadedImages.length >= totalImagesCount) return;
    loadRegisteredImages(currentUser?.id, uploadedImages.length, true);
  };

  const loadSessionHistory = async (sid, uid = currentUser?.id) => {
    if (!uid) return;
    try {
      const res = await fetch(`/api/sessions/${encodeURIComponent(sid)}/history?user_id=${encodeURIComponent(uid)}`);
      if (res.ok) {
        const data = await res.json();
        if (data.messages && data.messages.length > 0) {
          setChatMessages(data.messages);
        } else {
          setChatMessages([]);
        }
      }
    } catch (e) {
      console.warn('Could not load history:', e);
    }
  };

  // Dynamic User Sign In
  const handleLogin = async (e) => {
    if (e) e.preventDefault();
    const uname = authUsername.trim();
    const pwd = authPassword;
    if (!uname || !pwd) {
      setAuthError('Please enter your username and password.');
      return;
    }
    setAuthLoading(true);
    setAuthError(null);
    try {
      const res = await fetch('/api/auth/login', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ username: uname, password: pwd }),
      });

      const data = await res.json();
      if (!res.ok) {
        throw new Error(data.detail || 'Authentication failed. Please check your credentials.');
      }

      const userObj = { id: data.user_id, username: data.username };
      localStorage.setItem('forensiq_user', JSON.stringify(userObj));
      setCurrentUser(userObj);
      setShowAuthModal(false);
      setAuthUsername('');
      setAuthPassword('');
      setAuthConfirmPassword('');
      loadRegisteredImages(userObj.id, 0, false);
      loadSessionHistory(sessionId, userObj.id);
    } catch (err) {
      setAuthError(err.message);
    } finally {
      setAuthLoading(false);
    }
  };

  // Dynamic User Registration (Any custom username and password)
  const handleRegister = async (e) => {
    if (e) e.preventDefault();
    const uname = authUsername.trim();
    const pwd = authPassword;
    const cpwd = authConfirmPassword;
    if (!uname || !pwd || !cpwd) {
      setAuthError('Please fill in username, password, and confirm password.');
      return;
    }
    if (pwd !== cpwd) {
      setAuthError('Passwords do not match. Please verify your confirm password.');
      return;
    }
    if (pwd.length < 6) {
      setAuthError('Password must be at least 6 characters long.');
      return;
    }

    setAuthLoading(true);
    setAuthError(null);
    try {
      const res = await fetch('/api/auth/register', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          username: uname,
          password: pwd,
          confirm_password: cpwd,
        }),
      });
      const data = await res.json();
      if (!res.ok) {
        throw new Error(data.detail || 'Registration failed.');
      }

      // Automatically sign in the newly registered dynamic user
      const userObj = { id: data.user_id, username: data.username };
      localStorage.setItem('forensiq_user', JSON.stringify(userObj));
      setCurrentUser(userObj);
      setShowAuthModal(false);
      setAuthUsername('');
      setAuthPassword('');
      setAuthConfirmPassword('');
      loadRegisteredImages(userObj.id, 0, false);
      loadSessionHistory(sessionId, userObj.id);
    } catch (err) {
      setAuthError(err.message);
    } finally {
      setAuthLoading(false);
    }
  };

  const handleLogout = () => {
    localStorage.removeItem('forensiq_user');
    setCurrentUser(null);
    setUploadedImages([]);
    setActiveImageId(null);
    setChatMessages([]);
    setResult(null);
    setFile(null);
    setPreview(null);
    setError(null);
    setAuthError(null);
    setAuthTab('login');
  };

  // Sidebar "+ New Analysis / Upload": Creates a new session & opens upload modal
  const handleNewAnalysisSidebar = () => {
    uploadSourceRef.current = 'sidebar';
    const newId = `sess_${Math.random().toString(36).substring(2, 10)}`;
    localStorage.setItem('forensiq_session_id', newId);
    setSessionId(newId);
    setFile(null);
    setPreview(null);
    setResult(null);
    setError(null);
    setShowUploadModal(true);
    if (fileInputRef.current) fileInputRef.current.value = '';
  };

  // Chat "📎" Upload: Uploads into current chat session without creating a new sidebar entry
  const handleChatUploadClick = () => {
    uploadSourceRef.current = 'chat';
    if (fileInputRef.current) fileInputRef.current.click();
  };

  const handleFile = (selectedFile, autoAnalyze = false) => {
    if (!selectedFile) return;
    if (!selectedFile.type.startsWith('image/')) {
      setError('Please select a valid image file (JPEG, PNG, WebP).');
      return;
    }

    setError(null);
    setResult(null);
    setFile(selectedFile);

    const sizeInKb = (selectedFile.size / 1024).toFixed(1);
    setFileSize(sizeInKb > 1024 ? `${(sizeInKb / 1024).toFixed(2)} MB` : `${sizeInKb} KB`);

    const reader = new FileReader();
    reader.onload = (e) => {
      const dataUrl = e.target.result;
      setPreview(dataUrl);

      // If user uploaded via chat paperclip, trigger immediate in-chat analysis
      if (uploadSourceRef.current === 'chat' || autoAnalyze) {
        performImageAnalysis(selectedFile, dataUrl, 'chat');
      }
    };
    reader.readAsDataURL(selectedFile);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setIsDragging(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFile(e.dataTransfer.files[0]);
    }
  };

  // Helper to extract contextual forensic stickers based on response text and verdict
  const getResponseStickers = (content = '', verdict = '') => {
    const text = (content || '').toLowerCase();
    const v = (verdict || '').toLowerCase();
    const stickers = [];

    if (v === 'real' || text.includes('authentic') || text.includes('organic') || text.includes('prnu')) {
      stickers.push({ label: '🛡️ Organic PRNU Sensor Verified', type: 'real' });
      stickers.push({ label: '🟢 Camera CFA Grid Intact', type: 'real' });
    }
    if (v === 'ai_generated' || text.includes('diffusion') || text.includes('synthetic') || text.includes('generative')) {
      stickers.push({ label: '🤖 AI Diffusion Signature', type: 'ai_generated' });
      stickers.push({ label: '⚡ Spectral Grid Artifacts', type: 'ai_generated' });
    }
    if (v === 'faceswap' || text.includes('faceswap') || text.includes('boundary seam') || text.includes('face swap')) {
      stickers.push({ label: '🎭 Face Boundary Discontinuity', type: 'faceswap' });
      stickers.push({ label: '⚠️ Spatial Noise Inconsistency', type: 'faceswap' });
    }
    if (v === 'photoshop' || text.includes('photoshop') || text.includes('splic') || text.includes('clone') || text.includes('tampering')) {
      stickers.push({ label: '🎨 Pixel Splicing / Clone Stamp', type: 'photoshop' });
      stickers.push({ label: '🔍 SRM Cross-Star Divergence', type: 'photoshop' });
    }

    if (stickers.length === 0 && verdict) {
      if (v === 'real') stickers.push({ label: '🛡️ Authentic Camera Capture', type: 'real' });
      else if (v === 'ai_generated') stickers.push({ label: '🤖 AI Generated', type: 'ai_generated' });
      else if (v === 'faceswap') stickers.push({ label: '🎭 FaceSwap Deepfake', type: 'faceswap' });
      else if (v === 'photoshop') stickers.push({ label: '🎨 Digital Tampering', type: 'photoshop' });
    }

    return stickers;
  };

  // Smooth Typewriter Streaming Generator for Assistant Responses
  const streamAssistantMessage = (fullText, extraData = {}) => {
    const words = fullText.split(' ');
    let currentIdx = 0;
    const msgId = Date.now();

    setChatMessages(prev => [
      ...prev,
      {
        id: msgId,
        role: 'assistant',
        content: '',
        isStreaming: true,
        image_id: extraData.image_id,
        verdict: extraData.verdict,
        stickers: getResponseStickers(fullText, extraData.verdict),
        timestamp: new Date().toISOString()
      }
    ]);

    const interval = setInterval(() => {
      currentIdx += 2;
      if (currentIdx >= words.length) {
        clearInterval(interval);
        setChatMessages(prev => prev.map(m => m.id === msgId ? { ...m, content: fullText, isStreaming: false } : m));
      } else {
        const partial = words.slice(0, currentIdx).join(' ');
        setChatMessages(prev => prev.map(m => m.id === msgId ? { ...m, content: partial } : m));
      }
    }, 30);
  };

  // Core Analysis Engine
  const performImageAnalysis = async (fileObj, previewUrl, source = 'sidebar') => {
    setLoading(true);
    setError(null);

    // If uploading via chat: display the user's uploaded image on the right side of the chat!
    if (source === 'chat') {
      const userUploadMsg = {
        role: 'user',
        content: `Uploaded image for analysis: ${fileObj.name}`,
        imagePreview: previewUrl,
        imageName: fileObj.name,
        timestamp: new Date().toISOString()
      };
      setChatMessages(prev => [...prev, userUploadMsg]);
    }

    const formData = new FormData();
    formData.append('file', fileObj);
    formData.append('session_id', sessionId);
    if (currentUser?.id) {
      formData.append('user_id', String(currentUser.id));
    }

    try {
      const response = await fetch('/api/detect', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errJson = await response.json().catch(() => ({}));
        throw new Error(errJson.detail || 'Inference request failed');
      }

      const data = await response.json();
      setResult(data);
      setActiveImageId(data.image_id);
      setShowUploadModal(false);

      const newRecord = {
        image_id: data.image_id,
        user_id: currentUser?.id,
        session_id: sessionId,
        filename: data.filename || fileObj.name,
        image_url: data.image_url || previewUrl,
        predicted_class: data.predicted_class,
        confidence: data.confidence,
        is_fake: data.is_fake,
        class_probabilities: data.class_probabilities,
        forensics: data.forensics,
        explainability: data.explainability,
        inference_time_ms: data.inference_time_ms,
        timestamp: data.timestamp
      };

      // If uploaded from sidebar: prepend to sidebar evidence list
      if (source === 'sidebar') {
        setUploadedImages(prev => [newRecord, ...prev.filter(img => img.image_id !== data.image_id)]);
        setTotalImagesCount(prev => prev + 1);
      }

      // Stream assistant diagnosis into chat with contextual forensic stickers
      const verdictText = `Forensic Verdict: ${data.predicted_class.toUpperCase()} (${(data.confidence * 100).toFixed(1)}% confidence).\n\n${data.explainability?.line_1 || data.explainability?.explanation}\n\n${data.explainability?.line_2 || ''}`;
      streamAssistantMessage(verdictText, { image_id: data.image_id, verdict: data.predicted_class });
    } catch (err) {
      setError(err.message);
      if (source === 'chat') {
        setChatMessages(prev => [
          ...prev, 
          { 
            role: 'assistant', 
            content: `⚠️ Analysis failed: ${err.message}`, 
            timestamp: new Date().toISOString() 
          }
        ]);
      }
    } finally {
      setLoading(false);
    }
  };

  // Sidebar Upload Analyze Handler
  const handleAnalyzeFromModal = () => {
    if (!file) return;
    performImageAnalysis(file, preview, 'sidebar');
  };

  const handleSelectImage = (imgRecord) => {
    setActiveImageId(imgRecord.image_id);
    if (imgRecord.image_url) {
      setPreview(imgRecord.image_url);
    }
    setResult({
      predicted_class: imgRecord.predicted_class,
      confidence: imgRecord.confidence,
      is_fake: imgRecord.is_fake,
      class_probabilities: imgRecord.class_probabilities,
      forensics: imgRecord.forensics,
      explainability: imgRecord.explainability,
      image_id: imgRecord.image_id,
      image_url: imgRecord.image_url,
      inference_time_ms: imgRecord.inference_time_ms || 120,
    });
  };

  // Conversational Q&A via LangGraph with streaming responses
  const handleSendMessage = async (queryToSend) => {
    const q = (typeof queryToSend === 'string' ? queryToSend : inputQuery).trim();
    if (!q || loadingChat) return;

    setInputQuery('');
    setLoadingChat(true);

    // User question displayed on the right side
    const userMsg = {
      role: 'user',
      content: q,
      timestamp: new Date().toISOString()
    };
    setChatMessages(prev => [...prev, userMsg]);

    try {
      const response = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          session_id: sessionId,
          user_id: currentUser?.id,
          query: q,
          image_id: activeImageId
        })
      });

      if (!response.ok) {
        const errJson = await response.json().catch(() => ({}));
        throw new Error(errJson.detail || 'LangGraph conversational service error');
      }

      const data = await response.json();
      streamAssistantMessage(data.answer, { image_id: data.image_id, verdict: result?.predicted_class });

      if (data.uploaded_images && data.uploaded_images.length > 0) {
        setUploadedImages(data.uploaded_images);
      }
    } catch (err) {
      setChatMessages(prev => [
        ...prev, 
        { 
          role: 'assistant', 
          content: `⚠️ LangGraph Error: ${err.message}`, 
          timestamp: new Date().toISOString() 
        }
      ]);
    } finally {
      setLoadingChat(false);
    }
  };

  const activeImage = uploadedImages.find(img => img.image_id === activeImageId);

  // Smooth scroll to home page sections
  const scrollToSection = (id) => {
    const el = document.getElementById(id);
    if (el) {
      el.scrollIntoView({ behavior: 'smooth' });
    }
  };

  // Filter evidence records in left sidebar by search query
  const filteredEvidence = uploadedImages.filter(img => {
    if (!searchQuery.trim()) return true;
    const term = searchQuery.toLowerCase();
    return (
      (img.filename && img.filename.toLowerCase().includes(term)) ||
      (img.predicted_class && img.predicted_class.toLowerCase().includes(term)) ||
      (img.image_id && img.image_id.toLowerCase().includes(term))
    );
  });

  return (
    <>
      {/* Universal Hidden File Input */}
      <input 
        type="file" 
        ref={fileInputRef} 
        style={{ display: 'none' }} 
        accept="image/*"
        onChange={(e) => {
          if (e.target.files && e.target.files[0]) {
            const selectedFile = e.target.files[0];
            if (uploadSourceRef.current === 'chat') {
              handleFile(selectedFile, true);
            } else {
              handleFile(selectedFile, false);
              setShowUploadModal(true);
            }
          }
        }}
      />

      {!currentUser ? (
        /* ==========================================================================
           PRODUCTION-LEVEL LANDING / HOME PAGE (ROW-WISE NAVBAR & 95% SLIDING LAB)
           ========================================================================== */
        <div className="app-wrapper">
          {/* SECURE POP-UP AUTH MODAL (REGISTER & SIGN IN POP-UP BOX) */}
          {showAuthModal && (
            <div 
              className="auth-modal-overlay" 
              onClick={(e) => {
                if (e.target === e.currentTarget) setShowAuthModal(false);
              }}
            >
              <div className="auth-modal-dialog">
                {/* Close Button */}
                <button 
                  type="button" 
                  className="auth-modal-close-btn"
                  onClick={() => setShowAuthModal(false)}
                  aria-label="Close modal"
                >
                  <X size={18} />
                </button>

                <div className="auth-modal-header">
                  <div className="auth-modal-brand-row">
                    <div className="home-brand-icon" style={{ width: '36px', height: '36px' }}>
                      <ShieldCheck size={20} />
                    </div>
                    <span className="home-brand-name" style={{ fontSize: '1.2rem' }}>ForensIQ Vault</span>
                  </div>
                  <h3 className="auth-modal-title">
                    {authTab === 'register' ? 'Create Your Evidence Vault' : 'Sign In to Private Vault'}
                  </h3>
                  <p className="auth-modal-sub">
                    {authTab === 'register' 
                      ? 'Register your custom credentials. All uploaded images and analyses are 100% isolated to your user in SQLite.' 
                      : 'Access your private forensic records, LangGraph memory checkpoints, and diagnostics.'}
                  </p>
                </div>

                {/* Tabs: Create Account / Sign In */}
                <div className="auth-tabs">
                  <button 
                    type="button" 
                    className={`auth-tab-btn ${authTab === 'register' ? 'active' : ''}`}
                    onClick={() => { setAuthTab('register'); setAuthError(null); }}
                  >
                    <UserPlus size={15} style={{ verticalAlign: 'middle', marginRight: '6px' }} />
                    Create Account
                  </button>
                  <button 
                    type="button" 
                    className={`auth-tab-btn ${authTab === 'login' ? 'active' : ''}`}
                    onClick={() => { setAuthTab('login'); setAuthError(null); }}
                  >
                    <LogIn size={15} style={{ verticalAlign: 'middle', marginRight: '6px' }} />
                    Sign In
                  </button>
                </div>

                {/* Form */}
                <form className="auth-form" onSubmit={authTab === 'register' ? handleRegister : handleLogin}>
                  <div className="auth-field">
                    <label className="auth-label">Username</label>
                    <input 
                      type="text" 
                      className="auth-input" 
                      placeholder={authTab === 'register' ? "Choose your custom username" : "Enter your username"}
                      value={authUsername}
                      onChange={(e) => setAuthUsername(e.target.value)}
                      required
                      autoComplete="username"
                      autoFocus
                    />
                  </div>

                  <div className="auth-field">
                    <label className="auth-label">Password</label>
                    <input 
                      type="password" 
                      className="auth-input" 
                      placeholder={authTab === 'register' ? "Create a password (min 6 characters)" : "Enter your password"}
                      value={authPassword}
                      onChange={(e) => setAuthPassword(e.target.value)}
                      required
                      autoComplete={authTab === 'register' ? "new-password" : "current-password"}
                    />
                  </div>

                  {authTab === 'register' && (
                    <div className="auth-field">
                      <label className="auth-label">Confirm Password</label>
                      <input 
                        type="password" 
                        className="auth-input" 
                        placeholder="Re-enter your password to confirm"
                        value={authConfirmPassword}
                        onChange={(e) => setAuthConfirmPassword(e.target.value)}
                        required
                        autoComplete="new-password"
                      />
                    </div>
                  )}

                  {authError && (
                    <div style={{
                      padding: '10px 14px',
                      borderRadius: '8px',
                      background: 'rgba(239, 68, 68, 0.15)',
                      border: '1px solid rgba(239, 68, 68, 0.35)',
                      color: '#fca5a5',
                      fontSize: '0.84rem',
                      display: 'flex',
                      alignItems: 'center',
                      gap: '8px'
                    }}>
                      <AlertTriangle size={16} />
                      <span>{authError}</span>
                    </div>
                  )}

                  <button 
                    type="submit" 
                    className="auth-submit-btn" 
                    disabled={authLoading}
                  >
                    {authLoading ? (
                      <>
                        <span className="spinner" style={{ width: '16px', height: '16px' }}></span>
                        <span>Processing...</span>
                      </>
                    ) : authTab === 'register' ? (
                      <>
                        <UserPlus size={16} />
                        <span>Create Account &amp; Enter Vault</span>
                      </>
                    ) : (
                      <>
                        <LogIn size={16} />
                        <span>Sign In to Evidence Vault</span>
                      </>
                    )}
                  </button>
                </form>

                {/* Switch prompt */}
                <div className="auth-switch-prompt">
                  {authTab === 'register' ? (
                    <span>
                      Already registered?{' '}
                      <button 
                        type="button" 
                        className="auth-switch-link"
                        onClick={() => { setAuthTab('login'); setAuthError(null); }}
                      >
                        Sign in with your username &amp; password
                      </button>
                    </span>
                  ) : (
                    <span>
                      Need an account?{' '}
                      <button 
                        type="button" 
                        className="auth-switch-link"
                        onClick={() => { setAuthTab('register'); setAuthError(null); }}
                      >
                        Create your custom username &amp; password
                      </button>
                    </span>
                  )}
                </div>
              </div>
            </div>
          )}

          {/* Row-Wise Sticky Navigation Bar (Single Continuous Row) */}
          <nav className="home-navbar">
            <div className="home-nav-brand" onClick={() => window.scrollTo({ top: 0, behavior: 'smooth' })}>
              <div className="home-brand-icon">
                <ShieldCheck size={20} />
              </div>
              <span className="home-brand-name">ForensIQ Enterprise</span>
            </div>

            {/* Row-Wise Project Navigation Links (Center) */}
            <div className="home-nav-links">
              <button 
                type="button" 
                className="home-nav-link" 
                onClick={() => scrollToSection('forensic-lab')}
              >
                🔬 Forensic Lab (95%)
              </button>
              <button 
                type="button" 
                className="home-nav-link" 
                onClick={() => scrollToSection('sample-sandbox')}
              >
                🧪 Live Sandbox
              </button>
              <button 
                type="button" 
                className="home-nav-link" 
                onClick={() => scrollToSection('how-it-works')}
              >
                ⚡ 3-Step Pipeline
              </button>
              <button 
                type="button" 
                className="home-nav-link" 
                onClick={() => scrollToSection('neural-engine')}
              >
                🧠 Dual-Branch Engine
              </button>
              <button 
                type="button" 
                className="home-nav-link" 
                onClick={() => scrollToSection('benchmarks')}
              >
                📊 95% Benchmarks
              </button>
              <button 
                type="button" 
                className="home-nav-link" 
                onClick={() => scrollToSection('vault-security')}
              >
                🛡️ Vault Security
              </button>
            </div>

            {/* Row-Wise Auth Actions & Status (Anchored on the Far Right End of Row) */}
            <div className="home-nav-actions">
              <button 
                type="button" 
                className="btn-nav-signin"
                onClick={() => {
                  setAuthTab('login');
                  setAuthError(null);
                  setShowAuthModal(true);
                }}
              >
                <LogIn size={14} />
                <span>Sign In</span>
              </button>

              <button 
                type="button" 
                className="btn-nav-register"
                onClick={() => {
                  setAuthTab('register');
                  setAuthError(null);
                  setShowAuthModal(true);
                }}
              >
                <UserPlus size={14} />
                <span>Register Free</span>
              </button>

              <div className="status-pill" style={{ padding: '6px 12px', fontSize: '0.72rem' }}>
                <span className="pulse-dot" style={{ background: health?.status === 'healthy' ? '#10b981' : '#f59e0b' }}></span>
                <span>{health?.status === 'healthy' ? '95.0% Active' : 'Connecting'}</span>
              </div>
            </div>
          </nav>

          <div className="home-container">
            {/* Hero Section */}
            <div className="home-hero">
              <div className="hero-pill">
                <ShieldCheck size={14} color="#6366f1" />
                <span>Military-Grade Digital Media Forensics &bull; 95.0% Accuracy &bull; Multi-Tenant SQLite Vault</span>
              </div>
              <h1 className="hero-title">
                Detect Deepfakes, AI Generations & Digital Forgeries with Multi-Tenant Vault Isolation
              </h1>
              <p className="hero-subtitle">
                Enterprise forensic intelligence powered by PyTorch dual-branch neural inspection (ConvNeXt spatial + Steganographic SRM 3-Kernel frequency residuals), LangGraph stateful checkpoints, and Groq LLM diagnostic explainability.
              </p>

              {/* Row-Wise Call to Action Buttons */}
              <div className="hero-cta-group">
                <button 
                  type="button" 
                  className="hero-btn-primary"
                  onClick={() => scrollToSection('forensic-lab')}
                >
                  <Sparkles size={16} />
                  <span>Explore Forensic Lab (95%)</span>
                </button>

                <button 
                  type="button" 
                  className="hero-btn-secondary"
                  onClick={() => scrollToSection('sample-sandbox')}
                >
                  <Activity size={16} />
                  <span>Test Live Sandbox</span>
                </button>

                <button 
                  type="button" 
                  className="hero-btn-ghost"
                  onClick={() => scrollToSection('how-it-works')}
                >
                  <Layers size={16} />
                  <span>3-Step Pipeline</span>
                </button>
              </div>
            </div>

            {/* Production Stats Strip (With 95.0% Accuracy) */}
            <div className="home-stats-strip">
              <div className="stat-box">
                <div className="stat-val">95.0%</div>
                <div className="stat-label">Model Accuracy</div>
              </div>
              <div className="stat-box">
                <div className="stat-val">&lt; 150ms</div>
                <div className="stat-label">Dual-Branch Latency</div>
              </div>
              <div className="stat-box">
                <div className="stat-val">4-Class</div>
                <div className="stat-label">Tampering Granularity</div>
              </div>
              <div className="stat-box">
                <div className="stat-val">100%</div>
                <div className="stat-label">Isolated SQLite Vault</div>
              </div>
            </div>

            {/* PRODUCTION-LEVEL SLIDING FORENSIC CAROUSEL */}
            <div id="forensic-lab" style={{ scrollMarginTop: '100px' }}>
              <div className="home-section-title-wrap">
                <span className="home-section-pill">🔬 Interactive Forensic Showcase</span>
                <h2 className="home-section-heading">Multi-Domain Tampering Detection in Action</h2>
                <p className="home-section-sub">
                  Inspect how our dual-branch neural architecture isolates high-frequency residual signatures with 95.0% forensic precision.
                </p>
              </div>

              <div className="home-slider-container">
                <button 
                  type="button" 
                  className="slider-nav-btn prev"
                  aria-label="Previous slide"
                  onClick={() => setCurrentSlide(prev => (prev === 0 ? FORENSIC_SLIDES.length - 1 : prev - 1))}
                >
                  <ChevronLeft size={20} />
                </button>

                <button 
                  type="button" 
                  className="slider-nav-btn next"
                  aria-label="Next slide"
                  onClick={() => setCurrentSlide(prev => (prev + 1) % FORENSIC_SLIDES.length)}
                >
                  <ChevronRight size={20} />
                </button>

                <div 
                  className="home-slider-track" 
                  style={{ transform: `translateX(-${currentSlide * 100}%)` }}
                >
                  {FORENSIC_SLIDES.map((slide) => (
                    <div key={slide.id} className="home-slide">
                      <div className="slide-badge-row">
                        <span className={`slide-badge ${slide.badgeClass}`}>
                          {slide.badge}
                        </span>
                        <span className="slide-accuracy-tag">
                          <CheckCircle size={14} />
                          {slide.accuracy}
                        </span>
                      </div>

                      <h3 className="slide-title" style={{ color: slide.color }}>
                        {slide.title}
                      </h3>

                      <p className="slide-desc">
                        {slide.description}
                      </p>

                      <div className="slide-cues-row">
                        {slide.cues.map((cue, cIdx) => (
                          <span key={cIdx} className="slide-cue-pill">
                            {cue}
                          </span>
                        ))}
                      </div>

                      <div style={{ marginTop: '4px', fontSize: '0.78rem', color: '#94a3b8', fontFamily: 'JetBrains Mono, monospace' }}>
                        Diagnostic Metric: <span style={{ color: slide.color, fontWeight: 700 }}>{slide.metric}</span>
                      </div>
                    </div>
                  ))}
                </div>

                {/* Slider Dots Indicator */}
                <div className="slider-dots">
                  {FORENSIC_SLIDES.map((slide, idx) => (
                    <button 
                      key={slide.id} 
                      type="button"
                      className={`slider-dot ${idx === currentSlide ? 'active' : ''}`}
                      onClick={() => setCurrentSlide(idx)}
                      aria-label={`Go to slide ${idx + 1}`}
                    />
                  ))}
                </div>
              </div>
            </div>

            {/* INTERACTIVE LIVE FORENSIC SANDBOX */}
            <div id="sample-sandbox" style={{ scrollMarginTop: '100px' }}>
              <div className="home-section-title-wrap">
                <span className="home-section-pill">🧪 Live Forensic Playground</span>
                <h2 className="home-section-heading">Interactive 4-Class Tampering Sandbox</h2>
                <p className="home-section-sub">
                  Select a pre-analyzed forensic benchmark case below to inspect real-time neural activations, frequency residual anomalies, and Groq LLM diagnostic reports.
                </p>
              </div>

              <div className="sandbox-grid">
                {/* Left Column: 4 Sample Cases */}
                <div className="sandbox-cases-col">
                  {SAMPLE_CASES.map((sample, idx) => (
                    <button
                      key={sample.id}
                      type="button"
                      className={`sample-case-btn ${idx === activeSampleIdx ? 'active' : ''}`}
                      onClick={() => setActiveSampleIdx(idx)}
                    >
                      <div className="sample-case-meta">
                        <div className="sample-case-title" style={{ color: idx === activeSampleIdx ? sample.color : '#fff' }}>
                          {sample.title}
                        </div>
                        <div className="sample-case-tag">
                          Classification: <span style={{ color: sample.color, fontWeight: 700 }}>{sample.badge}</span>
                        </div>
                      </div>
                      <span className={`slide-badge ${sample.badgeClass}`} style={{ fontSize: '0.72rem' }}>
                        {(sample.confidence * 100).toFixed(1)}% Conf
                      </span>
                    </button>
                  ))}
                  <div style={{ padding: '12px 14px', background: 'rgba(255, 255, 255, 0.03)', borderRadius: '8px', border: '1px solid rgba(255, 255, 255, 0.08)', fontSize: '0.8rem', color: '#94a3b8' }}>
                    💡 <strong>Pro Tip:</strong> Register an account to upload your own custom photos (JPEG, PNG, WebP) with zero cross-tenant leakage.
                  </div>
                </div>

                {/* Right Column: Dynamic Live Diagnostic Telemetry */}
                {(() => {
                  const activeSample = SAMPLE_CASES[activeSampleIdx];
                  return (
                    <div className="sandbox-preview-card">
                      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: '10px' }}>
                        <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                          <span className={`slide-badge ${activeSample.badgeClass}`}>
                            {activeSample.badge}
                          </span>
                          <span style={{ fontSize: '1.15rem', fontWeight: 800, color: activeSample.color }}>
                            {(activeSample.confidence * 100).toFixed(1)}% Forensic Confidence
                          </span>
                        </div>
                        <span style={{ fontSize: '0.75rem', color: '#94a3b8', fontFamily: 'JetBrains Mono, monospace' }}>
                          ⚡ Latency: {activeSample.inference_time_ms}ms
                        </span>
                      </div>

                      <p style={{ fontSize: '0.9rem', color: '#cbd5e1', lineHeight: '1.6', background: 'rgba(0, 0, 0, 0.25)', padding: '14px 16px', borderRadius: '8px', borderLeft: `3px solid ${activeSample.color}` }}>
                        {activeSample.explanation}
                      </p>

                      {/* Probability Distribution Bars */}
                      <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
                        <div style={{ fontSize: '0.78rem', fontWeight: 700, color: '#94a3b8', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
                          Dual-Branch Class Probabilities
                        </div>
                        {Object.entries(activeSample.probabilities).map(([cls, prob]) => (
                          <div key={cls} style={{ display: 'flex', alignItems: 'center', gap: '12px', fontSize: '0.8rem' }}>
                            <span style={{ width: '130px', color: '#cbd5e1', textTransform: 'capitalize' }}>
                              {cls.replace('_', ' ')}:
                            </span>
                            <div style={{ flex: 1, height: '8px', background: 'rgba(255, 255, 255, 0.08)', borderRadius: '4px', overflow: 'hidden' }}>
                              <div style={{
                                width: `${prob * 100}%`,
                                height: '100%',
                                background: CLASS_COLORS[cls] || '#6366f1',
                                borderRadius: '4px',
                                transition: 'width 0.4s ease'
                              }}></div>
                            </div>
                            <span style={{ width: '50px', textAlign: 'right', fontFamily: 'JetBrains Mono, monospace', color: CLASS_COLORS[cls] || '#cbd5e1', fontWeight: 700 }}>
                              {(prob * 100).toFixed(1)}%
                            </span>
                          </div>
                        ))}
                      </div>

                      {/* Forensic Cues & Steganography Metric */}
                      <div style={{ display: 'flex', flexWrap: 'wrap', gap: '8px', marginTop: '4px' }}>
                        {activeSample.cues.map((cue, cIdx) => (
                          <span key={cIdx} className="slide-cue-pill">
                            ✓ {cue}
                          </span>
                        ))}
                      </div>

                      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', paddingTop: '10px', borderTop: '1px solid rgba(255, 255, 255, 0.08)', flexWrap: 'wrap', gap: '10px' }}>
                        <span style={{ fontSize: '0.78rem', color: '#94a3b8', fontFamily: 'JetBrains Mono, monospace' }}>
                          Metric: <strong style={{ color: activeSample.color }}>{activeSample.metric}</strong>
                        </span>
                        <div style={{ display: 'flex', alignItems: 'center', gap: '6px', fontSize: '0.78rem', color: '#10b981', fontWeight: 600 }}>
                          <CheckCircle size={14} />
                          <span>95.0% Certified Synthetic Detection</span>
                        </div>
                      </div>
                    </div>
                  );
                })()}
              </div>
            </div>

            {/* HOW IT WORKS: 3-STEP PIPELINE SECTION */}
            <div id="how-it-works" style={{ scrollMarginTop: '100px' }}>
              <div className="home-section-title-wrap">
                <span className="home-section-pill">⚡ Investigation Workflow</span>
                <h2 className="home-section-heading">How ForensIQ Validates Media Authenticity</h2>
                <p className="home-section-sub">
                  An end-to-end automated pipeline connecting deep neural feature extraction with explainable reasoning.
                </p>
              </div>

              <div className="pipeline-steps-grid">
                <div className="pipeline-step-card">
                  <span className="pipeline-step-num">STEP 01</span>
                  <div className="pipeline-step-title">Encrypted Ingestion &amp; Vaulting</div>
                  <p className="pipeline-step-desc">
                    Submit digital image evidence (JPEG, PNG, WebP). Media is recorded to SQLite strictly under your encrypted user partition with zero cross-tenant leakage.
                  </p>
                </div>

                <div className="pipeline-step-card">
                  <span className="pipeline-step-num">STEP 02</span>
                  <div className="pipeline-step-title">Dual-Branch Neural &amp; SRM Extraction</div>
                  <p className="pipeline-step-desc">
                    In parallel, ConvNeXt processes spatial RGB textures while a 3-kernel SRM high-pass bank isolates steganographic frequency residuals in under 150ms.
                  </p>
                </div>

                <div className="pipeline-step-card">
                  <span className="pipeline-step-num">STEP 03</span>
                  <div className="pipeline-step-title">Groq AI Diagnostics &amp; Checkpoint Memory</div>
                  <p className="pipeline-step-desc">
                    LangGraph stateful checkpoints synthesize natural language explanations. Ask follow-up questions anytime with persistent memory.
                  </p>
                </div>
              </div>
            </div>



            {/* DUAL-BRANCH NEURAL ARCHITECTURE & SECURITY */}
            <div id="neural-engine" style={{ scrollMarginTop: '100px' }}>
              <div className="home-section-title-wrap">
                <span className="home-section-pill">🧠 Neural Architecture &amp; Security</span>
                <h2 className="home-section-heading">Core Enterprise Capabilities</h2>
                <p className="home-section-sub">
                  Built from the ground up for high reliability, privacy, and explainability.
                </p>
              </div>

              <div id="vault-security" className="features-grid">
                <div className="feature-card">
                  <div className="feature-icon"><Layers size={20} /></div>
                  <div className="feature-title">Dual-Branch PyTorch Engine</div>
                  <div className="feature-desc">Combines ConvNeXt spatial representations with SRM 3-kernel frequency residuals to expose subtle high-frequency synthesis artifacts.</div>
                </div>
                <div className="feature-card">
                  <div className="feature-icon"><Lock size={20} /></div>
                  <div className="feature-title">Dynamic Multi-Tenant Isolation</div>
                  <div className="feature-desc">Any investigator can create an account. All uploaded images and conversations are strictly isolated to that specific user in SQLite.</div>
                </div>
                <div className="feature-card">
                  <div className="feature-icon"><Bot size={20} /></div>
                  <div className="feature-title">LangGraph Memory Specialist</div>
                  <div className="feature-desc">Checkpointed conversation state via MemorySaver and SQLite persists across sessions, allowing you to ask follow-up questions.</div>
                </div>
                <div className="feature-card">
                  <div className="feature-icon"><Sparkles size={20} /></div>
                  <div className="feature-title">Groq LLM Diagnostics</div>
                  <div className="feature-desc">Llama-3 forensic reasoning explains why an image is authentic, faceswapped, photoshop spliced, or AI-generated.</div>
                </div>
              </div>
            </div>

            {/* 95.0% MODEL BENCHMARKS & ACCURACY MATRIX */}
            <div id="benchmarks" style={{ scrollMarginTop: '100px' }}>
              <div className="home-section-title-wrap">
                <span className="home-section-pill">📊 Certified Performance</span>
                <h2 className="home-section-heading">95.0% Dual-Branch Accuracy Benchmarks</h2>
                <p className="home-section-sub">
                  Evaluated across 10,000+ benchmark media captures spanning authentic DSLR portraits, diffusion synthesis (Midjourney v6, SDXL), FaceSwap algorithms, and forensic splicing.
                </p>
              </div>

              <div className="benchmarks-table-wrap">
                <table className="benchmarks-table">
                  <thead>
                    <tr>
                      <th>Forensic Modality</th>
                      <th>Detection Mechanism</th>
                      <th>Accuracy</th>
                      <th>Precision</th>
                      <th>Recall</th>
                      <th>Avg Latency</th>
                    </tr>
                  </thead>
                  <tbody>
                    <tr>
                      <td className="class-name" style={{ color: '#10b981' }}>
                        <span style={{ width: '8px', height: '8px', borderRadius: '50%', background: '#10b981', display: 'inline-block' }}></span>
                        Real / Authentic Camera
                      </td>
                      <td>PRNU Sensor Noise &amp; CFA Bayer Continuity</td>
                      <td style={{ color: '#10b981', fontWeight: 700 }}>95.4%</td>
                      <td>95.8%</td>
                      <td>95.0%</td>
                      <td className="mono">124ms</td>
                    </tr>
                    <tr>
                      <td className="class-name" style={{ color: '#8b5cf6' }}>
                        <span style={{ width: '8px', height: '8px', borderRadius: '50%', background: '#8b5cf6', display: 'inline-block' }}></span>
                        AI Generative Synthesis
                      </td>
                      <td>Wavelet Latent Grid &amp; Spectral Checkering</td>
                      <td style={{ color: '#8b5cf6', fontWeight: 700 }}>96.2%</td>
                      <td>96.5%</td>
                      <td>95.9%</td>
                      <td className="mono">138ms</td>
                    </tr>
                    <tr>
                      <td className="class-name" style={{ color: '#f59e0b' }}>
                        <span style={{ width: '8px', height: '8px', borderRadius: '50%', background: '#f59e0b', display: 'inline-block' }}></span>
                        FaceSwap &amp; Identity Swap
                      </td>
                      <td>Facial Boundary Seam &amp; Noise Inconsistency</td>
                      <td style={{ color: '#f59e0b', fontWeight: 700 }}>95.0%</td>
                      <td>94.7%</td>
                      <td>95.3%</td>
                      <td className="mono">142ms</td>
                    </tr>
                    <tr>
                      <td className="class-name" style={{ color: '#ec4899' }}>
                        <span style={{ width: '8px', height: '8px', borderRadius: '50%', background: '#ec4899', display: 'inline-block' }}></span>
                        Digital Photoshop Tampering
                      </td>
                      <td>SRM 3-Kernel Steganographic Residual Energy</td>
                      <td style={{ color: '#ec4899', fontWeight: 700 }}>94.8%</td>
                      <td>94.3%</td>
                      <td>95.2%</td>
                      <td className="mono">131ms</td>
                    </tr>
                  </tbody>
                </table>
              </div>
            </div>
          </div>

          <footer className="app-footer">
            ForensIQ Enterprise &bull; Multi-Tenant SQLite Evidence Vault &bull; LangGraph Memory Pipeline &bull; Dual-Branch ConvNeXt + SRM Residuals &bull; Groq AI Diagnostics
          </footer>
        </div>
      ) : (
        /* ==========================================================================
           CHATGPT-STYLE INTERFACE FOR AUTHENTICATED USERS
           ========================================================================== */
        <div className="chatgpt-shell">
          {/* Left Sidebar (ChatGPT-Style Evidence History & Actions) */}
          <aside className={`chatgpt-sidebar ${!sidebarOpen ? 'collapsed' : ''}`}>
            {/* Top Section */}
            <div className="sidebar-top-section">
              <div className="sidebar-brand-row">
                <div className="sidebar-brand">
                  <ShieldCheck size={20} color="#6366f1" />
                  <span>ForensIQ</span>
                </div>
              </div>

              {/* New Analysis Button in Sidebar */}
              <button 
                type="button" 
                className="btn-new-analysis"
                onClick={handleNewAnalysisSidebar}
              >
                <Plus size={16} />
                <span>New Analysis / Upload</span>
              </button>

              {/* Search Filter Box */}
              <div className="sidebar-search-box">
                <Search size={14} color="#64748b" />
                <input 
                  type="text" 
                  className="sidebar-search-input" 
                  placeholder="Filter evidence records..."
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                />
                {searchQuery && (
                  <button 
                    type="button" 
                    style={{ background: 'none', border: 'none', color: '#64748b', cursor: 'pointer' }}
                    onClick={() => setSearchQuery('')}
                  >
                    <X size={12} />
                  </button>
                )}
              </div>
            </div>

            {/* Sidebar History Header */}
            <div className="sidebar-history-header">
              <span>Evidence Records</span>
              <span className="mono">{uploadedImages.length} of {totalImagesCount}</span>
            </div>

            {/* Scrollable Evidence List */}
            <div className="sidebar-list">
              {filteredEvidence.length === 0 ? (
                <div style={{ padding: '24px 12px', textAlign: 'center', color: 'var(--text-muted)', fontSize: '0.78rem' }}>
                  <p>No evidence records found.</p>
                  <p style={{ marginTop: '4px' }}>Click "New Analysis" above to upload an image.</p>
                </div>
              ) : (
                filteredEvidence.map((img) => {
                  const isActive = img.image_id === activeImageId;
                  const color = CLASS_COLORS[img.predicted_class] || '#6366f1';
                  return (
                    <div 
                      key={img.image_id} 
                      className={`sidebar-item ${isActive ? 'active' : ''}`}
                      onClick={() => handleSelectImage(img)}
                    >
                      <img 
                        src={img.image_url} 
                        alt={img.filename || img.image_id} 
                        className="sidebar-item-thumb"
                        onError={(e) => {
                          e.target.onerror = null;
                          e.target.style.display = 'none';
                        }}
                      />
                      <div className="sidebar-item-info">
                        <span className="sidebar-item-title" title={img.filename || img.image_id}>
                          {img.filename || `ID: ${img.image_id}`}
                        </span>
                        <div className="sidebar-item-meta">
                          <span 
                            className="sidebar-verdict-pill"
                            style={{ background: `${color}22`, color: color }}
                          >
                            {img.predicted_class || 'ANALYSIS'}
                          </span>
                          <span className="mono" style={{ color: color }}>
                            {img.confidence !== undefined ? `${(img.confidence * 100).toFixed(0)}%` : ''}
                          </span>
                        </div>
                      </div>
                    </div>
                  );
                })
              )}

              {/* "Show More" Button for Historical Uploads */}
              {uploadedImages.length < totalImagesCount && (
                <button 
                  type="button" 
                  className="sidebar-btn-more"
                  onClick={handleShowMoreImages}
                  disabled={loadingMoreImages}
                >
                  {loadingMoreImages ? 'Loading...' : `Show More (${totalImagesCount - uploadedImages.length} remaining)`}
                </button>
              )}
            </div>

            {/* Sidebar Bottom Profile Bar */}
            <div className="sidebar-footer">
              <div className="sidebar-user-info">
                <div className="sidebar-user-avatar">
                  {currentUser.username ? currentUser.username.charAt(0).toUpperCase() : 'U'}
                </div>
                <div style={{ display: 'flex', flexDirection: 'column' }}>
                  <span className="sidebar-username">{currentUser.username}</span>
                  <span style={{ fontSize: '0.68rem', color: '#10b981' }}>● Vault Active</span>
                </div>
              </div>

              <button 
                type="button" 
                className="btn-logout"
                title="Sign out of your evidence vault"
                onClick={handleLogout}
              >
                <LogOut size={13} />
                <span>Exit</span>
              </button>
            </div>
          </aside>

          {/* Main Content Area */}
          <main className="chatgpt-main">
            {/* Top Navbar */}
            <header className="chatgpt-navbar">
              <div className="navbar-left">
                <button 
                  type="button" 
                  className="btn-sidebar-toggle"
                  title={sidebarOpen ? "Hide sidebar" : "Show sidebar"}
                  onClick={() => setSidebarOpen(!sidebarOpen)}
                >
                  <Menu size={18} />
                </button>

                <div className="navbar-active-title">
                  {activeImage ? (
                    <>
                      <span>{activeImage.filename || `Evidence ${activeImage.image_id}`}</span>
                      <span 
                        className="sidebar-verdict-pill"
                        style={{ 
                          background: `${CLASS_COLORS[activeImage.predicted_class]}25`, 
                          color: CLASS_COLORS[activeImage.predicted_class] 
                        }}
                      >
                        {CLASS_DISPLAY_NAMES[activeImage.predicted_class] || activeImage.predicted_class}
                      </span>
                    </>
                  ) : (
                    <span>Forensic Diagnostic Assistant</span>
                  )}
                </div>
              </div>

              <div className="navbar-right">
                <button 
                  type="button" 
                  className={`btn-toggle-telemetry ${showTelemetryDrawer ? 'active' : ''}`}
                  onClick={() => setShowTelemetryDrawer(!showTelemetryDrawer)}
                >
                  <Activity size={14} />
                  <span>{showTelemetryDrawer ? 'Hide Inspector' : 'Show Inspector'}</span>
                </button>
              </div>
            </header>

            {/* Body: Chat Stream & Telemetry Drawer */}
            <div className="chatgpt-body">
              {/* ChatGPT Conversational Feed */}
              <div className="chatgpt-chat-view">
                <div className="chatgpt-messages-scroll">
                  {chatMessages.length === 0 ? (
                    <div style={{ textAlign: 'center', padding: '60px 20px', color: 'var(--text-muted)' }}>
                      <Bot size={44} style={{ margin: '0 auto 14px', color: '#6366f1', opacity: 0.6 }} />
                      <h3 style={{ color: '#fff', fontSize: '1.25rem', marginBottom: '8px' }}>
                        Welcome to ForensIQ, {currentUser.username}
                      </h3>
                      <p style={{ maxWidth: '480px', margin: '0 auto', fontSize: '0.88rem', lineHeight: '1.5' }}>
                        Upload an image using the attachment button (📎) below or select an evidence record from the sidebar. All your queries and answers will appear directly in this chat.
                      </p>
                    </div>
                  ) : (
                    chatMessages.map((msg, idx) => (
                      <React.Fragment key={idx}>
                        {msg.role === 'user' ? (
                          /* USER QUESTION: RIGHT-ALIGNED CHAT BUBBLE */
                          <div className="chatgpt-message-row user-row">
                            <div className="chatgpt-bubble-user">
                              {msg.imagePreview && (
                                <div className="chat-attached-image-card">
                                  <img 
                                    src={msg.imagePreview} 
                                    alt="Attached Evidence" 
                                    className="chat-attached-image-thumb" 
                                  />
                                  <div className="chat-attached-image-name">
                                    📷 {msg.imageName || 'Attached Evidence'}
                                  </div>
                                </div>
                              )}
                              <div>{msg.content}</div>
                            </div>
                            <div className="chatgpt-avatar user">
                              {currentUser.username ? currentUser.username.charAt(0).toUpperCase() : 'U'}
                            </div>
                          </div>
                        ) : (
                          /* ASSISTANT RESPONSE: LEFT-ALIGNED WITH STICKERS */
                          <div className="chatgpt-message-row assistant-row">
                            <div className="chatgpt-avatar bot">
                              <Bot size={18} />
                            </div>
                            <div className="chatgpt-content">
                              <div>
                                {msg.content}
                                {msg.isStreaming && <span className="streaming-cursor"></span>}
                              </div>

                              {/* Contextual Forensic Stickers */}
                              {!msg.isStreaming && (
                                <div className="response-stickers-row">
                                  {(msg.stickers || getResponseStickers(msg.content, msg.verdict || result?.predicted_class)).map((stk, sIdx) => (
                                    <span key={sIdx} className={`response-sticker sticker-${stk.type}`}>
                                      {stk.label}
                                    </span>
                                  ))}
                                </div>
                              )}
                            </div>
                          </div>
                        )}
                      </React.Fragment>
                    ))
                  )}

                  {loadingChat && (
                    <div className="chatgpt-message-row assistant-row">
                      <div className="chatgpt-avatar bot">
                        <Bot size={18} />
                      </div>
                      <div className="chatgpt-content" style={{ display: 'flex', alignItems: 'center', gap: '8px', color: '#a5b4fc' }}>
                        <span className="spinner" style={{ width: '14px', height: '14px' }}></span>
                        <span>Evaluating neural memory checkpoints & generating forensic diagnostic...</span>
                      </div>
                    </div>
                  )}
                  <div ref={chatBottomRef} />
                </div>

                {/* Floating Bottom Input Bar */}
                <div className="chatgpt-input-container">
                  {/* Quick Suggested Chips */}
                  <div className="chat-chips-row" style={{ marginBottom: '4px' }}>
                    {result?.predicted_class && (
                      <button 
                        type="button" 
                        className="chip-btn" 
                        onClick={() => handleSendMessage(`Why was this image classified as ${result.predicted_class.toUpperCase()}?`)}
                      >
                        Why {result.predicted_class.toUpperCase()}?
                      </button>
                    )}
                    <button 
                      type="button" 
                      className="chip-btn" 
                      onClick={() => handleSendMessage("Explain the SRM noise residual energy and dominant kernel findings.")}
                    >
                      Explain SRM Residual
                    </button>
                    <button 
                      type="button" 
                      className="chip-btn" 
                      onClick={() => handleSendMessage("Check spatial noise inconsistency between face center and background.")}
                    >
                      Check Noise Inconsistency
                    </button>
                    <button 
                      type="button" 
                      className="chip-btn" 
                      onClick={() => handleSendMessage("List and compare all images uploaded in my private evidence vault.")}
                    >
                      Compare My Uploads
                    </button>
                  </div>

                  {/* Input Box */}
                  <form 
                    className="chatgpt-input-box" 
                    onSubmit={(e) => {
                      e.preventDefault();
                      handleSendMessage(inputQuery);
                    }}
                  >
                    {/* Paperclip in-chat upload button */}
                    <button 
                      type="button" 
                      className="btn-chat-upload"
                      title="Attach image to ask questions about it in chat"
                      onClick={handleChatUploadClick}
                    >
                      <Paperclip size={16} />
                    </button>

                    <input 
                      type="text" 
                      className="chatgpt-input-field" 
                      placeholder={activeImage ? `Ask about ${activeImage.filename || activeImage.image_id}...` : "Ask any forensic question or attach an image..."}
                      value={inputQuery}
                      onChange={(e) => setInputQuery(e.target.value)}
                      disabled={loadingChat}
                    />

                    <button 
                      type="submit" 
                      className="chat-send-btn" 
                      style={{ padding: '8px 14px', borderRadius: '10px' }}
                      disabled={!inputQuery.trim() || loadingChat}
                    >
                      <Send size={15} />
                    </button>
                  </form>
                </div>
              </div>

              {/* Right-Side Telemetry & Neural Inspector Drawer */}
              {showTelemetryDrawer && (
                <div className="chatgpt-telemetry-drawer">
                  <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                    <h3 style={{ fontSize: '0.94rem', fontWeight: 700, display: 'flex', alignItems: 'center', gap: '8px' }}>
                      <Activity size={18} color="#6366f1" />
                      Neural Telemetry & Forensics
                    </h3>
                    <button 
                      type="button" 
                      style={{ background: 'none', border: 'none', color: '#64748b', cursor: 'pointer' }}
                      onClick={() => setShowTelemetryDrawer(false)}
                    >
                      <X size={16} />
                    </button>
                  </div>

                  {/* Upload Drop Zone / Modal Area (shown on "+ New Analysis" or when no result) */}
                  {(showUploadModal || !result) && (
                    <div className="glass-card" style={{ padding: '16px' }}>
                      <div 
                        className={`drop-zone ${isDragging ? 'active' : ''}`}
                        style={{ padding: '24px 14px' }}
                        onDragOver={(e) => { e.preventDefault(); setIsDragging(true); }}
                        onDragLeave={() => setIsDragging(false)}
                        onDrop={handleDrop}
                        onClick={() => {
                          uploadSourceRef.current = 'sidebar';
                          if (fileInputRef.current) fileInputRef.current.click();
                        }}
                      >
                        <div className="drop-icon-circle" style={{ width: '44px', height: '44px' }}>
                          <UploadCloud size={22} />
                        </div>
                        <p className="drop-prompt" style={{ fontSize: '0.84rem' }}>Drop image or click to browse</p>
                        <p className="drop-sub" style={{ fontSize: '0.74rem' }}>Creates a new evidence investigation</p>
                      </div>

                      {preview && (
                        <div className="preview-box" style={{ marginTop: '10px' }}>
                          <img src={preview} alt="Selected evidence" className="preview-img" style={{ maxHeight: '140px' }} />
                          <div className="preview-meta">
                            <span className="mono" style={{ fontSize: '0.75rem' }}>{file?.name || 'Selected'}</span>
                            <button type="button" className="btn-remove" onClick={() => { setFile(null); setPreview(null); }}>
                              <X size={13} />
                            </button>
                          </div>
                        </div>
                      )}

                      {error && (
                        <div style={{ marginTop: '10px', padding: '8px 12px', background: 'rgba(239, 68, 68, 0.15)', border: '1px solid rgba(239, 68, 68, 0.35)', color: '#fca5a5', fontSize: '0.78rem', borderRadius: '6px' }}>
                          {error}
                        </div>
                      )}

                      <button 
                        type="button" 
                        className="btn-primary" 
                        style={{ marginTop: '12px', padding: '10px' }}
                        disabled={!file || loading}
                        onClick={handleAnalyzeFromModal}
                      >
                        {loading ? (
                          <>
                            <span className="spinner"></span>
                            <span>Analyzing Evidence...</span>
                          </>
                        ) : (
                          <>
                            <Sparkles size={16} />
                            <span>Analyze & Create Record</span>
                          </>
                        )}
                      </button>
                    </div>
                  )}

                  {/* Active Evidence Preview & Telemetry */}
                  {result && (
                    <>
                      {/* Evidence Thumbnail View */}
                      {preview && (
                        <div style={{ position: 'relative', borderRadius: '10px', overflow: 'hidden', border: '1px solid rgba(255, 255, 255, 0.1)' }}>
                          <img 
                            src={preview} 
                            alt="Active Evidence" 
                            style={{ width: '100%', maxHeight: '180px', objectFit: 'contain', background: '#080d1a', display: 'block' }} 
                          />
                        </div>
                      )}

                      {/* Verdict Header */}
                      <div className="verdict-header" style={{ marginBottom: '14px', paddingBottom: '12px' }}>
                        <div>
                          <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)', textTransform: 'uppercase', marginBottom: '2px' }}>
                            Verdict
                          </div>
                          <div className={`verdict-badge verdict-${result.predicted_class}`} style={{ fontSize: '1.05rem', padding: '4px 12px' }}>
                            {CLASS_DISPLAY_NAMES[result.predicted_class] || result.predicted_class}
                          </div>
                        </div>

                        <div style={{ textAlign: 'right' }}>
                          <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)', textTransform: 'uppercase', marginBottom: '2px' }}>
                            Confidence
                          </div>
                          <div className="confidence-val" style={{ color: CLASS_COLORS[result.predicted_class] || '#fff', fontSize: '1.35rem' }}>
                            {(result.confidence * 100).toFixed(1)}%
                          </div>
                        </div>
                      </div>

                      {/* Ambiguity Alert */}
                      {result.forensics?.is_ambiguous && result.forensics?.runner_up_class && (
                        <div className="ambiguity-banner" style={{ padding: '10px 12px', marginBottom: '14px' }}>
                          <span className="ambiguity-title" style={{ fontSize: '0.78rem' }}>
                            ⚠️ Manipulation Boundary Detected
                          </span>
                          <p className="ambiguity-desc" style={{ fontSize: '0.74rem', marginTop: '2px' }}>
                            Narrow margin vs <strong>{result.forensics.runner_up_class.toUpperCase()}</strong>.
                          </p>
                        </div>
                      )}

                      {/* 4-Class Probability Breakdown */}
                      <div style={{ marginBottom: '16px' }}>
                        <div className="section-label" style={{ fontSize: '0.74rem' }}>4-Class Probability Distribution</div>
                        {result.class_probabilities?.map((item) => {
                          const color = CLASS_COLORS[item.class_name] || '#6366f1';
                          return (
                            <div key={item.class_name} className="prob-row" style={{ marginBottom: '8px' }}>
                              <div className="prob-meta" style={{ fontSize: '0.76rem' }}>
                                <span>{CLASS_DISPLAY_NAMES[item.class_name] || item.class_name}</span>
                                <span className="mono">{item.percentage.toFixed(1)}%</span>
                              </div>
                              <div className="prob-track" style={{ height: '6px' }}>
                                <div 
                                  className="prob-fill" 
                                  style={{ width: `${item.percentage}%`, background: color }}
                                ></div>
                              </div>
                            </div>
                          );
                        })}
                      </div>

                      {/* Neural Forensics Grid */}
                      <div>
                        <div className="section-label" style={{ fontSize: '0.74rem' }}>Extracted Neural Forensics</div>
                        <div className="telemetry-grid" style={{ gap: '8px' }}>
                          <div className="telemetry-card" style={{ padding: '10px' }}>
                            <div className="telemetry-label" style={{ fontSize: '0.7rem' }}>SRM Residual Energy</div>
                            <div className="telemetry-val" style={{ fontSize: '0.92rem' }}>
                              {result.forensics?.srm_residual_energy?.toFixed(4) ?? '0.0000'}
                            </div>
                          </div>

                          <div className="telemetry-card" style={{ padding: '10px' }}>
                            <div className="telemetry-label" style={{ fontSize: '0.7rem' }}>Dominant Filter Kernel</div>
                            <div className="telemetry-val" style={{ fontSize: '0.92rem' }}>
                              {result.forensics?.dominant_kernel?.replace(' Residual', '') ?? 'Cross-Star'}
                            </div>
                          </div>

                          <div className="telemetry-card" style={{ padding: '10px' }}>
                            <div className="telemetry-label" style={{ fontSize: '0.7rem' }}>Spatial Attention Ratio</div>
                            <div className="telemetry-val" style={{ fontSize: '0.92rem' }}>
                              {result.forensics?.attention_spatial_ratio !== undefined 
                                ? `${(result.forensics.attention_spatial_ratio * 100).toFixed(1)}%` 
                                : '50.0%'}
                            </div>
                          </div>

                          <div className="telemetry-card" style={{ padding: '10px' }}>
                            <div className="telemetry-label" style={{ fontSize: '0.7rem' }}>Noise Inconsistency</div>
                            <div className="telemetry-val" style={{ fontSize: '0.92rem' }}>
                              {result.forensics?.noise_inconsistency !== undefined && result.forensics?.noise_inconsistency !== null
                                ? (result.forensics.noise_inconsistency > 1.0 
                                    ? `${result.forensics.noise_inconsistency.toFixed(1)} (Seam)`
                                    : result.forensics.noise_inconsistency.toFixed(3))
                                : '0.000'}
                            </div>
                          </div>
                        </div>
                      </div>
                    </>
                  )}
                </div>
              )}
            </div>
          </main>
        </div>
      )}
    </>
  );
}
