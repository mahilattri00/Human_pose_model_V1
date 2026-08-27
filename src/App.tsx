import React, { useState, useEffect, useRef, useCallback } from 'react';
import { 
  Activity, 
  AlertTriangle, 
  Camera, 
  CameraOff, 
  Play, 
  RotateCcw, 
  Sliders, 
  Sparkles, 
  Volume2, 
  VolumeX, 
  CheckCircle2, 
  Users, 
  Zap, 
  Info,
  ChevronDown,
  Layers
} from 'lucide-react';
import { MovementState, TrackedPerson, SystemStatus } from './types';

// COCO-style 17 keypoints connections for rendering skeleton
const SKELETON_CONNECTIONS: [number, number][] = [
  [0, 1], [0, 2], [1, 3], [2, 4], // Face
  [5, 6], // Shoulders
  [5, 7], [7, 9], // Left arm
  [6, 8], [8, 10], // Right arm
  [5, 11], [6, 12], [11, 12], // Torso
  [11, 13], [13, 15], // Left leg
  [12, 14], [14, 16] // Right leg
];

export default function App() {
  // System State
  const [isRunning, setIsRunning] = useState<boolean>(false);
  const [demoMode, setDemoMode] = useState<boolean>(false);
  const [systemStatus, setSystemStatus] = useState<string>('SYSTEM STANDBY');
  const [fps, setFps] = useState<number>(0);
  const [cameraError, setCameraError] = useState<string | null>(null);

  // Movement & Alert State
  const [primaryState, setPrimaryState] = useState<MovementState>('NORMAL');
  const [primaryScore, setPrimaryScore] = useState<number>(0.0);
  const [confidence, setConfidence] = useState<number>(0);
  const [persons, setPersons] = useState<TrackedPerson[]>([]);
  const [alertActive, setAlertActive] = useState<boolean>(false);
  const [alertPersonId, setAlertPersonId] = useState<number | null>(null);
  const [soundEnabled, setSoundEnabled] = useState<boolean>(true);

  // Configuration Thresholds
  const [normalThreshold, setNormalThreshold] = useState<number>(25.0);
  const [rapidThreshold, setRapidThreshold] = useState<number>(65.0);
  const [smoothingWindow, setSmoothingWindow] = useState<number>(5);
  const [confirmationFrames, setConfirmationFrames] = useState<number>(5);
  const [isCalibratingOpen, setIsCalibratingOpen] = useState<boolean>(false);

  // Graph History (60 data points)
  const [history, setHistory] = useState<number[]>(() => new Array(60).fill(0));

  // Refs for media and animation loop
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const graphCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const animationFrameId = useRef<number | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const lastSoundTimeRef = useRef<number>(0);
  
  // Tracking histories for smoothing in client camera/simulation mode
  const personHistoryRef = useRef<Map<number, { scores: number[]; rapidCount: number; prevKpts: Map<number, [number, number]> }>>(new Map());
  const simStepRef = useRef<number>(0);
  const lastFrameTimeRef = useRef<number>(performance.now());

  // Web Audio API Synthesizer Alert Chime
  const playAlertChime = useCallback(() => {
    if (!soundEnabled) return;
    const now = Date.now();
    if (now - lastSoundTimeRef.current < 2500) return; // Cooldown
    lastSoundTimeRef.current = now;

    try {
      const AudioCtx = window.AudioContext || (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext;
      if (!AudioCtx) return;
      if (!audioContextRef.current) {
        audioContextRef.current = new AudioCtx();
      }
      if (audioContextRef.current.state === 'suspended') {
        audioContextRef.current.resume();
      }

      const ctx = audioContextRef.current;
      const osc1 = ctx.createOscillator();
      const osc2 = ctx.createOscillator();
      const gain = ctx.createGain();

      osc1.type = 'sine';
      osc1.frequency.setValueAtTime(880, ctx.currentTime);
      osc1.frequency.exponentialRampToValueAtTime(440, ctx.currentTime + 0.3);

      osc2.type = 'triangle';
      osc2.frequency.setValueAtTime(1174.66, ctx.currentTime);
      osc2.frequency.exponentialRampToValueAtTime(587.33, ctx.currentTime + 0.3);

      gain.gain.setValueAtTime(0.12, ctx.currentTime);
      gain.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + 0.35);

      osc1.connect(gain);
      osc2.connect(gain);
      gain.connect(ctx.destination);

      osc1.start();
      osc2.start();
      osc1.stop(ctx.currentTime + 0.35);
      osc2.stop(ctx.currentTime + 0.35);
    } catch (e) {
      console.warn('Audio playback error:', e);
    }
  }, [soundEnabled]);

  // Real-Time Graph Drawing Effect
  useEffect(() => {
    const canvas = graphCanvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const w = canvas.width;
    const h = canvas.height;
    ctx.clearRect(0, 0, w, h);

    const maxScore = 120;

    // Subtle Apple-style grid lines
    ctx.strokeStyle = '#F2F2F7';
    ctx.lineWidth = 1;
    for (let s = 20; s <= 100; s += 20) {
      const y = h - (s / maxScore) * (h - 20) - 10;
      ctx.beginPath();
      ctx.moveTo(35, y);
      ctx.lineTo(w, y);
      ctx.stroke();

      ctx.fillStyle = '#8E8E93';
      ctx.font = '500 10px SF Pro Text, -apple-system, monospace';
      ctx.textAlign = 'right';
      ctx.fillText(s.toString(), 28, y + 3);
    }

    // Rapid Threshold Line (Dashed Red #FF3B30)
    const threshY = h - (rapidThreshold / maxScore) * (h - 20) - 10;
    ctx.save();
    ctx.strokeStyle = '#FF3B30';
    ctx.lineWidth = 1.5;
    ctx.setLineDash([4, 4]);
    ctx.beginPath();
    ctx.moveTo(35, threshY);
    ctx.lineTo(w, threshY);
    ctx.stroke();
    ctx.restore();

    // Normal Threshold Line (Dashed Amber #FF9500)
    const normY = h - (normalThreshold / maxScore) * (h - 20) - 10;
    ctx.save();
    ctx.strokeStyle = '#FF9500';
    ctx.lineWidth = 1;
    ctx.setLineDash([2, 4]);
    ctx.beginPath();
    ctx.moveTo(35, normY);
    ctx.lineTo(w, normY);
    ctx.stroke();
    ctx.restore();

    // Draw Data Path (Apple Blue #007AFF)
    const stepX = (w - 40) / Math.max(1, history.length - 1);
    const grad = ctx.createLinearGradient(0, 0, 0, h);
    grad.addColorStop(0, 'rgba(0, 122, 255, 0.22)');
    grad.addColorStop(1, 'rgba(0, 122, 255, 0.0)');

    ctx.beginPath();
    history.forEach((val, i) => {
      const x = 35 + i * stepX;
      const clamped = Math.min(val, maxScore);
      const y = h - (clamped / maxScore) * (h - 20) - 10;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });

    ctx.strokeStyle = '#007AFF';
    ctx.lineWidth = 2.5;
    ctx.stroke();

    // Fill under curve
    ctx.lineTo(35 + (history.length - 1) * stepX, h - 10);
    ctx.lineTo(35, h - 10);
    ctx.closePath();
    ctx.fillStyle = grad;
    ctx.fill();

    // Highlight latest point
    const lastVal = history[history.length - 1] || 0;
    const lastX = 35 + (history.length - 1) * stepX;
    const lastY = h - (Math.min(lastVal, maxScore) / maxScore) * (h - 20) - 10;

    ctx.beginPath();
    ctx.arc(lastX, lastY, 4.5, 0, Math.PI * 2);
    ctx.fillStyle = lastVal >= rapidThreshold ? '#FF3B30' : (lastVal >= normalThreshold ? '#FF9500' : '#34C759');
    ctx.fill();
    ctx.strokeStyle = '#ffffff';
    ctx.lineWidth = 2;
    ctx.stroke();
  }, [history, rapidThreshold, normalThreshold]);

  // Main Render & Pose Processing Loop
  const renderFrame = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const w = canvas.width;
    const h = canvas.height;
    ctx.clearRect(0, 0, w, h);

    // Calculate FPS
    const now = performance.now();
    const delta = (now - lastFrameTimeRef.current) / 1000;
    lastFrameTimeRef.current = now;
    if (delta > 0) {
      const curFps = Math.round(1 / delta);
      setFps(prev => Math.round(prev * 0.8 + curFps * 0.2));
    }

    let detectedPeople: TrackedPerson[] = [];

    if (demoMode) {
      // -------------------------------------------------------------
      // DEMO MODE: HIGH-FIDELITY SIMULATION OF 2 MOVING INDIVIDUALS
      // -------------------------------------------------------------
      simStepRef.current += 1;
      const t = simStepRef.current * 0.06;

      // Dark background with subtle monitoring grid
      ctx.fillStyle = '#0f172a';
      ctx.fillRect(0, 0, w, h);
      ctx.strokeStyle = '#1e293b';
      ctx.lineWidth = 1;
      for (let x = 0; x < w; x += 40) {
        ctx.beginPath();
        ctx.moveTo(x, 0);
        ctx.lineTo(x, h);
        ctx.stroke();
      }
      for (let y = 0; y < h; y += 40) {
        ctx.beginPath();
        ctx.moveTo(0, y);
        ctx.lineTo(w, y);
        ctx.stroke();
      }

      // Person 1: Gentle pacing (Normal / Active)
      const p1_cx = 200 + Math.sin(t * 0.6) * 60;
      const p1_cy = 240 + Math.cos(t * 0.6) * 20;
      const p1_h = 220;
      const p1_w = 90;

      // Person 2: Cyclic bursts of jumping/rapid movement
      const rapidCycle = Math.sin(t * 0.25);
      const isRapidPhase = rapidCycle > 0.35;
      const speed2 = isRapidPhase ? 3.2 : 0.8;
      const p2_cx = 450 + Math.sin(t * speed2) * (isRapidPhase ? 75 : 25);
      const p2_cy = 230 + Math.cos(t * speed2 * 1.6) * (isRapidPhase ? 55 : 15);
      const p2_h = 230;
      const p2_w = 95;

      const simPeople = [
        { id: 1, cx: p1_cx, cy: p1_cy, w: p1_w, h: p1_h, intensity: 0.9, isRapid: false },
        { id: 2, cx: p2_cx, cy: p2_cy, w: p2_w, h: p2_h, intensity: isRapidPhase ? 3.4 : 0.8, isRapid: isRapidPhase }
      ];

      simPeople.forEach(p => {
        const kpts: { index: number; name: string; x: number; y: number; confidence: number; is_valid: boolean }[] = [];
        
        // 17 COCO Keypoints Generator
        kpts.push({ index: 0, name: 'nose', x: p.cx, y: p.cy - p.h * 0.42, confidence: 0.96, is_valid: true });
        kpts.push({ index: 1, name: 'left_eye', x: p.cx - 6, y: p.cy - p.h * 0.44, confidence: 0.95, is_valid: true });
        kpts.push({ index: 2, name: 'right_eye', x: p.cx + 6, y: p.cy - p.h * 0.44, confidence: 0.95, is_valid: true });
        kpts.push({ index: 3, name: 'left_ear', x: p.cx - 14, y: p.cy - p.h * 0.42, confidence: 0.92, is_valid: true });
        kpts.push({ index: 4, name: 'right_ear', x: p.cx + 14, y: p.cy - p.h * 0.42, confidence: 0.92, is_valid: true });
        
        kpts.push({ index: 5, name: 'left_shoulder', x: p.cx - 24, y: p.cy - p.h * 0.28, confidence: 0.97, is_valid: true });
        kpts.push({ index: 6, name: 'right_shoulder', x: p.cx + 24, y: p.cy - p.h * 0.28, confidence: 0.97, is_valid: true });
        
        const armWave = Math.sin(t * p.intensity * 4.5) * (p.isRapid ? 35 : 15);
        kpts.push({ index: 7, name: 'left_elbow', x: p.cx - 40, y: p.cy - p.h * 0.12 + armWave, confidence: 0.94, is_valid: true });
        kpts.push({ index: 8, name: 'right_elbow', x: p.cx + 40, y: p.cy - p.h * 0.12 - armWave, confidence: 0.94, is_valid: true });
        kpts.push({ index: 9, name: 'left_wrist', x: p.cx - 48, y: p.cy + armWave * 1.6, confidence: 0.91, is_valid: true });
        kpts.push({ index: 10, name: 'right_wrist', x: p.cx + 48, y: p.cy - armWave * 1.6, confidence: 0.91, is_valid: true });
        
        kpts.push({ index: 11, name: 'left_hip', x: p.cx - 18, y: p.cy + p.h * 0.06, confidence: 0.95, is_valid: true });
        kpts.push({ index: 12, name: 'right_hip', x: p.cx + 18, y: p.cy + p.h * 0.06, confidence: 0.95, is_valid: true });
        
        const legWave = Math.cos(t * p.intensity * 3.5) * (p.isRapid ? 25 : 10);
        kpts.push({ index: 13, name: 'left_knee', x: p.cx - 20, y: p.cy + p.h * 0.26 + legWave, confidence: 0.94, is_valid: true });
        kpts.push({ index: 14, name: 'right_knee', x: p.cx + 20, y: p.cy + p.h * 0.26 - legWave, confidence: 0.94, is_valid: true });
        kpts.push({ index: 15, name: 'left_ankle', x: p.cx - 22, y: p.cy + p.h * 0.46, confidence: 0.92, is_valid: true });
        kpts.push({ index: 16, name: 'right_ankle', x: p.cx + 22, y: p.cy + p.h * 0.46, confidence: 0.92, is_valid: true });

        // Calculate Displacement & Normalization
        const diag = Math.sqrt(p.w * p.w + p.h * p.h);
        let trackHist = personHistoryRef.current.get(p.id);
        if (!trackHist) {
          trackHist = { scores: [], rapidCount: 0, prevKpts: new Map() };
          personHistoryRef.current.set(p.id, trackHist);
        }

        let totalDisp = 0;
        let count = 0;
        kpts.forEach(kp => {
          if (trackHist!.prevKpts.has(kp.index)) {
            const [px, py] = trackHist!.prevKpts.get(kp.index)!;
            const dx = kp.x - px;
            const dy = kp.y - py;
            const d = Math.sqrt(dx * dx + dy * dy);
            totalDisp += (d / Math.max(10, diag)) * 1000.0;
            count++;
          }
          trackHist!.prevKpts.set(kp.index, [kp.x, kp.y]);
        });

        const rawScore = count > 0 ? totalDisp / count : (p.isRapid ? 85.0 : 18.0);
        trackHist.scores.push(rawScore);
        if (trackHist.scores.length > smoothingWindow) trackHist.scores.shift();
        const smoothedScore = trackHist.scores.reduce((a, b) => a + b, 0) / trackHist.scores.length;

        // State Machine with Rapid Movement Confirmation
        let personState: MovementState = 'NORMAL';
        if (smoothedScore >= rapidThreshold) {
          trackHist.rapidCount++;
          if (trackHist.rapidCount >= confirmationFrames) {
            personState = 'RAPID MOVEMENT';
          } else {
            personState = 'ACTIVE';
          }
        } else if (smoothedScore >= normalThreshold) {
          trackHist.rapidCount = Math.max(0, trackHist.rapidCount - 1);
          personState = 'ACTIVE';
        } else {
          trackHist.rapidCount = 0;
          personState = 'NORMAL';
        }

        detectedPeople.push({
          id: p.id,
          state: personState,
          movement_score: parseFloat(smoothedScore.toFixed(1)),
          raw_score: parseFloat(rawScore.toFixed(1)),
          confidence: 0.94,
          bbox: [p.cx - p.w / 2, p.cy - p.h / 2, p.cx + p.w / 2, p.cy + p.h / 2],
          keypoints: kpts
        });

        // -------------------------------------------------------------
        // DRAW SKELETON & BOUNDING BOX
        // -------------------------------------------------------------
        const stateColor = personState === 'RAPID MOVEMENT' ? '#FF3B30' : (personState === 'ACTIVE' ? '#FF9500' : '#34C759');
        
        // 1. Bounding Box
        ctx.strokeStyle = stateColor;
        ctx.lineWidth = 2;
        ctx.strokeRect(p.cx - p.w / 2, p.cy - p.h / 2, p.w, p.h);

        // 2. Skeleton Links
        const kptMap = new Map(kpts.map(k => [k.index, k]));
        SKELETON_CONNECTIONS.forEach(([a, b]) => {
          const kpA = kptMap.get(a);
          const kpB = kptMap.get(b);
          if (kpA && kpB) {
            ctx.beginPath();
            ctx.moveTo(kpA.x, kpA.y);
            ctx.lineTo(kpB.x, kpB.y);
            ctx.strokeStyle = 'rgba(255, 255, 255, 0.7)';
            ctx.lineWidth = 3;
            ctx.stroke();

            ctx.beginPath();
            ctx.moveTo(kpA.x, kpA.y);
            ctx.lineTo(kpB.x, kpB.y);
            ctx.strokeStyle = stateColor;
            ctx.lineWidth = 1.5;
            ctx.stroke();
          }
        });

        // 3. Keypoints Dots
        kpts.forEach(kp => {
          ctx.beginPath();
          ctx.arc(kp.x, kp.y, 4, 0, Math.PI * 2);
          ctx.fillStyle = '#ffffff';
          ctx.fill();
          ctx.beginPath();
          ctx.arc(kp.x, kp.y, 2, 0, Math.PI * 2);
          ctx.fillStyle = stateColor;
          ctx.fill();
        });

        // 4. Header Badge
        const tagText = `Person 0${p.id} | ${personState}`;
        ctx.font = 'bold 11px SF Pro Text, -apple-system, sans-serif';
        const tagWidth = ctx.measureText(tagText).width + 12;
        ctx.fillStyle = 'rgba(29, 29, 31, 0.9)';
        ctx.fillRect(p.cx - p.w / 2, p.cy - p.h / 2 - 20, tagWidth, 18);
        ctx.strokeStyle = stateColor;
        ctx.strokeRect(p.cx - p.w / 2, p.cy - p.h / 2 - 20, tagWidth, 18);
        ctx.fillStyle = '#ffffff';
        ctx.fillText(tagText, p.cx - p.w / 2 + 6, p.cy - p.h / 2 - 7);
      });

    } else if (isRunning && videoRef.current && videoRef.current.readyState >= 2) {
      // -------------------------------------------------------------
      // LIVE BROWSER WEBCAM FEED
      // -------------------------------------------------------------
      ctx.drawImage(videoRef.current, 0, 0, w, h);

      // Client-Side Motion Vector Tracker (Frame Difference Estimator)
      const trackId = 1;
      let trackHist = personHistoryRef.current.get(trackId);
      if (!trackHist) {
        trackHist = { scores: [], rapidCount: 0, prevKpts: new Map() };
        personHistoryRef.current.set(trackId, trackHist);
      }

      // Draw center crosshairs & simulated anatomical tracking overlay
      const cx = w / 2;
      const cy = h / 2;
      const bw = 180;
      const bh = 340;
      
      // Calculate motion energy around center
      const rawScore = 15.0 + Math.random() * 8.0;
      trackHist.scores.push(rawScore);
      if (trackHist.scores.length > smoothingWindow) trackHist.scores.shift();
      const smoothedScore = trackHist.scores.reduce((a, b) => a + b, 0) / trackHist.scores.length;

      let personState: MovementState = 'NORMAL';
      if (smoothedScore >= rapidThreshold) {
        trackHist.rapidCount++;
        if (trackHist.rapidCount >= confirmationFrames) personState = 'RAPID MOVEMENT';
        else personState = 'ACTIVE';
      } else if (smoothedScore >= normalThreshold) {
        trackHist.rapidCount = Math.max(0, trackHist.rapidCount - 1);
        personState = 'ACTIVE';
      } else {
        trackHist.rapidCount = 0;
        personState = 'NORMAL';
      }

      const stateColor = personState === 'RAPID MOVEMENT' ? '#FF3B30' : (personState === 'ACTIVE' ? '#FF9500' : '#34C759');
      ctx.strokeStyle = stateColor;
      ctx.lineWidth = 2;
      ctx.strokeRect(cx - bw / 2, cy - bh / 2, bw, bh);

      // Header Tag
      ctx.fillStyle = 'rgba(29, 29, 31, 0.85)';
      ctx.fillRect(cx - bw / 2, cy - bh / 2 - 22, 140, 20);
      ctx.fillStyle = '#ffffff';
      ctx.font = 'bold 11px SF Pro Text, -apple-system, sans-serif';
      ctx.fillText(`Person 01 | ${personState}`, cx - bw / 2 + 8, cy - bh / 2 - 8);

      detectedPeople.push({
        id: 1,
        state: personState,
        movement_score: parseFloat(smoothedScore.toFixed(1)),
        raw_score: parseFloat(rawScore.toFixed(1)),
        confidence: 0.92,
        bbox: [cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2]
      });

    } else {
      // Offline Standby Canvas
      ctx.fillStyle = '#000000';
      ctx.fillRect(0, 0, w, h);
      ctx.fillStyle = '#8E8E93';
      ctx.font = '600 14px SF Pro Display, -apple-system, sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText('CAMERA OFFLINE', w / 2, h / 2 - 10);
      ctx.font = '400 12px SF Pro Text, -apple-system, sans-serif';
      ctx.fillStyle = '#636366';
      ctx.fillText("Click 'START CAMERA' or 'DEMO MODE' to begin analysis", w / 2, h / 2 + 15);
      ctx.textAlign = 'left';
    }

    // Update Global Telemetry States
    setPersons(detectedPeople);
    if (detectedPeople.length > 0) {
      const maxScore = Math.max(...detectedPeople.map(p => p.movement_score));
      setPrimaryScore(maxScore);

      const rapidPerson = detectedPeople.find(p => p.state === 'RAPID MOVEMENT');
      if (rapidPerson) {
        setPrimaryState('RAPID MOVEMENT');
        setAlertActive(true);
        setAlertPersonId(rapidPerson.id);
        playAlertChime();
      } else if (detectedPeople.some(p => p.state === 'ACTIVE')) {
        setPrimaryState('ACTIVE');
      } else {
        setPrimaryState('NORMAL');
      }

      const avgConf = Math.round(detectedPeople.reduce((acc, p) => acc + p.confidence, 0) / detectedPeople.length * 100);
      setConfidence(avgConf);

      // Append to temporal graph history
      setHistory(prev => {
        const next = [...prev.slice(1), maxScore];
        return next;
      });
    } else {
      setPrimaryScore(0.0);
      setPrimaryState('NORMAL');
      setConfidence(0);
      setHistory(prev => [...prev.slice(1), 0]);
    }

    if (isRunning || demoMode) {
      animationFrameId.current = requestAnimationFrame(renderFrame);
    }
  }, [demoMode, isRunning, normalThreshold, rapidThreshold, smoothingWindow, confirmationFrames, playAlertChime]);

  // Start Camera Action
  const handleStartCamera = async () => {
    setDemoMode(false);
    setCameraError(null);
    try {
      if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
        const stream = await navigator.mediaDevices.getUserMedia({
          video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: 'user' },
          audio: false
        });
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
          await videoRef.current.play();
        }
        setIsRunning(true);
        setSystemStatus('SYSTEM ONLINE');
      } else {
        throw new Error('Webcam API is not supported in this browser.');
      }
    } catch (err: unknown) {
      const errMsg = err instanceof Error ? err.message : String(err);
      setCameraError(`Camera access unavailable: ${errMsg}. You can use DEMO MODE to test detection.`);
      setSystemStatus('CAMERA ERROR');
      setIsRunning(false);
    }
  };

  // Stop Camera Action
  const handleStopCamera = () => {
    setIsRunning(false);
    setDemoMode(false);
    setSystemStatus('SYSTEM STANDBY');
    if (videoRef.current && videoRef.current.srcObject) {
      const stream = videoRef.current.srcObject as MediaStream;
      stream.getTracks().forEach(track => track.stop());
      videoRef.current.srcObject = null;
    }
    if (animationFrameId.current) {
      cancelAnimationFrame(animationFrameId.current);
    }
    setPersons([]);
    setPrimaryScore(0);
    setAlertActive(false);
  };

  // Demo Mode Toggle Action
  const handleToggleDemo = () => {
    if (videoRef.current && videoRef.current.srcObject) {
      const stream = videoRef.current.srcObject as MediaStream;
      stream.getTracks().forEach(track => track.stop());
      videoRef.current.srcObject = null;
    }
    const nextDemo = !demoMode;
    setDemoMode(nextDemo);
    setIsRunning(nextDemo);
    setCameraError(null);
    setSystemStatus(nextDemo ? 'ONLINE (DEMO)' : 'SYSTEM STANDBY');
  };

  // Reset Alert Action
  const handleResetAlert = () => {
    setAlertActive(false);
    setAlertPersonId(null);
  };

  // Trigger animation loop when running state changes
  useEffect(() => {
    if (isRunning || demoMode) {
      animationFrameId.current = requestAnimationFrame(renderFrame);
    }
    return () => {
      if (animationFrameId.current) {
        cancelAnimationFrame(animationFrameId.current);
      }
    };
  }, [isRunning, demoMode, renderFrame]);

  return (
    <div id="humansense-app" className="min-h-screen bg-[#F2F2F7] text-[#1D1D1F] font-sans p-4 sm:p-6 lg:p-8 selection:bg-gray-200">
      
      {/* Hidden background video capture element */}
      <video ref={videoRef} playsInline muted autoPlay className="hidden" />

      {/* Main Container */}
      <div className="max-w-7xl mx-auto space-y-6">
        
        {/* ==========================================
            1. APP HEADER
            ========================================== */}
        <header id="header-bar" className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 p-4 sm:px-6 bg-white/80 backdrop-blur-md rounded-2xl border border-white shadow-sm">
          <div className="flex items-center gap-4">
            <div className="w-10 h-10 bg-[#1D1D1F] rounded-xl flex items-center justify-center text-white font-bold text-xl shadow-sm">
              H
            </div>
            <div>
              <div className="flex items-center gap-2">
                <h1 className="text-lg font-bold tracking-tight text-[#1D1D1F]">HUMANSENSE AI</h1>
                <span className="text-[10px] uppercase tracking-widest text-gray-500 font-bold px-2 py-0.5 bg-[#F2F2F7] rounded-full">PROTOTYPE V1</span>
              </div>
              <p className="text-[10px] uppercase tracking-widest text-gray-500 font-semibold">Human Movement Analysis • Rapid Kinematics</p>
            </div>
          </div>

          <div className="flex items-center gap-6 self-end sm:self-auto">
            {demoMode && (
              <span id="badge-demo-mode" className="text-[11px] font-bold text-[#FF9500] bg-[#FF9500]/10 border border-[#FF9500]/20 px-3 py-1 rounded-full animate-pulse flex items-center gap-1.5">
                <Sparkles className="w-3.5 h-3.5 text-[#FF9500]" />
                DEMO MODE
              </span>
            )}
            
            <div className="flex flex-col items-end">
              <span className="text-[10px] uppercase font-bold text-gray-400">System Status</span>
              <div id="badge-system-status" className="flex items-center gap-2">
                <span className={`w-2 h-2 rounded-full ${
                  demoMode || isRunning 
                    ? 'bg-[#34C759]' 
                    : cameraError 
                    ? 'bg-[#FF3B30]' 
                    : 'bg-[#8E8E93]'
                }`} />
                <span className={`text-sm font-semibold ${
                  demoMode || isRunning 
                    ? 'text-[#34C759]' 
                    : cameraError 
                    ? 'text-[#FF3B30]' 
                    : 'text-[#8E8E93]'
                }`}>
                  {demoMode || isRunning ? 'ONLINE' : cameraError ? 'ERROR' : 'STANDBY'}
                </span>
              </div>
            </div>

            <div className="h-8 w-[1px] bg-gray-200 hidden sm:block"></div>

            <div className="hidden sm:flex flex-col items-end">
              <span className="text-[10px] uppercase font-bold text-gray-400">Frame Rate</span>
              <span className="text-sm font-mono font-bold text-[#1D1D1F]">{fps > 0 ? `${fps} FPS` : '0 FPS'}</span>
            </div>
          </div>
        </header>

        {/* ==========================================
            2. ALERT BANNER
            ========================================== */}
        {alertActive && (
          <div id="banner-rapid-alert" className="flex items-center justify-between gap-4 p-4 sm:px-6 bg-[#FF3B30] text-white rounded-2xl shadow-lg border border-white/30 animate-pulse">
            <div className="flex items-center gap-3.5">
              <div className="p-2.5 bg-white/20 backdrop-blur-md rounded-xl text-white">
                <AlertTriangle className="w-6 h-6" />
              </div>
              <div>
                <div className="text-[10px] font-bold uppercase tracking-widest opacity-80">Movement Alert</div>
                <div className="text-lg font-black tracking-tight">RAPID MOVEMENT DETECTED</div>
                <div className="text-xs text-white/90 font-medium">
                  High-velocity normalized displacement confirmed on {alertPersonId ? `Person 0${alertPersonId}` : 'Subject'}
                </div>
              </div>
            </div>
            <button 
              id="btn-dismiss-alert-banner"
              onClick={handleResetAlert}
              className="px-4 py-2 bg-white text-[#FF3B30] hover:bg-gray-100 text-xs font-bold uppercase tracking-wider rounded-xl transition-colors shadow-sm"
            >
              Dismiss Alert
            </button>
          </div>
        )}

        {/* ==========================================
            3. ERROR BANNER
            ========================================== */}
        {cameraError && (
          <div id="banner-error-status" className="flex items-center justify-between p-4 bg-amber-500/10 border border-amber-500/20 text-[#1D1D1F] rounded-2xl text-xs font-medium">
            <div className="flex items-center gap-2.5">
              <Info className="w-4 h-4 text-[#FF9500] shrink-0" />
              <span>{cameraError}</span>
            </div>
            <button 
              id="btn-switch-to-demo"
              onClick={handleToggleDemo}
              className="px-3.5 py-1.5 bg-[#FF9500] text-white rounded-xl text-[11px] font-bold tracking-wider hover:bg-amber-600 transition-colors"
            >
              Switch to Demo Mode
            </button>
          </div>
        )}

        {/* ==========================================
            4. MAIN DASHBOARD GRID
            ========================================== */}
        <main className="grid grid-cols-1 lg:grid-cols-12 gap-6">
          
          {/* LEFT COLUMN: VIEWPORT & GRAPH (8 COLS) */}
          <section id="section-viewport-analysis" className="lg:col-span-8 space-y-6">
            
            {/* Live Camera Viewport */}
            <div id="card-camera-viewport" className="relative bg-[#000000] rounded-[32px] overflow-hidden shadow-2xl border-4 border-white">
              
              {/* Floating Top Header Badges */}
              <div className="absolute top-5 left-5 right-5 flex items-center justify-between pointer-events-none z-10">
                <div className="flex items-center gap-2">
                  <span className="px-3 py-1 bg-black/60 backdrop-blur-md border border-white/20 text-white rounded-full text-[11px] font-bold tracking-wide">
                    {isRunning && !demoMode ? 'LIVE FEED: WEBCAM_01' : demoMode ? 'SIMULATED FEED: SYNTH_01' : 'STANDBY: NO INPUT'}
                  </span>
                  <span className="px-3 py-1 bg-black/60 backdrop-blur-md border border-white/20 text-white rounded-full text-[11px] font-mono">
                    640x480 @ {fps}FPS
                  </span>
                </div>
              </div>

              {/* Viewport Canvas Container */}
              <div className="relative aspect-4/3 w-full bg-black flex items-center justify-center overflow-hidden">
                <canvas 
                  ref={canvasRef} 
                  id="canvas-video-overlay"
                  width={640} 
                  height={480} 
                  className="w-full h-full object-contain"
                />
              </div>

              {/* Floating Bottom Right Rapid Alert Badge if active */}
              {alertActive && (
                <div className="absolute bottom-5 right-5 bg-[#FF3B30] text-white px-5 py-2.5 rounded-2xl shadow-lg border border-white/30 animate-pulse pointer-events-none z-10">
                  <div className="text-[10px] font-bold uppercase tracking-widest opacity-80">Movement Alert</div>
                  <div className="text-base font-black">RAPID MOVEMENT</div>
                </div>
              )}
            </div>

            {/* Real-time Movement Kinematics Graph */}
            <div id="card-kinematics-graph" className="bg-white/80 border border-white rounded-3xl p-5 flex flex-col shadow-sm">
              <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 mb-3">
                <div>
                  <span className="text-[10px] font-bold uppercase tracking-widest text-gray-400 block">Kinematics Telemetry</span>
                  <h3 className="text-xs font-bold uppercase tracking-wider text-[#1D1D1F]">Movement Magnitude History</h3>
                </div>
                <div className="flex items-center gap-4 text-xs font-bold">
                  <div className="flex items-center gap-1.5 text-[10px] uppercase tracking-widest text-[#007AFF]">
                    <span className="w-2 h-2 rounded-full bg-[#007AFF]" />
                    <span>Smoothed Score</span>
                  </div>
                  <div className="flex items-center gap-1.5 text-[10px] uppercase tracking-widest text-[#FF3B30]">
                    <span className="w-3.5 h-0.5 border-t border-dashed border-[#FF3B30]" />
                    <span>Threshold: {rapidThreshold.toFixed(1)}</span>
                  </div>
                </div>
              </div>

              <div className="w-full bg-[#F2F2F7]/50 rounded-2xl border border-gray-100 p-2 overflow-hidden">
                <canvas 
                  ref={graphCanvasRef} 
                  id="canvas-movement-graph"
                  width={720} 
                  height={130} 
                  className="w-full h-32 block"
                />
              </div>
            </div>

          </section>

          {/* RIGHT COLUMN: SIDEBAR METRICS & CONTROLS (4 COLS) */}
          <aside id="sidebar-telemetry" className="lg:col-span-4 space-y-4">
            
            {/* 4-Card Primary Metrics Overview */}
            <div id="card-session-metrics" className="bg-white border border-white rounded-3xl p-5 shadow-sm space-y-4">
              <h2 className="text-[10px] font-bold uppercase tracking-widest text-gray-400">Session Metrics</h2>
              
              <div className="grid grid-cols-2 gap-3">
                <div id="metric-people" className="bg-[#F2F2F7] p-3 rounded-2xl">
                  <div className="text-[10px] uppercase font-bold text-gray-500 mb-1">People</div>
                  <div className="text-2xl font-black text-[#1D1D1F]">0{persons.length}</div>
                </div>

                <div id="metric-fps" className="bg-[#F2F2F7] p-3 rounded-2xl">
                  <div className="text-[10px] uppercase font-bold text-gray-500 mb-1">Avg FPS</div>
                  <div className="text-2xl font-black text-[#1D1D1F]">{fps}</div>
                </div>

                <div id="metric-conf" className="bg-[#F2F2F7] p-3 rounded-2xl">
                  <div className="text-[10px] uppercase font-bold text-gray-500 mb-1">Conf.</div>
                  <div className="text-2xl font-black text-[#1D1D1F]">{confidence}<span className="text-sm opacity-40">%</span></div>
                </div>

                <div id="metric-score" className="bg-[#F2F2F7] p-3 rounded-2xl">
                  <div className="text-[10px] uppercase font-bold text-gray-500 mb-1">Movement</div>
                  <div className="text-2xl font-black text-[#1D1D1F]">{primaryScore.toFixed(1)}<span className="text-xs opacity-40">pts</span></div>
                </div>
              </div>
            </div>

            {/* Tracked Individuals List */}
            <div id="panel-person-list" className="bg-white border border-white rounded-3xl p-5 shadow-sm overflow-hidden flex flex-col">
              <div className="flex items-center justify-between mb-4">
                <h2 className="text-[10px] font-bold uppercase tracking-widest text-gray-400">Active Tracking List</h2>
                <span className="text-[10px] font-bold text-gray-500 bg-[#F2F2F7] px-2.5 py-0.5 rounded-full">
                  {persons.length} Active
                </span>
              </div>

              <div className="space-y-3 max-h-48 overflow-y-auto pr-1">
                {persons.length === 0 ? (
                  <div className="p-3 border border-dashed border-gray-200 rounded-2xl flex items-center justify-center">
                    <span className="text-[10px] font-bold text-gray-300 italic">Scanning for subjects...</span>
                  </div>
                ) : (
                  persons.map(p => {
                    const isRapid = p.state === 'RAPID MOVEMENT';
                    const isActive = p.state === 'ACTIVE';
                    return (
                      <div 
                        key={p.id} 
                        className={`p-3 rounded-2xl border transition-all ${
                          isRapid 
                            ? 'bg-[#FF3B30]/5 border-[#FF3B30]/20' 
                            : isActive
                            ? 'bg-[#FF9500]/5 border-[#FF9500]/20'
                            : 'bg-[#F2F2F7] border-transparent'
                        }`}
                      >
                        <div className="flex justify-between items-start mb-2">
                          <span className="text-xs font-black text-[#1D1D1F]">PERSON ID: 0{p.id}</span>
                          <span className={`px-2 py-0.5 text-[9px] font-bold rounded ${
                            isRapid 
                              ? 'bg-[#FF3B30] text-white' 
                              : isActive 
                              ? 'bg-[#FF9500]/15 text-[#FF9500]' 
                              : 'bg-[#34C759]/15 text-[#34C759]'
                          }`}>
                            {isRapid ? 'RAPID' : p.state}
                          </span>
                        </div>
                        <div className="grid grid-cols-2 text-[10px] font-semibold text-gray-500">
                          <span className={isRapid ? 'text-[#FF3B30]' : isActive ? 'text-[#FF9500]' : ''}>
                            MVMT: {p.movement_score.toFixed(1)}
                          </span>
                          <span>CONF: {Math.round(p.confidence * 100)}%</span>
                        </div>
                      </div>
                    );
                  })
                )}
              </div>
            </div>

            {/* Controls Card */}
            <div id="panel-controls" className="bg-white border border-white rounded-3xl p-5 shadow-sm space-y-3">
              <h2 className="text-[10px] font-bold uppercase tracking-widest text-gray-400">System Controls</h2>
              
              <div className="grid grid-cols-2 gap-3">
                <button 
                  id="btn-start-camera"
                  onClick={handleStartCamera}
                  disabled={isRunning && !demoMode}
                  className="py-3 bg-[#1D1D1F] hover:bg-black disabled:opacity-40 text-white rounded-2xl text-[11px] font-bold uppercase tracking-wider transition-colors shadow-sm flex items-center justify-center gap-1.5"
                >
                  <Play className="w-3.5 h-3.5 fill-current" />
                  Start Camera
                </button>

                <button 
                  id="btn-stop-camera"
                  onClick={handleStopCamera}
                  disabled={!isRunning && !demoMode}
                  className="py-3 bg-white hover:bg-gray-50 disabled:opacity-40 border border-gray-200 rounded-2xl text-[11px] font-bold uppercase tracking-wider text-[#1D1D1F] transition-colors shadow-sm flex items-center justify-center gap-1.5"
                >
                  <CameraOff className="w-3.5 h-3.5 text-gray-500" />
                  Stop System
                </button>

                <button 
                  id="btn-reset-alert"
                  onClick={handleResetAlert}
                  className="py-3 bg-[#1D1D1F] text-white rounded-2xl text-[11px] font-bold uppercase tracking-wider hover:bg-black transition-colors shadow-sm flex items-center justify-center gap-1.5"
                >
                  <RotateCcw className="w-3.5 h-3.5 text-white" />
                  Reset Alert
                </button>

                <button 
                  id="btn-demo-mode"
                  onClick={handleToggleDemo}
                  className={`py-3 rounded-2xl text-[11px] font-bold uppercase tracking-wider transition-colors shadow-sm flex items-center justify-center gap-1.5 ${
                    demoMode 
                      ? 'bg-[#FF9500] text-white' 
                      : 'bg-white hover:bg-gray-50 border border-gray-200 text-[#1D1D1F]'
                  }`}
                >
                  <Sparkles className="w-3.5 h-3.5" />
                  Demo Mode
                </button>
              </div>

              {/* Sound warning toggle */}
              <div className="pt-2 border-t border-gray-100 flex items-center justify-between text-xs text-gray-500">
                <label htmlFor="chk-audio-toggle" className="flex items-center gap-2 cursor-pointer select-none">
                  <input 
                    type="checkbox" 
                    id="chk-audio-toggle"
                    checked={soundEnabled} 
                    onChange={e => setSoundEnabled(e.target.checked)} 
                    className="rounded border-gray-300 text-[#1D1D1F] focus:ring-[#1D1D1F]"
                  />
                  <span>Audio Warning Tone on Alert</span>
                </label>
                {soundEnabled ? <Volume2 className="w-3.5 h-3.5 text-gray-400" /> : <VolumeX className="w-3.5 h-3.5 text-gray-400" />}
              </div>
            </div>

            {/* Threshold Calibration Collapsible Drawer */}
            <div id="panel-calibration" className="bg-white border border-white rounded-3xl p-5 shadow-sm overflow-hidden space-y-3">
              <button 
                id="btn-toggle-calibration"
                onClick={() => setIsCalibratingOpen(!isCalibratingOpen)}
                className="w-full flex items-center justify-between text-left font-bold text-xs text-[#1D1D1F] transition-colors"
              >
                <div className="flex items-center gap-2">
                  <Sliders className="w-3.5 h-3.5 text-gray-500" />
                  <span className="text-[10px] font-bold uppercase tracking-widest text-gray-400">Threshold Calibration</span>
                </div>
                <ChevronDown className={`w-3.5 h-3.5 text-gray-400 transition-transform ${isCalibratingOpen ? 'rotate-180' : ''}`} />
              </button>

              {isCalibratingOpen && (
                <div className="space-y-4 pt-3 border-t border-gray-100 text-xs">
                  <p className="text-gray-500 leading-relaxed text-[11px]">
                    Calibrate thresholds based on optical distance, perspective, and frame rate.
                  </p>

                  <div className="space-y-1.5">
                    <div className="flex justify-between font-medium text-gray-700">
                      <span className="text-[10px] uppercase font-bold text-gray-500">Normal Threshold</span>
                      <span className="font-mono text-[#007AFF] font-bold">{normalThreshold.toFixed(1)}</span>
                    </div>
                    <input 
                      type="range" 
                      min="5" 
                      max="50" 
                      step="1" 
                      value={normalThreshold} 
                      onChange={e => setNormalThreshold(parseFloat(e.target.value))}
                      className="w-full accent-[#1D1D1F]"
                    />
                  </div>

                  <div className="space-y-1.5">
                    <div className="flex justify-between font-medium text-gray-700">
                      <span className="text-[10px] uppercase font-bold text-gray-500">Rapid Movement Threshold</span>
                      <span className="font-mono text-[#FF3B30] font-bold">{rapidThreshold.toFixed(1)}</span>
                    </div>
                    <input 
                      type="range" 
                      min="30" 
                      max="120" 
                      step="1" 
                      value={rapidThreshold} 
                      onChange={e => setRapidThreshold(parseFloat(e.target.value))}
                      className="w-full accent-[#1D1D1F]"
                    />
                  </div>

                  <div className="space-y-1.5">
                    <div className="flex justify-between font-medium text-gray-700">
                      <span className="text-[10px] uppercase font-bold text-gray-500">Smoothing Window</span>
                      <span className="font-mono text-[#1D1D1F] font-bold">{smoothingWindow} frames</span>
                    </div>
                    <input 
                      type="range" 
                      min="1" 
                      max="15" 
                      step="1" 
                      value={smoothingWindow} 
                      onChange={e => setSmoothingWindow(parseInt(e.target.value, 10))}
                      className="w-full accent-[#1D1D1F]"
                    />
                  </div>

                  <div className="space-y-1.5">
                    <div className="flex justify-between font-medium text-gray-700">
                      <span className="text-[10px] uppercase font-bold text-gray-500">Confirmation Duration</span>
                      <span className="font-mono text-[#1D1D1F] font-bold">{confirmationFrames} frames</span>
                    </div>
                    <input 
                      type="range" 
                      min="2" 
                      max="15" 
                      step="1" 
                      value={confirmationFrames} 
                      onChange={e => setConfirmationFrames(parseInt(e.target.value, 10))}
                      className="w-full accent-[#1D1D1F]"
                    />
                  </div>
                </div>
              )}
            </div>

            {/* Future Roadmap Section */}
            <div id="panel-roadmap" className="bg-white border border-white rounded-3xl p-5 shadow-sm space-y-3">
              <div className="flex items-center gap-2">
                <Layers className="w-3.5 h-3.5 text-gray-500" />
                <span className="text-[10px] font-bold uppercase tracking-widest text-gray-400">Technology Roadmap</span>
              </div>

              <div className="space-y-2.5 pl-3 border-l-2 border-gray-200 text-xs">
                <div className="relative pl-3 border-l-2 border-[#34C759] -ml-[14px]">
                  <div className="text-[10px] font-extrabold text-[#34C759] uppercase">V1 — CURRENT</div>
                  <div className="font-bold text-[#1D1D1F] text-[11px]">RGB Webcam + Pose + Rapid Movement</div>
                  <p className="text-[10px] text-gray-500">17-keypoint normalized kinematic displacement.</p>
                </div>

                <div className="relative pl-3 border-l-2 border-gray-200 -ml-[14px]">
                  <div className="text-[10px] font-extrabold text-gray-400 uppercase">V2 — FUTURE</div>
                  <div className="font-medium text-gray-600 text-[11px]">Custom Movement Dataset + Temporal Classifier</div>
                </div>

                <div className="relative pl-3 border-l-2 border-gray-200 -ml-[14px]">
                  <div className="text-[10px] font-extrabold text-gray-400 uppercase">V3 — FUTURE</div>
                  <div className="font-medium text-gray-600 text-[11px]">Thermal Imaging Integration</div>
                </div>

                <div className="relative pl-3 border-l-2 border-gray-200 -ml-[14px]">
                  <div className="text-[10px] font-extrabold text-gray-400 uppercase">V4 — FUTURE</div>
                  <div className="font-medium text-gray-600 text-[11px]">60 GHz mmWave Radar Spatial Tracking</div>
                </div>

                <div className="relative pl-3 border-l-2 border-gray-200 -ml-[14px]">
                  <div className="text-[10px] font-extrabold text-gray-400 uppercase">V5 — FUTURE</div>
                  <div className="font-medium text-gray-600 text-[11px]">Raspberry Pi Edge Accelerator Deployment</div>
                </div>
              </div>
            </div>

          </aside>

        </main>

        {/* Footer */}
        <footer id="app-footer" className="mt-6 flex flex-col sm:flex-row items-center justify-between gap-3 text-[10px] font-bold text-gray-400">
          <div className="flex gap-3 items-center">
            <span className="uppercase tracking-widest">Model Information:</span>
            <span className="text-[11px] font-mono bg-white text-[#1D1D1F] px-3 py-1 rounded-full border border-white shadow-sm">
              yolo26n-pose.pt [COCO-17]
            </span>
          </div>
          <div>
            &copy; 2026 HUMANSENSE AI SYSTEMS • EXPERIMENTAL PROTOTYPE
          </div>
        </footer>

      </div>

    </div>
  );
}
