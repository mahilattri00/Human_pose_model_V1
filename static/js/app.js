/**
 * HUMANSENSE AI — Client Dashboard & Kinematics Controller
 * Connects frontend dashboard, handles camera streams, runs real-time graph,
 * manages alert audio synthesizer, and synchronizes status with the Flask backend.
 */

// Global State
const state = {
  isRunning: false,
  demoMode: false,
  soundEnabled: true,
  audioCtx: null,
  alertActive: false,
  alertPersonId: null,
  rapidThreshold: 65.0,
  normalThreshold: 25.0,
  smoothingWindow: 5,
  confirmationFrames: 5,
  historyData: [],
  maxHistoryLength: 60,
  lastAlertSoundTime: 0
};

// DOM Element References
const elements = {
  statusPill: document.getElementById('system-status-pill'),
  statusText: document.getElementById('system-status-text'),
  demoBadge: document.getElementById('demo-badge'),
  alertBanner: document.getElementById('alert-banner'),
  alertDescText: document.getElementById('alert-desc-text'),
  btnBannerReset: document.getElementById('btn-banner-reset'),
  errorBanner: document.getElementById('error-banner'),
  errorText: document.getElementById('error-text'),
  btnErrorDemo: document.getElementById('btn-error-demo'),
  
  streamImg: document.getElementById('stream-img'),
  clientVideo: document.getElementById('client-video'),
  clientCanvas: document.getElementById('client-canvas'),
  
  valFps: document.getElementById('val-fps'),
  valPeopleCount: document.getElementById('val-people-count'),
  valPrimaryState: document.getElementById('val-primary-state'),
  valPrimaryScore: document.getElementById('val-primary-score'),
  valConfidence: document.getElementById('val-confidence'),
  badgeTrackCount: document.getElementById('badge-track-count'),
  personContainer: document.getElementById('person-container'),
  
  btnStart: document.getElementById('btn-start'),
  btnStop: document.getElementById('btn-stop'),
  btnResetAlert: document.getElementById('btn-reset-alert'),
  btnDemo: document.getElementById('btn-demo'),
  chkSound: document.getElementById('chk-sound'),
  
  sliderNormal: document.getElementById('slider-normal'),
  sliderRapid: document.getElementById('slider-rapid'),
  sliderSmoothing: document.getElementById('slider-smoothing'),
  sliderConfirm: document.getElementById('slider-confirm'),
  lblValNormal: document.getElementById('lbl-val-normal'),
  lblValRapid: document.getElementById('lbl-val-rapid'),
  lblValSmoothing: document.getElementById('lbl-val-smoothing'),
  lblValConfirm: document.getElementById('lbl-val-confirm'),
  lblRapidThresh: document.getElementById('lbl-rapid-thresh'),
  btnSaveCal: document.getElementById('btn-save-cal'),
  
  chartCanvas: document.getElementById('movement-chart')
};

// Initialize Graph History
for (let i = 0; i < state.maxHistoryLength; i++) {
  state.historyData.push(0);
}

// ==========================================
// Web Audio API Synthesizer (Warning Chime)
// ==========================================
function playAlertChime() {
  if (!state.soundEnabled) return;
  const now = Date.now();
  if (now - state.lastAlertSoundTime < 2500) return; // Cooldown between audio chimes
  state.lastAlertSoundTime = now;

  try {
    const AudioContext = window.AudioContext || window.webkitAudioContext;
    if (!AudioContext) return;
    if (!state.audioCtx) {
      state.audioCtx = new AudioContext();
    }
    if (state.audioCtx.state === 'suspended') {
      state.audioCtx.resume();
    }

    const ctx = state.audioCtx;
    const osc1 = ctx.createOscillator();
    const osc2 = ctx.createOscillator();
    const gain = ctx.createGain();

    osc1.type = 'sine';
    osc1.frequency.setValueAtTime(880, ctx.currentTime); // A5
    osc1.frequency.exponentialRampToValueAtTime(440, ctx.currentTime + 0.3); // Drop to A4

    osc2.type = 'triangle';
    osc2.frequency.setValueAtTime(1174.66, ctx.currentTime); // D6
    osc2.frequency.exponentialRampToValueAtTime(587.33, ctx.currentTime + 0.3);

    gain.gain.setValueAtTime(0.15, ctx.currentTime);
    gain.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + 0.35);

    osc1.connect(gain);
    osc2.connect(gain);
    gain.connect(ctx.destination);

    osc1.start();
    osc2.start();
    osc1.stop(ctx.currentTime + 0.35);
    osc2.stop(ctx.currentTime + 0.35);
  } catch (err) {
    console.warn("Audio playback blocked by browser policy:", err);
  }
}

// ==========================================
// Real-Time Kinematics Graph Drawing
// ==========================================
function drawMovementGraph() {
  const canvas = elements.chartCanvas;
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const w = canvas.width;
  const h = canvas.height;

  ctx.clearRect(0, 0, w, h);

  // Background grid lines
  ctx.strokeStyle = '#f1f5f9';
  ctx.lineWidth = 1;
  const maxScore = 120;

  for (let s = 20; s <= 100; s += 20) {
    const y = h - (s / maxScore) * (h - 20) - 10;
    ctx.beginPath();
    ctx.moveTo(35, y);
    ctx.lineTo(w, y);
    ctx.stroke();

    ctx.fillStyle = '#94a3b8';
    ctx.font = '10px JetBrains Mono';
    ctx.textAlign = 'right';
    ctx.fillText(s, 28, y + 3);
  }

  // Draw Rapid Threshold Line (Dashed Crimson)
  const threshY = h - (state.rapidThreshold / maxScore) * (h - 20) - 10;
  ctx.save();
  ctx.strokeStyle = '#ef4444';
  ctx.lineWidth = 1.5;
  ctx.setLineDash([4, 4]);
  ctx.beginPath();
  ctx.moveTo(35, threshY);
  ctx.lineTo(w, threshY);
  ctx.stroke();
  ctx.restore();

  // Draw Movement History Path
  const data = state.historyData;
  const stepX = (w - 40) / (data.length - 1);

  // Gradient fill under the curve
  const grad = ctx.createLinearGradient(0, 0, 0, h);
  grad.addColorStop(0, 'rgba(59, 130, 246, 0.25)');
  grad.addColorStop(1, 'rgba(59, 130, 246, 0.0)');

  ctx.beginPath();
  data.forEach((val, i) => {
    const x = 35 + i * stepX;
    const clamped = Math.min(val, maxScore);
    const y = h - (clamped / maxScore) * (h - 20) - 10;
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });

  // Stroke movement line
  ctx.strokeStyle = '#3b82f6';
  ctx.lineWidth = 2.5;
  ctx.stroke();

  // Fill area under line
  ctx.lineTo(35 + (data.length - 1) * stepX, h - 10);
  ctx.lineTo(35, h - 10);
  ctx.closePath();
  ctx.fillStyle = grad;
  ctx.fill();

  // Highlight latest point
  const lastVal = data[data.length - 1] || 0;
  const lastX = 35 + (data.length - 1) * stepX;
  const lastY = h - (Math.min(lastVal, maxScore) / maxScore) * (h - 20) - 10;

  ctx.beginPath();
  ctx.arc(lastX, lastY, 4, 0, Math.PI * 2);
  ctx.fillStyle = lastVal >= state.rapidThreshold ? '#ef4444' : (lastVal >= state.normalThreshold ? '#f59e0b' : '#10b981');
  ctx.fill();
  ctx.strokeStyle = '#ffffff';
  ctx.lineWidth = 2;
  ctx.stroke();
}

// ==========================================
// Status Polling & UI Synchronization
// ==========================================
async function fetchStatus() {
  try {
    const res = await fetch('/api/status');
    if (!res.ok) return;
    const data = await res.json();

    // 1. System Status & Demo Badge
    state.isRunning = data.is_running;
    state.demoMode = data.demo_mode;

    if (data.demo_mode) {
      elements.demoBadge.classList.remove('hidden');
      elements.statusPill.className = 'status-pill status-online';
      elements.statusText.textContent = 'ONLINE (DEMO)';
    } else if (data.is_running) {
      elements.demoBadge.classList.add('hidden');
      elements.statusPill.className = 'status-pill status-online';
      elements.statusText.textContent = 'SYSTEM ONLINE';
    } else {
      elements.demoBadge.classList.add('hidden');
      elements.statusPill.className = 'status-pill status-standby';
      elements.statusText.textContent = data.system || 'SYSTEM STANDBY';
    }

    // 2. Camera or Model Errors
    if (data.camera_error || (!data.model_loaded && data.is_running && !data.demo_mode)) {
      elements.errorBanner.classList.remove('hidden');
      elements.errorText.textContent = data.camera_error || data.model_error || 'Camera offline or YOLO model missing.';
    } else {
      elements.errorBanner.classList.add('hidden');
    }

    // 3. Telemetry Numbers
    elements.valFps.textContent = Math.round(data.fps || 0);
    elements.valPeopleCount.textContent = data.people || 0;
    elements.badgeTrackCount.textContent = `${data.people || 0} Active`;
    elements.valPrimaryScore.textContent = (data.current_movement_score || 0.0).toFixed(1);

    // Calculate Average Confidence
    if (data.persons && data.persons.length > 0) {
      const avgConf = data.persons.reduce((acc, p) => acc + (p.confidence || 0), 0) / data.persons.length;
      elements.valConfidence.textContent = `${Math.round(avgConf * 100)}%`;
    } else {
      elements.valConfidence.textContent = '0%';
    }

    // 4. Primary Movement State Badge
    const curState = data.current_state || 'NORMAL';
    elements.valPrimaryState.textContent = curState;
    if (curState === 'RAPID MOVEMENT') {
      elements.valPrimaryState.className = 'state-pill pill-rapid';
    } else if (curState === 'ACTIVE') {
      elements.valPrimaryState.className = 'state-pill pill-active';
    } else {
      elements.valPrimaryState.className = 'state-pill pill-normal';
    }

    // 5. Update Movement History for Graph
    state.historyData.push(data.current_movement_score || 0);
    if (state.historyData.length > state.maxHistoryLength) {
      state.historyData.shift();
    }
    drawMovementGraph();

    // 6. Person Matrix Rendering
    renderPersonList(data.persons || []);

    // 7. Alert Banner & Sound Chime Trigger
    if (data.alert) {
      elements.alertBanner.classList.remove('hidden');
      const targetPid = data.alert_person_id ? `Person ${String(data.alert_person_id).padStart(2, '0')}` : 'Detected Subject';
      elements.alertDescText.textContent = `High-velocity kinematic displacement confirmed on ${targetPid}`;
      playAlertChime();
    } else {
      elements.alertBanner.classList.add('hidden');
    }

    // 8. Threshold Sync from backend config
    if (data.config) {
      state.rapidThreshold = data.config.rapid_threshold;
      state.normalThreshold = data.config.normal_threshold;
      elements.lblRapidThresh.textContent = data.config.rapid_threshold;
    }

  } catch (err) {
    // Graceful offline degradation
  }
}

function renderPersonList(persons) {
  const container = elements.personContainer;
  if (!container) return;

  if (persons.length === 0) {
    container.innerHTML = `
      <div class="empty-state">
        <p>No persons currently detected in frame</p>
        <span class="empty-sub">Start camera or activate Demo Mode to simulate subjects</span>
      </div>
    `;
    return;
  }

  let html = '';
  persons.forEach(p => {
    let pillClass = 'pill-normal';
    if (p.state === 'RAPID MOVEMENT') pillClass = 'pill-rapid';
    else if (p.state === 'ACTIVE') pillClass = 'pill-active';

    html += `
      <div class="person-card-item">
        <div class="person-info-left">
          <span class="person-id-label">Person ${String(p.id).padStart(2, '0')}</span>
          <span class="person-meta">Score: ${p.movement_score.toFixed(1)} | Conf: ${Math.round(p.confidence * 100)}%</span>
        </div>
        <span class="state-pill ${pillClass}">${p.state}</span>
      </div>
    `;
  });
  container.innerHTML = html;
}

// ==========================================
// User Actions & Event Handlers
// ==========================================

async function handleStart() {
  if (state.audioCtx && state.audioCtx.state === 'suspended') {
    state.audioCtx.resume();
  }
  try {
    const res = await fetch('/api/start', { method: 'POST' });
    const data = await res.json();
    if (!data.success) {
      // If server camera fails (e.g. cloud environment without physical USB webcam),
      // prompt to switch to Demo Mode or in-browser camera
      console.warn("Server camera unavailable:", data.message);
    }
  } catch (err) {
    console.error(err);
  }
  fetchStatus();
}

async function handleStop() {
  try {
    await fetch('/api/stop', { method: 'POST' });
  } catch (err) {
    console.error(err);
  }
  fetchStatus();
}

async function handleDemoToggle() {
  if (state.audioCtx && state.audioCtx.state === 'suspended') {
    state.audioCtx.resume();
  }
  try {
    await fetch('/api/demo-toggle', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ enable: !state.demoMode })
    });
  } catch (err) {
    console.error(err);
  }
  fetchStatus();
}

async function handleResetAlert() {
  try {
    await fetch('/api/reset-alert', { method: 'POST' });
    elements.alertBanner.classList.add('hidden');
  } catch (err) {
    console.error(err);
  }
  fetchStatus();
}

async function handleSaveCalibration() {
  const norm = parseFloat(elements.sliderNormal.value);
  const rap = parseFloat(elements.sliderRapid.value);
  const smooth = parseInt(elements.sliderSmoothing.value, 10);
  const conf = parseInt(elements.sliderConfirm.value, 10);

  try {
    const res = await fetch('/api/calibrate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        normal_threshold: norm,
        rapid_threshold: rap,
        smoothing_window: smooth,
        rapid_confirmation_frames: conf
      })
    });
    const data = await res.json();
    if (data.success) {
      state.normalThreshold = norm;
      state.rapidThreshold = rap;
      state.smoothingWindow = smooth;
      state.confirmationFrames = conf;
      elements.lblRapidThresh.textContent = rap;
      drawMovementGraph();
    }
  } catch (err) {
    console.error("Calibration update failed", err);
  }
}

// Setup Event Listeners
function setupEventListeners() {
  elements.btnStart.addEventListener('click', handleStart);
  elements.btnStop.addEventListener('click', handleStop);
  elements.btnDemo.addEventListener('click', handleDemoToggle);
  elements.btnResetAlert.addEventListener('click', handleResetAlert);
  elements.btnBannerReset.addEventListener('click', handleResetAlert);
  elements.btnErrorDemo.addEventListener('click', () => {
    fetch('/api/demo-toggle', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ enable: true })
    }).then(fetchStatus);
  });

  elements.chkSound.addEventListener('change', (e) => {
    state.soundEnabled = e.target.checked;
    if (state.soundEnabled && state.audioCtx && state.audioCtx.state === 'suspended') {
      state.audioCtx.resume();
    }
  });

  // Calibration slider live labels
  elements.sliderNormal.addEventListener('input', (e) => {
    elements.lblValNormal.textContent = parseFloat(e.target.value).toFixed(1);
  });
  elements.sliderRapid.addEventListener('input', (e) => {
    elements.lblValRapid.textContent = parseFloat(e.target.value).toFixed(1);
  });
  elements.sliderSmoothing.addEventListener('input', (e) => {
    elements.lblValSmoothing.textContent = `${e.target.value} frames`;
  });
  elements.sliderConfirm.addEventListener('input', (e) => {
    elements.lblValConfirm.textContent = `${e.target.value} frames`;
  });
  elements.btnSaveCal.addEventListener('click', handleSaveCalibration);

  // Resume AudioContext on first user interaction anywhere
  document.addEventListener('click', () => {
    if (state.audioCtx && state.audioCtx.state === 'suspended') {
      state.audioCtx.resume();
    }
  }, { once: true });
}

// Start polling loop
document.addEventListener('DOMContentLoaded', () => {
  setupEventListeners();
  drawMovementGraph();
  fetchStatus();
  setInterval(fetchStatus, 150);
});
