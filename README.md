# HUMANSENSE AI — Human Rapid Movement Detection (Prototype V1)

HUMANSENSE AI is a real-time computer-vision web application that demonstrates human rapid-movement detection from an RGB webcam stream using YOLO26n-Pose keypoint estimation and normalized kinematic mathematics.

---

## 1. Project Overview & Pipeline

The system processes video frames through a mathematical movement pipeline without requiring heavy action-recognition models or additional neural network training:

```
PC WEBCAM / DEMO STREAM
          ↓
       OpenCV
          ↓
    YOLO26n-Pose (Ultralytics)
          ↓
  17 Body Keypoints Extraction
          ↓
  Nearest-Neighbor Person Tracking
          ↓
  Scale-Normalized Displacement (dx, dy)
          ↓
  Temporal Smoothing (Moving Average)
          ↓
  State Threshold Comparison (NORMAL / ACTIVE / RAPID)
          ↓
  Sustained Rapid Movement Confirmation
          ↓
  Web Dashboard UI + Alert & Synthesizer Tone
```

---

## 2. Project Folder Structure

```
HUMANSENSE_AI/
├── app.py                   # Flask server, MJPEG streaming & API endpoints
├── pose.py                  # YOLO26n-Pose wrapper & 17-keypoint skeleton renderer
├── movement_detector.py     # Kinematics engine, normalization, smoothing & alerts
├── requirements.txt         # Python package dependencies
├── README.md                # Setup, calibration & execution guide
│
├── models/
│   ├── README.md            # Model directory guide
│   └── yolo26n-pose.pt      # Ultralytics YOLO26n-Pose weights file
│
├── templates/
│   └── index.html           # macOS-inspired scientific monitoring dashboard
│
└── static/
    ├── css/
    │   └── style.css        # Dashboard styling & visual state indicators
    └── js/
        └── app.js           # Client controller, live graph & audio synthesizer
```

---

## 3. Installation & Setup

### Prerequisites
- Python 3.10+ or Python 3.11+
- USB Webcam or integrated laptop RGB camera
- Modern Web Browser (Chrome, Edge, Safari, Firefox)

### Step A: Create and Activate a Virtual Environment

**On Linux / macOS:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**On Windows (Command Prompt / PowerShell):**
```cmd
python -m venv venv
venv\Scripts\activate
```

### Step B: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step C: Model Placement
Place your `yolo26n-pose.pt` weights file in the `models/` directory:
```
models/yolo26n-pose.pt
```

*(If the model file is not present or you are testing without hardware, the application will display a clear notification and allows full operation using DEMO MODE.)*

---

## 4. Running the Application

Start the Flask application server:
```bash
python app.py
```

Open your web browser and navigate to:
```
http://127.0.0.1:5000
```

---

## 5. How Movement Detection Works (Kinematic Mathematics)

Unlike black-box video classifiers, HUMANSENSE AI computes transparent anatomical kinematics:

1. **Keypoint Tracking**: On each frame $t$, the system records $(x_{t,k}, y_{t,k})$ for all 17 COCO keypoints with confidence $\ge 0.40$.
2. **Displacement Calculation**: For each reliable keypoint tracked from frame $t-1$:
   $$\Delta x_k = x_{t,k} - x_{t-1,k}$$
   $$\Delta y_k = y_{t,k} - y_{t-1,k}$$
   $$d_k = \sqrt{\Delta x_k^2 + \Delta y_k^2}$$
3. **Scale Normalization**:
   To prevent false alarms caused by people walking closer to the camera vs. farther away, displacement is normalized by the bounding box diagonal:
   $$D_{\text{bbox}} = \sqrt{\text{width}^2 + \text{height}^2}$$
   $$d_{\text{norm}, k} = \left(\frac{d_k}{D_{\text{bbox}}}\right) \times 1000$$
4. **Frame Movement Score**: Average normalized displacement across all valid joints.
5. **Smoothing**: Filtered via moving average over $N=5$ frames to eliminate camera sensor jitter.
6. **Classification & Confirmation**:
   - $\text{Score} < 25.0 \implies \textbf{NORMAL}$
   - $25.0 \le \text{Score} < 65.0 \implies \textbf{ACTIVE}$
   - $\text{Score} \ge 65.0 \implies \text{Candidate Rapid Movement}$
   - Rapid movement is **confirmed** only when sustained for $\ge 5$ consecutive frames ($\sim 160\text{ ms}$).

---

## 6. Calibration Procedure

Camera angle, field of view, and distance vary across rooms. Use the **Threshold Calibration** panel in the web UI:

1. **Baseline Standing**: Stand in front of the camera and remain still. Observe that the score stays $< 15.0$.
2. **Normal Walking**: Walk at a standard pace. Observe that scores range between $20.0 - 45.0$ (**NORMAL / ACTIVE**).
3. **Rapid Action**: Perform fast wave, jump, or quick sprint. Observe peak scores exceeding $70.0+$.
4. **Adjust Thresholds**: Use the sliders in the dashboard to set `RAPID_THRESHOLD` just above your brisk walking baseline.

*Note: Thresholds are empirical kinematic approximations for Prototype V1 and are not certified for medical or critical life-safety applications.*

---

## 7. Troubleshooting

- **Camera Error / Blank Feed**: Ensure no other application (Zoom, Teams, etc.) is holding a lock on camera index 0. You can also click **DEMO MODE** to inspect all dashboard features without a camera.
- **Model File Error**: Verify that `yolo26n-pose.pt` is inside `models/yolo26n-pose.pt`.
- **Audio Tone Not Playing**: Click anywhere on the dashboard to grant the browser Web Audio permission.

---

## 8. Stopping the Application

Press `Ctrl + C` in the terminal running `app.py`.
