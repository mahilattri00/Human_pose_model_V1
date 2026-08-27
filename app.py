"""
HUMANSENSE AI — Flask Application & Streaming Server
Prototype V1: Real-time Human Rapid Movement Detection
"""

import os
import time
import math
import threading
import numpy as np
import cv2
from flask import Flask, render_template, Response, jsonify, request

from pose import PoseEstimator, KEYPOINT_NAMES, SKELETON_CONNECTIONS
from movement_detector import MovementDetector

# ==========================================
# 20. CONFIGURATION SECTION
# ==========================================
CAMERA_INDEX = 0
MAX_PEOPLE = 5
CONFIDENCE_THRESHOLD = 0.40
NORMAL_THRESHOLD = 25.0
RAPID_THRESHOLD = 65.0
SMOOTHING_WINDOW = 5
RAPID_CONFIRMATION_FRAMES = 5
ALERT_COOLDOWN = 3.0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

app = Flask(__name__, template_folder="templates", static_folder="static")

class VideoPipeline:
    """
    Thread-safe video capture, YOLO pose inference, and movement pipeline.
    """
    def __init__(self):
        self.lock = threading.Lock()
        self.is_running = False
        self.demo_mode = False
        self.cap = None
        self.camera_error = None
        self.current_frame_bytes = None
        self.fps = 0.0
        self.last_frame_time = time.time()
        self.system_status = "STANDBY"
        
        # Initialize Pose Estimator & Movement Detector
        model_path = os.path.join("models", "yolo26n-pose.pt")
        self.pose_estimator = PoseEstimator(model_path=model_path, conf_threshold=CONFIDENCE_THRESHOLD)
        self.movement_detector = MovementDetector(
            max_people=MAX_PEOPLE,
            normal_threshold=NORMAL_THRESHOLD,
            rapid_threshold=RAPID_THRESHOLD,
            smoothing_window=SMOOTHING_WINDOW,
            rapid_confirmation_frames=RAPID_CONFIRMATION_FRAMES,
            alert_cooldown=ALERT_COOLDOWN
        )
        
        self.current_persons = []
        self.thread = None
        self.demo_step = 0

    def start_camera(self):
        with self.lock:
            if self.is_running and not self.demo_mode:
                return True, "Camera is already running."
            
            self.demo_mode = False
            self.camera_error = None
            try:
                self.cap = cv2.VideoCapture(CAMERA_INDEX)
                # Try setting resolution
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
                
                if not self.cap.isOpened():
                    self.camera_error = f"Unable to open video capture on camera index {CAMERA_INDEX}. Verify USB camera connection or switch to DEMO MODE."
                    self.system_status = "CAMERA ERROR"
                    return False, self.camera_error

                self.is_running = True
                self.system_status = "ONLINE"
            except Exception as e:
                self.camera_error = f"Camera initialization failed: {str(e)}"
                self.system_status = "CAMERA ERROR"
                return False, self.camera_error

        if not self.thread or not self.thread.is_alive():
            self.thread = threading.Thread(target=self._capture_loop, daemon=True)
            self.thread.start()

        return True, "Camera started successfully."

    def stop_camera(self):
        with self.lock:
            self.is_running = False
            self.demo_mode = False
            self.system_status = "STANDBY"
            if self.cap:
                try:
                    self.cap.release()
                except Exception:
                    pass
                self.cap = None
            self.current_persons = []
        return True, "Camera stopped."

    def toggle_demo_mode(self, enable=None):
        with self.lock:
            if enable is None:
                self.demo_mode = not self.demo_mode
            else:
                self.demo_mode = bool(enable)

            if self.demo_mode:
                self.is_running = True
                self.camera_error = None
                self.system_status = "ONLINE (DEMO)"
                if self.cap:
                    try:
                        self.cap.release()
                    except Exception:
                        pass
                    self.cap = None
            else:
                self.is_running = False
                self.system_status = "STANDBY"

        if self.demo_mode and (not self.thread or not self.thread.is_alive()):
            self.thread = threading.Thread(target=self._capture_loop, daemon=True)
            self.thread.start()

        return True, f"Demo mode {'enabled' if self.demo_mode else 'disabled'}."

    def _generate_synthetic_frame(self):
        """
        Creates a synthetic simulation frame with realistic animated people
        cycling between NORMAL, ACTIVE, and RAPID MOVEMENT states.
        """
        self.demo_step += 1
        t = self.demo_step * 0.08
        
        # Dark canvas with macOS grid
        frame = np.full((FRAME_HEIGHT, FRAME_WIDTH, 3), (24, 27, 34), dtype=np.uint8)
        
        # Subtle grid background
        for x in range(0, FRAME_WIDTH, 40):
            cv2.line(frame, (x, 0), (x, FRAME_HEIGHT), (32, 36, 45), 1)
        for y in range(0, FRAME_HEIGHT, 40):
            cv2.line(frame, (0, y), (FRAME_WIDTH, y), (32, 36, 45), 1)

        # Person 1 (Normal / Active wanderer)
        p1_x = 220 + int(math.sin(t * 0.5) * 40)
        p1_y = 260 + int(math.cos(t * 0.5) * 15)
        
        # Person 2 (Periodic rapid mover: jumping / fast motion)
        rapid_phase = (math.sin(t * 0.3) > 0.4)
        rapid_speed = 3.5 if rapid_phase else 0.8
        p2_x = 440 + int(math.sin(t * rapid_speed) * (60 if rapid_phase else 20))
        p2_y = 250 + int(math.cos(t * rapid_speed * 1.5) * (40 if rapid_phase else 10))

        simulated_people = [
            {"center": (p1_x, p1_y), "scale": 0.9, "intensity": 0.8},
            {"center": (p2_x, p2_y), "scale": 1.0, "intensity": 3.0 if rapid_phase else 0.9}
        ]

        detections = []
        for i, p in enumerate(simulated_people):
            cx, cy = p["center"]
            w, h = int(90 * p["scale"]), int(200 * p["scale"])
            x1, y1 = max(0, cx - w // 2), max(0, cy - h // 2)
            x2, y2 = min(FRAME_WIDTH, cx + w // 2), min(FRAME_HEIGHT, cy + h // 2)
            diag = math.sqrt(w ** 2 + h ** 2)
            
            # Construct 17 synthetic keypoints matching anatomical positions
            kpts = []
            # Head
            kpts.append({"index": 0, "name": "nose", "x": cx, "y": cy - h * 0.4, "confidence": 0.95, "is_valid": True})
            kpts.append({"index": 1, "name": "left_eye", "x": cx - 5, "y": cy - h * 0.42, "confidence": 0.95, "is_valid": True})
            kpts.append({"index": 2, "name": "right_eye", "x": cx + 5, "y": cy - h * 0.42, "confidence": 0.95, "is_valid": True})
            kpts.append({"index": 3, "name": "left_ear", "x": cx - 12, "y": cy - h * 0.4, "confidence": 0.92, "is_valid": True})
            kpts.append({"index": 4, "name": "right_ear", "x": cx + 12, "y": cy - h * 0.4, "confidence": 0.92, "is_valid": True})
            # Shoulders
            kpts.append({"index": 5, "name": "left_shoulder", "x": cx - 22, "y": cy - h * 0.28, "confidence": 0.96, "is_valid": True})
            kpts.append({"index": 6, "name": "right_shoulder", "x": cx + 22, "y": cy - h * 0.28, "confidence": 0.96, "is_valid": True})
            # Arms
            arm_wiggle = math.sin(t * p["intensity"] * 4) * 20
            kpts.append({"index": 7, "name": "left_elbow", "x": cx - 35, "y": cy - h * 0.15 + arm_wiggle, "confidence": 0.93, "is_valid": True})
            kpts.append({"index": 8, "name": "right_elbow", "x": cx + 35, "y": cy - h * 0.15 - arm_wiggle, "confidence": 0.93, "is_valid": True})
            kpts.append({"index": 9, "name": "left_wrist", "x": cx - 40, "y": cy + arm_wiggle * 1.5, "confidence": 0.91, "is_valid": True})
            kpts.append({"index": 10, "name": "right_wrist", "x": cx + 40, "y": cy - arm_wiggle * 1.5, "confidence": 0.91, "is_valid": True})
            # Hips
            kpts.append({"index": 11, "name": "left_hip", "x": cx - 16, "y": cy + h * 0.05, "confidence": 0.94, "is_valid": True})
            kpts.append({"index": 12, "name": "right_hip", "x": cx + 16, "y": cy + h * 0.05, "confidence": 0.94, "is_valid": True})
            # Legs
            leg_wiggle = math.cos(t * p["intensity"] * 3) * 12
            kpts.append({"index": 13, "name": "left_knee", "x": cx - 18, "y": cy + h * 0.26 + leg_wiggle, "confidence": 0.95, "is_valid": True})
            kpts.append({"index": 14, "name": "right_knee", "x": cx + 18, "y": cy + h * 0.26 - leg_wiggle, "confidence": 0.95, "is_valid": True})
            kpts.append({"index": 15, "name": "left_ankle", "x": cx - 20, "y": cy + h * 0.46, "confidence": 0.90, "is_valid": True})
            kpts.append({"index": 16, "name": "right_ankle", "x": cx + 20, "y": cy + h * 0.46, "confidence": 0.90, "is_valid": True})

            detections.append({
                "bbox": [x1, y1, x2, y2],
                "bbox_conf": 0.94,
                "center": (cx, cy),
                "diagonal": diag,
                "width": w,
                "height": h,
                "keypoints": kpts
            })

        # Process through kinematics engine
        processed = self.movement_detector.process_detections(detections)
        self.current_persons = processed

        # Draw overlays
        for det in detections:
            pid = det.get("person_id", 1)
            state = det.get("state", "NORMAL")
            self.pose_estimator.draw_skeleton(frame, pid, det, state=state)

        # Simulation watermark banner
        cv2.rectangle(frame, (10, 10), (320, 36), (15, 23, 42), -1)
        cv2.putText(frame, "DEMO MODE - SIMULATED DATA", (18, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 215, 255), 1, cv2.LINE_AA)
        
        return frame

    def _capture_loop(self):
        """Continuous inference worker loop."""
        while self.is_running:
            start_t = time.time()
            frame = None

            if self.demo_mode:
                frame = self._generate_synthetic_frame()
                time.sleep(0.033) # ~30 FPS
            else:
                if not self.cap or not self.cap.isOpened():
                    self.camera_error = "Camera capture disconnected."
                    break

                ret, raw_frame = self.cap.read()
                if not ret or raw_frame is None:
                    self.camera_error = "Failed to capture frame from webcam."
                    time.sleep(0.1)
                    continue

                frame = cv2.resize(raw_frame, (FRAME_WIDTH, FRAME_HEIGHT))

                # Step 1-4: Run YOLO Pose Estimation
                if self.pose_estimator.is_loaded:
                    detections = self.pose_estimator.estimate(frame)
                    # Step 5-9: Kinematic movement & state classification
                    processed = self.movement_detector.process_detections(detections)
                    self.current_persons = processed

                    # Step 3 & 16: Draw skeleton and HUD
                    for det in detections:
                        pid = det.get("person_id", 1)
                        state = det.get("state", "NORMAL")
                        self.pose_estimator.draw_skeleton(frame, pid, det, state=state)
                else:
                    # Model not loaded error overlay
                    err_msg = self.pose_estimator.load_error or "Model missing"
                    cv2.putText(frame, "YOLO MODEL ERROR", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    cv2.putText(frame, err_msg[:50], (30, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

            # Calculate FPS
            elapsed = time.time() - start_t
            if elapsed > 0:
                cur_fps = 1.0 / elapsed
                self.fps = round(self.fps * 0.85 + cur_fps * 0.15, 1)

            # Encode as JPEG
            ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if ret:
                with self.lock:
                    self.current_frame_bytes = buffer.tobytes()

    def get_frame(self):
        with self.lock:
            if self.current_frame_bytes is not None:
                return self.current_frame_bytes
        
        # Fallback offline frame
        blank = np.full((FRAME_HEIGHT, FRAME_WIDTH, 3), (20, 24, 33), dtype=np.uint8)
        cv2.putText(blank, "CAMERA OFFLINE", (FRAME_WIDTH//2 - 90, FRAME_HEIGHT//2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (150, 150, 150), 1, cv2.LINE_AA)
        cv2.putText(blank, "Click 'START CAMERA' or 'DEMO MODE'", (FRAME_WIDTH//2 - 140, FRAME_HEIGHT//2 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 100, 100), 1, cv2.LINE_AA)
        ret, buf = cv2.imencode('.jpg', blank)
        return buf.tobytes()

# Global pipeline instance
pipeline = VideoPipeline()

# ==========================================
# 13. BACKEND ENDPOINTS
# ==========================================

@app.route("/")
def index():
    return render_template("index.html")

def generate_video_stream():
    """MJPEG streaming generator."""
    while True:
        frame_bytes = pipeline.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        time.sleep(0.033)

@app.route("/video_feed")
def video_feed():
    return Response(
        generate_video_stream(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )

@app.route("/api/status")
def get_status():
    with pipeline.lock:
        persons = pipeline.current_persons
        people_count = len(persons)
        alert_active = pipeline.movement_detector.alert_active
        alert_pid = pipeline.movement_detector.alert_trigger_person_id
        
        # Determine overall max movement score and predominant state
        max_score = 0.0
        current_state = "NORMAL"
        if persons:
            max_score = max(p["movement_score"] for p in persons)
            if any(p["state"] == "RAPID MOVEMENT" for p in persons):
                current_state = "RAPID MOVEMENT"
            elif any(p["state"] == "ACTIVE" for p in persons):
                current_state = "ACTIVE"

        return jsonify({
            "system": pipeline.system_status,
            "is_running": pipeline.is_running,
            "demo_mode": pipeline.demo_mode,
            "camera_error": pipeline.camera_error,
            "model_loaded": pipeline.pose_estimator.is_loaded,
            "model_error": pipeline.pose_estimator.load_error,
            "people": people_count,
            "fps": pipeline.fps,
            "alert": alert_active,
            "alert_person_id": alert_pid,
            "current_state": current_state,
            "current_movement_score": max_score,
            "persons": persons,
            "config": {
                "normal_threshold": pipeline.movement_detector.normal_threshold,
                "rapid_threshold": pipeline.movement_detector.rapid_threshold,
                "smoothing_window": pipeline.movement_detector.smoothing_window,
                "rapid_confirmation_frames": pipeline.movement_detector.rapid_confirmation_frames
            }
        })

@app.route("/api/start", methods=["POST"])
def start_camera():
    success, message = pipeline.start_camera()
    return jsonify({"success": success, "message": message})

@app.route("/api/stop", methods=["POST"])
def stop_camera():
    success, message = pipeline.stop_camera()
    return jsonify({"success": success, "message": message})

@app.route("/api/demo-toggle", methods=["POST"])
def toggle_demo():
    data = request.get_json(silent=True) or {}
    enable = data.get("enable", None)
    success, message = pipeline.toggle_demo_mode(enable)
    return jsonify({"success": success, "message": message, "demo_mode": pipeline.demo_mode})

@app.route("/api/reset-alert", methods=["POST"])
def reset_alert():
    pipeline.movement_detector.reset_alert()
    return jsonify({"success": True, "message": "Alert reset."})

@app.route("/api/calibrate", methods=["POST"])
def calibrate():
    data = request.get_json(silent=True) or {}
    pipeline.movement_detector.update_config(
        normal_thresh=data.get("normal_threshold"),
        rapid_thresh=data.get("rapid_threshold"),
        smoothing_win=data.get("smoothing_window"),
        conf_frames=data.get("rapid_confirmation_frames")
    )
    return jsonify({
        "success": True,
        "config": {
            "normal_threshold": pipeline.movement_detector.normal_threshold,
            "rapid_threshold": pipeline.movement_detector.rapid_threshold,
            "smoothing_window": pipeline.movement_detector.smoothing_window,
            "rapid_confirmation_frames": pipeline.movement_detector.rapid_confirmation_frames
        }
    })

if __name__ == "__main__":
    print("==================================================")
    print(" HUMANSENSE AI — Human Rapid Movement Detection ")
    print(" Prototype V1 (Flask + OpenCV + YOLO26n-Pose)     ")
    print(" Dashboard URL: http://127.0.0.1:5000             ")
    print("==================================================")
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
