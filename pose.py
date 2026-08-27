"""
HUMANSENSE AI — Pose Estimation & Keypoint Extraction Module
Handles YOLO26n-Pose model verification, keypoint parsing, and skeleton drawing.
"""

import os
import math
import numpy as np
import cv2

# Standard 17 COCO-style body keypoint indices
KEYPOINT_NAMES = [
    "nose",           # 0
    "left_eye",       # 1
    "right_eye",      # 2
    "left_ear",       # 3
    "right_ear",      # 4
    "left_shoulder",  # 5
    "right_shoulder", # 6
    "left_elbow",     # 7
    "right_elbow",    # 8
    "left_wrist",     # 9
    "right_wrist",    # 10
    "left_hip",       # 11
    "right_hip",      # 12
    "left_knee",      # 13
    "right_knee",     # 14
    "left_ankle",     # 15
    "right_ankle"     # 16
]

# Anatomical skeleton connections connecting keypoint indices
SKELETON_CONNECTIONS = [
    (0, 1), (0, 2),        # Facial features (nose -> eyes)
    (1, 3), (2, 4),        # Facial features (eyes -> ears)
    (5, 6),                # Clavicle / Shoulders
    (5, 7), (7, 9),        # Left Arm (shoulder -> elbow -> wrist)
    (6, 8), (8, 10),       # Right Arm (shoulder -> elbow -> wrist)
    (5, 11), (6, 12),      # Torso lateral bounds
    (11, 12),              # Pelvis / Hips
    (11, 13), (13, 15),    # Left Leg (hip -> knee -> ankle)
    (12, 14), (14, 16)     # Right Leg (hip -> knee -> ankle)
]

class PoseEstimator:
    """
    Wrapper around Ultralytics YOLO Pose model.
    Safely handles loading, verification, and human pose extraction.
    """
    def __init__(self, model_path: str = "models/yolo26n-pose.pt", conf_threshold: float = 0.40):
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.model = None
        self.is_loaded = False
        self.load_error = None
        self._load_model()

    def _load_model(self):
        """Verify model existence and initialize Ultralytics YOLO."""
        if not os.path.exists(self.model_path):
            self.load_error = f"Model file not found at '{self.model_path}'. Please place 'yolo26n-pose.pt' in the models/ directory."
            self.is_loaded = False
            return

        try:
            from ultralytics import YOLO
            self.model = YOLO(self.model_path)
            self.is_loaded = True
            self.load_error = None
            print(f"[HUMANSENSE AI] Loaded pose model successfully from {self.model_path}")
        except Exception as e:
            self.load_error = f"Failed to initialize YOLO model: {str(e)}"
            self.is_loaded = False
            print(f"[HUMANSENSE AI ERROR] {self.load_error}")

    def estimate(self, frame: np.ndarray):
        """
        Run pose estimation on a single BGR OpenCV frame.
        Returns a list of detected person dictionaries.
        """
        if not self.is_loaded or self.model is None:
            return []

        try:
            # Perform YOLO pose inference with low overhead
            results = self.model(frame, conf=self.conf_threshold, verbose=False)
            if not results or len(results) == 0:
                return []

            result = results[0]
            detections = []

            # Check if keypoints and boxes exist in result
            if result.boxes is None or result.keypoints is None:
                return []

            boxes_xyxy = result.boxes.xyxy.cpu().numpy()
            boxes_conf = result.boxes.conf.cpu().numpy()
            keypoints_data = result.keypoints.data.cpu().numpy() # Shape: (N, 17, 3) -> x, y, conf

            for i in range(len(boxes_xyxy)):
                x1, y1, x2, y2 = boxes_xyxy[i]
                bbox_conf = float(boxes_conf[i])
                kpts = keypoints_data[i] # (17, 3)

                width = max(1.0, float(x2 - x1))
                height = max(1.0, float(y2 - y1))
                center_x = float(x1 + width / 2.0)
                center_y = float(y1 + height / 2.0)
                bbox_diagonal = float(math.sqrt(width ** 2 + height ** 2))

                parsed_keypoints = []
                for kp_idx, kp in enumerate(kpts):
                    kx, ky = float(kp[0]), float(kp[1])
                    kc = float(kp[2]) if len(kp) > 2 else 1.0
                    parsed_keypoints.append({
                        "index": kp_idx,
                        "name": KEYPOINT_NAMES[kp_idx] if kp_idx < len(KEYPOINT_NAMES) else f"kp_{kp_idx}",
                        "x": kx,
                        "y": ky,
                        "confidence": kc,
                        "is_valid": bool(kc >= self.conf_threshold and kx > 0 and ky > 0)
                    })

                detections.append({
                    "bbox": [float(x1), float(y1), float(x2), float(y2)],
                    "bbox_conf": bbox_conf,
                    "center": (center_x, center_y),
                    "diagonal": bbox_diagonal,
                    "width": width,
                    "height": height,
                    "keypoints": parsed_keypoints
                })

            return detections
        except Exception as e:
            print(f"[HUMANSENSE AI] Inference warning: {e}")
            return []

    def draw_skeleton(self, frame: np.ndarray, person_id: int, person_data: dict, state: str = "NORMAL"):
        """
        Draw bounding box, 17 keypoints, skeleton links, and person ID tag.
        Uses clean Apple-inspired color aesthetics.
        """
        # State colors (BGR for OpenCV)
        if state == "RAPID MOVEMENT":
            accent_color = (68, 68, 239)   # Vibrant red #EF4444
            text_bg = (50, 50, 200)
        elif state == "ACTIVE":
            accent_color = (11, 158, 245)  # Amber #F59E0B
            text_bg = (10, 130, 200)
        else:
            accent_color = (129, 185, 16)  # Green #10B981
            text_bg = (100, 150, 15)

        bbox = person_data.get("bbox", [0, 0, 0, 0])
        x1, y1, x2, y2 = map(int, bbox)
        keypoints = person_data.get("keypoints", [])

        # 1. Draw rounded/crisp bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), accent_color, 2, cv2.LINE_AA)

        # 2. Draw Skeleton Lines
        kpt_dict = {kp["index"]: (int(kp["x"]), int(kp["y"]), kp["is_valid"]) for kp in keypoints}
        for pt1_idx, pt2_idx in SKELETON_CONNECTIONS:
            if pt1_idx in kpt_dict and pt2_idx in kpt_dict:
                x_a, y_a, valid_a = kpt_dict[pt1_idx]
                x_b, y_b, valid_b = kpt_dict[pt2_idx]
                if valid_a and valid_b:
                    cv2.line(frame, (x_a, y_a), (x_b, y_b), (220, 220, 220), 2, cv2.LINE_AA)
                    cv2.line(frame, (x_a, y_a), (x_b, y_b), accent_color, 1, cv2.LINE_AA)

        # 3. Draw Keypoint Dots
        for kp in keypoints:
            if kp.get("is_valid", False):
                cx, cy = int(kp["x"]), int(kp["y"])
                # Outer ring + solid center
                cv2.circle(frame, (cx, cy), 4, (255, 255, 255), -1, cv2.LINE_AA)
                cv2.circle(frame, (cx, cy), 2, accent_color, -1, cv2.LINE_AA)

        # 4. Header Label (Person ID + Movement State)
        label = f"Person {person_id:02d} | {state}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        label_y = max(y1 - 8, th + 10)
        cv2.rectangle(frame, (x1, label_y - th - 6), (x1 + tw + 10, label_y + 4), (20, 24, 33), -1)
        cv2.rectangle(frame, (x1, label_y - th - 6), (x1 + tw + 10, label_y + 4), accent_color, 1)
        cv2.putText(frame, label, (x1 + 5, label_y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
