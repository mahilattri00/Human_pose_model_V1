"""
HUMANSENSE AI — Movement Detector & Kinematic Mathematics Module
Calculates normalized multi-person keypoint displacements, applies smoothing filters,
classifies movement states, and confirms rapid sustained movement.
"""

import math
import time
from collections import deque
from typing import Dict, List, Optional, Tuple

class PersonTrack:
    """
    Tracks state and movement metrics for an individual person.
    """
    def __init__(self, person_id: int, smoothing_window: int = 5):
        self.person_id = person_id
        self.smoothing_window = smoothing_window
        self.center = (0.0, 0.0)
        self.prev_keypoints: Dict[int, Tuple[float, float]] = {}
        self.movement_history: deque = deque(maxlen=smoothing_window)
        self.raw_movement: float = 0.0
        self.smoothed_movement: float = 0.0
        self.confidence: float = 0.0
        self.state: str = "NORMAL"
        self.rapid_frame_counter: int = 0
        self.last_seen_timestamp: float = time.time()

    def update_pose(
        self,
        center: Tuple[float, float],
        diagonal: float,
        keypoints: List[dict],
        confidence: float,
        normal_threshold: float = 25.0,
        rapid_threshold: float = 65.0,
        rapid_confirmation_frames: int = 5
    ) -> float:
        """
        Calculates normalized displacement across valid keypoints,
        updates smoothed scores, and transitions movement state.

        FORMULA:
        1. For each keypoint with valid detection in both previous and current frames:
           dx = current_x - previous_x
           dy = current_y - previous_y
           displacement = sqrt(dx^2 + dy^2)
        2. Scale normalization:
           norm_displacement = (displacement / bbox_diagonal) * 1000.0
           This scale factor makes movement invariant to distance from the lens.
        3. Raw frame score = Mean of normalized displacements for all tracked joints.
        4. Smoothed score = Moving average over the configured window.
        """
        self.center = center
        self.confidence = confidence
        self.last_seen_timestamp = time.time()

        current_kpts: Dict[int, Tuple[float, float]] = {}
        displacements: List[float] = []

        # Ensure diagonal is non-zero to avoid division by zero
        safe_diagonal = max(diagonal, 10.0)

        for kp in keypoints:
            kp_idx = kp["index"]
            if kp.get("is_valid", False):
                cx, cy = kp["x"], kp["y"]
                current_kpts[kp_idx] = (cx, cy)

                if kp_idx in self.prev_keypoints:
                    px, py = self.prev_keypoints[kp_idx]
                    dx = cx - px
                    dy = cy - py
                    pixel_dist = math.sqrt(dx * dx + dy * dy)

                    # Scale invariant normalization
                    norm_dist = (pixel_dist / safe_diagonal) * 1000.0
                    displacements.append(norm_dist)

        # Store current keypoints for the next frame's comparison
        self.prev_keypoints = current_kpts

        # Calculate raw movement score
        if displacements:
            self.raw_movement = float(sum(displacements) / len(displacements))
        else:
            self.raw_movement = 0.0

        # Movement smoothing (Moving average)
        self.movement_history.append(self.raw_movement)
        self.smoothed_movement = float(sum(self.movement_history) / len(self.movement_history))

        # State determination with rapid movement confirmation
        if self.smoothed_movement >= rapid_threshold:
            self.rapid_frame_counter += 1
            if self.rapid_frame_counter >= rapid_confirmation_frames:
                self.state = "RAPID MOVEMENT"
            else:
                self.state = "ACTIVE"
        elif self.smoothed_movement >= normal_threshold:
            self.rapid_frame_counter = max(0, self.rapid_frame_counter - 1)
            self.state = "ACTIVE"
        else:
            self.rapid_frame_counter = 0
            self.state = "NORMAL"

        return self.smoothed_movement


class MovementDetector:
    """
    Coordinates multi-person tracking (up to MAX_PEOPLE), state updates,
    alert cooldowns, and global telemetry.
    """
    def __init__(
        self,
        max_people: int = 5,
        normal_threshold: float = 25.0,
        rapid_threshold: float = 65.0,
        smoothing_window: int = 5,
        rapid_confirmation_frames: int = 5,
        alert_cooldown: float = 3.0
    ):
        # Configuration section
        self.max_people = max_people
        self.normal_threshold = normal_threshold
        self.rapid_threshold = rapid_threshold
        self.smoothing_window = smoothing_window
        self.rapid_confirmation_frames = rapid_confirmation_frames
        self.alert_cooldown = alert_cooldown

        # Tracking state
        self.tracks: Dict[int, PersonTrack] = {}
        self.next_id = 1
        self.available_ids = list(range(1, max_people + 1))

        # Alert state
        self.alert_active: bool = False
        self.alert_trigger_person_id: Optional[int] = None
        self.last_alert_time: float = 0.0

    def update_config(self, normal_thresh=None, rapid_thresh=None, smoothing_win=None, conf_frames=None):
        """Allow dynamic calibration of thresholds from frontend UI."""
        if normal_thresh is not None:
            self.normal_threshold = float(normal_thresh)
        if rapid_thresh is not None:
            self.rapid_threshold = float(rapid_thresh)
        if smoothing_win is not None:
            self.smoothing_window = int(smoothing_win)
        if conf_frames is not None:
            self.rapid_confirmation_frames = int(conf_frames)

    def process_detections(self, detections: List[dict]) -> List[dict]:
        """
        Matches detected people to existing tracks using nearest-center association.
        Limits tracking to max_people.
        """
        now = time.time()
        # 1. Clean up stale tracks (not seen in > 1.5 seconds)
        stale_ids = [
            pid for pid, track in self.tracks.items()
            if (now - track.last_seen_timestamp) > 1.5
        ]
        for pid in stale_ids:
            del self.tracks[pid]
            if pid not in self.available_ids:
                self.available_ids.append(pid)
                self.available_ids.sort()

        # Limit detections to max_people
        active_detections = detections[:self.max_people]
        assigned_tracks: Dict[int, dict] = {}
        unmatched_detections = []

        # 2. Nearest-neighbor matching
        for det in active_detections:
            center = det["center"]
            best_match_id = None
            min_dist = float("inf")

            for pid, track in self.tracks.items():
                if pid in assigned_tracks:
                    continue
                tx, ty = track.center
                dist = math.sqrt((center[0] - tx) ** 2 + (center[1] - ty) ** 2)
                # Maximum tracking jump threshold (approx 200px)
                if dist < 220 and dist < min_dist:
                    min_dist = dist
                    best_match_id = pid

            if best_match_id is not None:
                assigned_tracks[best_match_id] = det
            else:
                unmatched_detections.append(det)

        # 3. Create new tracks for unmatched detections if slots available
        for det in unmatched_detections:
            if self.available_ids:
                new_id = self.available_ids.pop(0)
                new_track = PersonTrack(new_id, smoothing_window=self.smoothing_window)
                self.tracks[new_id] = new_track
                assigned_tracks[new_id] = det

        # 4. Update kinematic movement on all active matched tracks
        processed_persons = []
        rapid_detected_in_frame = False
        rapid_person_id = None

        for pid, det in assigned_tracks.items():
            track = self.tracks[pid]
            smoothed_score = track.update_pose(
                center=det["center"],
                diagonal=det["diagonal"],
                keypoints=det["keypoints"],
                confidence=det["bbox_conf"],
                normal_threshold=self.normal_threshold,
                rapid_threshold=self.rapid_threshold,
                rapid_confirmation_frames=self.rapid_confirmation_frames
            )

            det["person_id"] = pid
            det["state"] = track.state
            det["movement_score"] = round(smoothed_score, 1)

            if track.state == "RAPID MOVEMENT":
                rapid_detected_in_frame = True
                rapid_person_id = pid

            processed_persons.append({
                "id": pid,
                "state": track.state,
                "movement_score": round(smoothed_score, 1),
                "raw_score": round(track.raw_movement, 1),
                "confidence": round(track.confidence, 2)
            })

        # 5. Alert state management
        if rapid_detected_in_frame:
            if (now - self.last_alert_time) > self.alert_cooldown:
                self.alert_active = True
                self.alert_trigger_person_id = rapid_person_id
                self.last_alert_time = now
        elif not any(t.state == "RAPID MOVEMENT" for t in self.tracks.values()):
            # Auto clear alert after 2.5 seconds if nobody is in rapid state
            if (now - self.last_alert_time) > 2.5:
                self.alert_active = False
                self.alert_trigger_person_id = None

        return processed_persons

    def reset_alert(self):
        """Manually dismiss the active alert."""
        self.alert_active = False
        self.alert_trigger_person_id = None
        self.last_alert_time = 0.0
