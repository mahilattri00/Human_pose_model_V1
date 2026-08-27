export type MovementState = 'NORMAL' | 'ACTIVE' | 'RAPID MOVEMENT';

export interface Keypoint {
  index: number;
  name: string;
  x: number;
  y: number;
  confidence: number;
  is_valid: boolean;
}

export interface TrackedPerson {
  id: number;
  state: MovementState;
  movement_score: number;
  raw_score: number;
  confidence: number;
  bbox?: [number, number, number, number];
  center?: [number, number];
  keypoints?: Keypoint[];
}

export interface SystemStatus {
  system: string;
  is_running: boolean;
  demo_mode: boolean;
  camera_error: string | null;
  model_loaded: boolean;
  model_error: string | null;
  people: number;
  fps: number;
  alert: boolean;
  alert_person_id: number | null;
  current_state: MovementState;
  current_movement_score: number;
  persons: TrackedPerson[];
  config: {
    normal_threshold: number;
    rapid_threshold: number;
    smoothing_window: number;
    rapid_confirmation_frames: number;
  };
}
