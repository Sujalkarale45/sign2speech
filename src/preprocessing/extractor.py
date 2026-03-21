"""
extractor.py
Extracts MediaPipe holistic landmarks from video files.
Uses the legacy Solutions API (mp.solutions.holistic).
Output per frame: 183-dimensional vector (63+63+45+12).
"""
import cv2
import numpy as np
import mediapipe as mp


class KeypointExtractor:
    """
    Processes a video file frame-by-frame using MediaPipe Holistic.
    Returns a (T, 183) float32 numpy array.
    """

    POSE_INDICES         = [11,12,13,14,15,16,23,24,25,26,27,28,0,1,2]
    FACE_EMOTION_INDICES = [33,263,61,291,199,1]

    def __init__(self, model_complexity: int = 1):
        self.mp_holistic = mp.solutions.holistic
        self.holistic    = self.mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=model_complexity,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def extract_hand(self, hand_landmarks) -> np.ndarray:
        """Returns (63,) array for one hand. Zeros if not detected."""
        if hand_landmarks is None:
            return np.zeros(63, dtype=np.float32)
        pts = []
        for lm in hand_landmarks.landmark:
            pts.extend([lm.x, lm.y, lm.z])
        return np.array(pts, dtype=np.float32)

    def extract_pose(self, pose_landmarks) -> np.ndarray:
        """Returns (45,) array from 15 selected pose landmarks."""
        if pose_landmarks is None:
            return np.zeros(45, dtype=np.float32)
        pts = []
        for idx in self.POSE_INDICES:
            lm = pose_landmarks.landmark[idx]
            pts.extend([lm.x, lm.y, lm.z])
        return np.array(pts, dtype=np.float32)

    def extract_face_emotion(self, face_landmarks) -> np.ndarray:
        """Returns (12,) array from 6 emotion face landmarks (x,y only)."""
        if face_landmarks is None:
            return np.zeros(12, dtype=np.float32)
        pts = []
        for idx in self.FACE_EMOTION_INDICES:
            lm = face_landmarks.landmark[idx]
            pts.extend([lm.x, lm.y])
        return np.array(pts, dtype=np.float32)

    def process_video(self, video_path: str) -> np.ndarray:
        """
        Extracts keypoints from all frames in a video.

        Args:
            video_path: Path to .mp4 file.

        Returns:
            np.ndarray of shape (T, 183).

        Raises:
            ValueError: If no frames extracted.
        """
        cap    = cv2.VideoCapture(video_path)
        frames = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.holistic.process(rgb)

            left  = self.extract_hand(results.left_hand_landmarks)
            right = self.extract_hand(results.right_hand_landmarks)
            pose  = self.extract_pose(results.pose_landmarks)
            face  = self.extract_face_emotion(results.face_landmarks)

            frames.append(np.concatenate([left, right, pose, face]))

        cap.release()

        if not frames:
            raise ValueError(f"No frames extracted from: {video_path}")

        return np.stack(frames, axis=0)  # (T, 183)

    def __del__(self):
        if hasattr(self, 'holistic'):
            self.holistic.close()