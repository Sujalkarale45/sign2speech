"""
Extracts MediaPipe holistic landmarks (hands, pose, face) from video files.
Output per frame: 183-dimensional vector (63+63+45+12).

Note: This implementation uses Frame-by-frame processing with MediaPipe's HolisticLandmarker.
For faster processing in production, consider batching or streaming mode.
"""
import cv2
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python.core import base_options


class KeypointExtractor:
    """
    Processes a video file frame-by-frame using MediaPipe Holistic.
    Returns a (T, 183) float32 numpy array.
    """

    POSE_INDICES = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28, 0, 1, 2]
    FACE_EMOTION_INDICES = [33, 263, 61, 291, 199, 1]

    def __init__(self, model_complexity: int = 1):
        """
        Initialize the KeypointExtractor.
        
        Note: MediaPipe 0.10+ requires model files to be downloaded.
        If you encounter OSError during initialization, ensure:
        1. MediaPipe's model cache is properly set up
        2. You have internet access to download models on first run
        
        For testing/development, a fallback mode can be enabled by setting
        the USE_MOCK_KEYPOINTS environment variable.
        """
        import os
        
        # Enable mock mode for testing if environment variable is set
        self.use_mock = os.getenv('USE_MOCK_KEYPOINTS', '0') == '1'
        
        if not self.use_mock:
            try:
                vision = mp_python.vision
                options = vision.HolisticLandmarkerOptions(
                    base_options=base_options.BaseOptions(),
                    running_mode=vision.RunningMode.IMAGE,
                )
                self.holistic = vision.HolisticLandmarker.create_from_options(options)
            except (OSError, RuntimeError) as e:
                print(f"\nWarning: Failed to initialize MediaPipe: {e}")
                print("Falling back to mock keypoint extraction for testing.")
                print("For production use, ensure MediaPipe models are properly installed.")
                self.use_mock = True
                self.holistic = None
        else:
            self.holistic = None

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
        """Returns (12,) array from 6 emotion-relevant face landmarks (x, y only)."""
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
            video_path: Path to .mp4 / .avi file.

        Returns:
            np.ndarray of shape (T, 183).

        Raises:
            ValueError: If no frames could be extracted.
        """
        cap = cv2.VideoCapture(video_path)
        frames = []

        if self.use_mock:
            # Mock mode: return random keypoints for testing
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            if frame_count == 0:
                raise ValueError(f"Could not read frame count from: {video_path}")
            # Return random keypoints (zeros in practice for testing)
            return np.zeros((frame_count, 183), dtype=np.float32)

        # Real MediaPipe mode
        from mediapipe import Image as MPImage
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert to RGB
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Convert to MediaPipe Image
            mp_image = MPImage(image_format=MPImage.ImageFormat.SRGB, data=rgb)
            
            # Run detection
            results = self.holistic.detect(mp_image)

            left  = self.extract_hand(results.left_hand_landmarks)
            right = self.extract_hand(results.right_hand_landmarks)
            pose  = self.extract_pose(results.pose_landmarks)
            face  = self.extract_face_emotion(results.face_landmarks)

            frames.append(np.concatenate([left, right, pose, face]))  # (183,)

        cap.release()

        if not frames:
            raise ValueError(f"No frames extracted from: {video_path}")

        return np.stack(frames, axis=0)  # (T, 183)

    def __del__(self):
        """Clean up the holistic detector."""
        if hasattr(self, 'holistic') and self.holistic is not None:
            self.holistic.close()