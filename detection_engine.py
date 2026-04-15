"""
Detection Engine - Pro-Vision v2.0
Handles face, eye, and drowsiness detection.

Pipeline:
  1. CLAHE preprocessing
  2. Haar-cascade face detection
  3. Eye detection within upper-face ROI
  4. EAR-inspired closure ratio + temporal smoothing
  5. DrowsinessClassifier scoring (0-100)
"""

import cv2
import numpy as np
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
import json


# ---------------------------------------------------------------------------
# Domain types
# ---------------------------------------------------------------------------

class DrowsinessLevel(Enum):
    AWAKE    = "AWAKE"
    WARNING  = "WARNING"
    DROWSY   = "DROWSY"
    OFFLINE  = "OFFLINE"


@dataclass
class DetectionMetrics:
    """All detection results for a single processed frame."""
    status:          DrowsinessLevel
    drowsy_score:    float
    eyes_detected:   int
    faces_detected:  int
    is_blink:        bool
    yawn_detected:   bool
    frame_annotated: np.ndarray = field(default=None, repr=False)


# ---------------------------------------------------------------------------
# Face / eye detector
# ---------------------------------------------------------------------------

class FaceDetector:
    """Wraps OpenCV Haar cascades with CLAHE preprocessing."""

    def __init__(self, config: dict):
        cfg = config["detection"]
        self._cfg = cfg

        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        self.eye_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_eye.xml"
        )
        self.smile_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_smile.xml"
        )
        self.clahe = cv2.createCLAHE(
            clipLimit=cfg["clahe_clip_limit"],
            tileGridSize=(cfg["clahe_tile_size"], cfg["clahe_tile_size"]),
        )

    # --- Frame preprocessing ---

    def preprocess(self, frame: np.ndarray) -> np.ndarray:
        """Convert to grayscale and apply CLAHE."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return self.clahe.apply(gray)

    # --- Detection calls ---

    def detect_faces(self, gray: np.ndarray) -> list:
        cfg = self._cfg
        return self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=cfg["face_scale_factor"],
            minNeighbors=cfg["face_min_neighbors"],
            minSize=(cfg["face_min_size"], cfg["face_min_size"]),
        )

    def detect_eyes(self, roi_gray: np.ndarray, sensitivity: int) -> list:
        return self.eye_cascade.detectMultiScale(
            roi_gray,
            scaleFactor=self._cfg["eye_scale_factor"],
            minNeighbors=sensitivity,
        )

    def detect_yawn(self, roi_gray: np.ndarray) -> list:
        return self.smile_cascade.detectMultiScale(
            roi_gray,
            scaleFactor=1.8,
            minNeighbors=20,
            minSize=(25, 25),
        )


# ---------------------------------------------------------------------------
# Drowsiness classifier
# ---------------------------------------------------------------------------

class DrowsinessClassifier:
    """
    Scores drowsiness 0-100 using temporal smoothing over a history window.

    Eye-open frames pull score down; eye-closed frames push it up.
    Blink detection uses the consecutive-closure count: a very short
    closure (< blink_threshold_frames) followed by re-opening counts
    as a normal blink rather than drowsiness.
    """

    def __init__(self, config: dict):
        self._cfg = config["drowsiness"]
        history_len = self._cfg["history_window"]
        self._eye_history: deque = deque(maxlen=history_len)
        self._score: float = 0.0
        self._consecutive_closed: int = 0
        self._blink_window: deque = deque(maxlen=60)  # last 60 frames

    # ---- public ---

    def update(
        self, eyes_detected: int, faces_detected: int
    ) -> tuple[float, bool]:
        """
        Update state from the latest frame.

        Returns
        -------
        (score, is_blink)
            score    : float 0-100
            is_blink : True on the frame a completed blink is registered
        """
        cfg = self._cfg
        is_closed = faces_detected > 0 and eyes_detected == 0
        is_blink = False

        if is_closed:
            self._consecutive_closed += 1
        else:
            if 0 < self._consecutive_closed <= cfg["blink_threshold_frames"]:
                is_blink = True
            self._consecutive_closed = 0

        self._eye_history.append(1 if is_closed else 0)

        # Rolling closure ratio
        if self._eye_history:
            ratio = sum(self._eye_history) / len(self._eye_history)
        else:
            ratio = 0.0

        # Hysteresis scoring
        if ratio > cfg["eye_closed_ratio"]:
            self._score = min(100.0, self._score + cfg["score_increase"])
        else:
            self._score = max(0.0, self._score - cfg["score_decrease"])

        return self._score, is_blink

    def classify(self, score: float) -> DrowsinessLevel:
        cfg = self._cfg
        if score >= cfg["sleep_threshold"]:
            return DrowsinessLevel.DROWSY
        if score >= cfg["drowsy_threshold"]:
            return DrowsinessLevel.WARNING
        return DrowsinessLevel.AWAKE


# ---------------------------------------------------------------------------
# Frame annotator
# ---------------------------------------------------------------------------

class FrameAnnotator:
    """Static helpers for drawing overlays on BGR frames."""

    @staticmethod
    def draw_face_box(
        frame: np.ndarray,
        faces,
        color: tuple = (255, 0, 0),
        thickness: int = 2,
    ) -> None:
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness)

    @staticmethod
    def draw_eye_boxes(
        frame: np.ndarray,
        eyes,
        x_off: int,
        y_off: int,
        color: tuple = (0, 255, 0),
        thickness: int = 2,
    ) -> None:
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(
                frame,
                (x_off + ex, y_off + ey),
                (x_off + ex + ew, y_off + ey + eh),
                color,
                thickness,
            )

    @staticmethod
    def draw_alert_border(
        frame: np.ndarray,
        color: tuple = (0, 0, 255),
        thickness: int = 15,
    ) -> None:
        h, w = frame.shape[:2]
        cv2.rectangle(frame, (0, 0), (w, h), color, thickness)

    @staticmethod
    def draw_hud(frame: np.ndarray, lines: list[str]) -> None:
        """
        Draw a multi-line HUD in the top-left corner.

        Parameters
        ----------
        lines : list of strings, each drawn on its own row.
        """
        y = 28
        for text in lines:
            # Shadow for readability
            cv2.putText(
                frame, text, (9, y + 1),
                cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 0, 0), 2,
            )
            cv2.putText(
                frame, text, (10, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 1,
            )
            y += 26

    @staticmethod
    def draw_status_banner(
        frame: np.ndarray,
        text: str,
        bg_color: tuple,
    ) -> None:
        """Draw a filled status banner at the bottom of the frame."""
        h, w = frame.shape[:2]
        banner_h = 40
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, h - banner_h), (w, h), bg_color, -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        cv2.putText(
            frame, text,
            (10, h - 12),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2,
        )


# ---------------------------------------------------------------------------
# Main engine
# ---------------------------------------------------------------------------

class ProVisionEngine:
    """
    Orchestrates detection, classification, and annotation.

    Usage
    -----
    engine = ProVisionEngine("config.json")
    metrics = engine.process_frame(bgr_frame, sensitivity=12)
    """

    def __init__(self, config_path: str = "config.json"):
        with open(config_path, "r") as fh:
            self.config = json.load(fh)

        self.detector   = FaceDetector(self.config)
        self.classifier = DrowsinessClassifier(self.config)
        self.annotator  = FrameAnnotator()

        self.frame_count: int = 0
        self._skip: int = self.config["performance"]["skip_frames"]

        # Cache last detection so skipped frames still return valid data
        self._last_faces = []
        self._last_eyes_count = 0
        self._last_yawn = False

    # ---- public ---

    def process_frame(
        self, frame: np.ndarray, sensitivity: int = 12
    ) -> DetectionMetrics:
        """
        Run the full detection pipeline on one BGR frame.

        Frame skipping applies to the expensive cascade detection;
        classifier update and annotation run every frame.
        """
        self.frame_count += 1
        run_detection = (self.frame_count % self._skip) == 0

        gray = self.detector.preprocess(frame)

        if run_detection:
            faces = self.detector.detect_faces(gray)
            self._last_faces = faces
            eyes_count = 0
            yawn = False

            for (x, y, w, h) in faces:
                # Upper ~30 % of face = eye region
                ey_top = y + int(h * 0.22)
                ey_bot = y + int(h * 0.52)
                roi_eye = gray[ey_top:ey_bot, x: x + w]
                eyes = self.detector.detect_eyes(roi_eye, sensitivity)
                eyes_count += len(eyes)

                self.annotator.draw_face_box(frame, [(x, y, w, h)])
                self.annotator.draw_eye_boxes(frame, eyes, x, ey_top)

                # Yawn: open mouth detected in lower face ROI
                roi_mouth = gray[y + int(h * 0.60): y + int(h * 0.95), x: x + w]
                yawn = yawn or len(self.detector.detect_yawn(roi_mouth)) > 0

            self._last_eyes_count = eyes_count
            self._last_yawn = yawn
        else:
            faces = self._last_faces
            eyes_count = self._last_eyes_count
            yawn = self._last_yawn

        score, is_blink = self.classifier.update(eyes_count, len(faces))
        status = self.classifier.classify(score)

        return DetectionMetrics(
            status=status,
            drowsy_score=score,
            eyes_detected=eyes_count,
            faces_detected=len(faces),
            is_blink=is_blink,
            yawn_detected=yawn,
            frame_annotated=frame,
        )
