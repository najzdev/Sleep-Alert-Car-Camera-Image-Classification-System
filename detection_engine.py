"""
Core Detection Engine - Pro-Vision v2.0
Handles face, eye, and drowsiness detection with advanced analytics
"""

import cv2
import numpy as np
from collections import deque
from dataclasses import dataclass
from enum import Enum
import json


class DrowsinessLevel(Enum):
    """Classification states"""
    AWAKE = "AWAKE"
    DROWSY = "DROWSY"
    SLEEPING = "SLEEPING"
    OFFLINE = "OFFLINE"


@dataclass
class DetectionMetrics:
    """Encapsulates detection results"""
    status: DrowsinessLevel
    drowsy_score: float
    eyes_detected: int
    faces_detected: int
    blink_count: int
    yawn_detected: bool
    frame_annotated: np.ndarray = None


class FaceDetector:
    """Specialized face and eye detection"""
    
    def __init__(self, config: dict):
        self.config = config["detection"]
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        self.eye_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_eye.xml'
        )
        self.smile_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_smile.xml'
        )
        self.clahe = cv2.createCLAHE(
            clipLimit=self.config["clahe_clip_limit"],
            tileGridSize=(self.config["clahe_tile_size"], 
                         self.config["clahe_tile_size"])
        )
    
    def preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """Enhance grayscale frame with CLAHE"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return self.clahe.apply(gray)
    
    def detect_faces(self, gray: np.ndarray) -> list:
        """Detect faces in frame"""
        return self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=self.config["face_scale_factor"],
            minNeighbors=self.config["face_min_neighbors"],
            minSize=(self.config["face_min_size"], self.config["face_min_size"])
        )
    
    def detect_eyes_in_roi(self, roi_gray: np.ndarray, sensitivity: int) -> list:
        """Detect eyes in face ROI"""
        return self.eye_cascade.detectMultiScale(
            roi_gray,
            scaleFactor=self.config["eye_scale_factor"],
            minNeighbors=sensitivity
        )
    
    def detect_smile(self, roi_gray: np.ndarray) -> list:
        """Detect smile/mouth in face ROI"""
        return self.smile_cascade.detectMultiScale(
            roi_gray,
            scaleFactor=1.8,
            minNeighbors=20,
            minSize=(25, 25)
        )


class DrowsinessClassifier:
    """Advanced drowsiness scoring with blink detection"""
    
    def __init__(self, config: dict):
        self.config = config
        self.history = deque(maxlen=config["drowsiness"]["history_window"])
        self.drowsy_score = 0
        self.blink_history = deque(maxlen=10)
        self.consecutive_closed = 0
    
    def update(self, eyes_detected: int, faces_detected: int) -> tuple:
        """
        Update drowsiness score based on eye detection
        Returns: (drowsy_score, is_blinking, blink_rate)
        """
        cfg = self.config["drowsiness"]
        
        # Detect blink: eyes suddenly close then open
        is_closed = 1 if (faces_detected > 0 and eyes_detected == 0) else 0
        is_blinking = False
        
        if is_closed:
            self.consecutive_closed += 1
        else:
            # Was closed, now open = blink detected
            if self.consecutive_closed > 0 and self.consecutive_closed <= cfg["blink_threshold_frames"]:
                is_blinking = True
                self.blink_history.append(1)
            self.consecutive_closed = 0
            self.blink_history.append(0)
        
        self.history.append(is_closed)
        
        # Calculate sleep ratio from history
        sleep_ratio = sum(self.history) / len(self.history) if self.history else 0
        
        # Update drowsy score with hysteresis
        if sleep_ratio > cfg["eye_closed_ratio"]:
            self.drowsy_score += cfg["score_increase"]
        else:
            self.drowsy_score -= cfg["score_decrease"]
        
        self.drowsy_score = max(0, min(100, self.drowsy_score))
        
        # Calculate blink rate (blinks per second, assuming 30fps)
        blink_rate = sum(self.blink_history) / len(self.blink_history) * 30 if self.blink_history else 0
        
        return self.drowsy_score, is_blinking, blink_rate
    
    def classify(self, score: float) -> DrowsinessLevel:
        """Classify drowsiness level from score"""
        cfg = self.config["drowsiness"]
        
        if score > cfg["sleep_threshold"]:
            return DrowsinessLevel.SLEEPING
        elif score > cfg["drowsy_threshold"]:
            return DrowsinessLevel.DROWSY
        else:
            return DrowsinessLevel.AWAKE


class FrameAnnotator:
    """Handles frame annotations and visualizations"""
    
    @staticmethod
    def draw_face_box(frame: np.ndarray, faces: list, color=(255, 0, 0), thickness=2):
        """Draw bounding boxes around detected faces"""
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, thickness)
    
    @staticmethod
    def draw_eye_boxes(frame: np.ndarray, eyes: list, x_offset: int, 
                       y_offset: int, color=(0, 255, 0), thickness=2):
        """Draw bounding boxes around detected eyes in ROI"""
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(frame, 
                         (x_offset + ex, y_offset + ey),
                         (x_offset + ex + ew, y_offset + ey + eh),
                         color, thickness)
    
    @staticmethod
    def draw_alert_border(frame: np.ndarray, color=(0, 0, 255), thickness=15):
        """Draw colored border for alerts"""
        h, w = frame.shape[:2]
        cv2.rectangle(frame, (0, 0), (w, h), color, thickness)
    
    @staticmethod
    def draw_metrics_text(frame: np.ndarray, metrics_dict: dict):
        """Draw metrics text on frame"""
        y_pos = 30
        for key, value in metrics_dict.items():
            text = f"{key}: {value}"
            cv2.putText(frame, text, (10, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            y_pos += 30


class ProVisionEngine:
    """Main detection engine orchestrating all components"""
    
    def __init__(self, config_path: str = "config.json"):
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        self.detector = FaceDetector(self.config)
        self.classifier = DrowsinessClassifier(self.config)
        self.annotator = FrameAnnotator()
        self.frame_count = 0
        self.skip_frames = self.config["performance"]["skip_frames"]
    
    def process_frame(self, frame: np.ndarray, sensitivity: int = 12) -> DetectionMetrics:
        """
        Main processing pipeline
        Returns: DetectionMetrics object
        """
        self.frame_count += 1
        
        # Skip frames for performance optimization
        should_process_detection = (self.frame_count % self.skip_frames) == 0
        
        # Always preprocess
        gray = self.detector.preprocess_frame(frame)
        
        if should_process_detection:
            # Detect faces
            faces = self.detector.detect_faces(gray)
            faces_count = len(faces)
            
            # Detect eyes and other features
            eyes_count = 0
            yawn_detected = False
            
            for (x, y, w, h) in faces:
                # Eye ROI: upper portion of face
                roi_gray = gray[y + int(h*0.22):y + int(h*0.52), x:x + w]
                eyes = self.detector.detect_eyes_in_roi(roi_gray, sensitivity)
                eyes_count += len(eyes)
                
                # Annotate detected features
                self.annotator.draw_face_box(frame, [(x, y, w, h)])
                self.annotator.draw_eye_boxes(frame, eyes, x, y + int(h*0.22))
                
                # Yawn detection: smile in lower portion
                roi_mouth = gray[y + int(h*0.60):y + int(h*0.95), x:x + w]
                smiles = self.detector.detect_smile(roi_mouth)
                yawn_detected = len(smiles) > 0
        else:
            faces_count = 0
            eyes_count = 0
            yawn_detected = False
        
        # Update classifier
        drowsy_score, is_blinking, blink_rate = self.classifier.update(eyes_count, faces_count)
        status = self.classifier.classify(drowsy_score)
        
        return DetectionMetrics(
            status=status,
            drowsy_score=drowsy_score,
            eyes_detected=eyes_count,
            faces_detected=faces_count,
            blink_count=int(blink_rate),
            yawn_detected=yawn_detected,
            frame_annotated=frame
        )
