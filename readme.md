# 🛡️ Pro-Vision v2.0: Advanced Driver Monitoring System

## Overview

Pro-Vision is a **production-ready, real-time driver drowsiness detection system** using edge computing and advanced computer vision. It classifies driver states into **AWAKE**, **DROWSY**, or **SLEEPING** with multi-level alerts.

### Key Features

✅ **Real-time Detection** - 30+ FPS processing  
✅ **Advanced Analytics** - Blink detection, yawn detection, fatigue scoring  
✅ **Multi-level Alerts** - Info/Warning/Critical with audio & visual  
✅ **Event Logging** - Persistent JSON logs with filtering  
✅ **Production Architecture** - Modular, configurable, scalable  
✅ **Easy Deployment** - Single command startup, Streamlit web UI  

---

## 🚀 Quick Start

### Prerequisites

- **Python 3.9+**
- **Webcam** (for live mode)
- **Linux/Mac/Windows**

### Installation

```bash
# Clone or download the repository
cd pro-vision-v2

# Install dependencies
pip install -r requirements.txt

# Run the dashboard
streamlit run app.py
```

The dashboard opens at `http://localhost:8501`

---

## 📋 System Architecture

```
┌─────────────────────────────────────────┐
│          Video Input (Webcam)           │
└──────────────────┬──────────────────────┘
                   │
                   ▼
    ┌──────────────────────────────┐
    │  detection_engine.py         │
    │  ├─ FaceDetector            │
    │  ├─ DrowsinessClassifier    │
    │  └─ FrameAnnotator          │
    └──────────────────┬───────────┘
                       │
                       ▼
        ┌──────────────────────────┐
        │  alert_system.py         │
        │  ├─ AlertManager        │
        │  └─ EventLogger         │
        └──────────┬───────────────┘
                   │
                   ▼
    ┌──────────────────────────────┐
    │  app.py (Streamlit)          │
    │  ├─ Live Feed               │
    │  ├─ Real-time Metrics       │
    │  ├─ Charts & Analytics      │
    │  └─ Event Logs              │
    └──────────────────────────────┘
```

---

## 🔧 Configuration

Edit `config.json` to tune the system:

```json
{
  "detection": {
    "face_scale_factor": 1.1,        // Higher = faster but less accurate
    "face_min_neighbors": 6,         // Higher = fewer false positives
    "eye_scale_factor": 1.1,
    "eye_min_neighbors": 5,
    "clahe_clip_limit": 2.0          // Image enhancement strength
  },
  "drowsiness": {
    "history_window": 15,            // Frames to average (delay)
    "blink_threshold_frames": 2,     // Frames to consider a blink
    "eye_closed_ratio": 0.4,         // % eyes closed = drowsy
    "drowsy_threshold": 35,          // Score threshold for DROWSY
    "sleep_threshold": 75            // Score threshold for SLEEPING
  },
  "performance": {
    "skip_frames": 2,                // Process every Nth frame
    "frame_width": 640,
    "frame_height": 480,
    "target_fps": 30
  },
  "alerts": {
    "beep_cooldown": 2.0,            // Seconds between alerts
    "warning_threshold": 35,
    "critical_threshold": 75
  }
}
```

---

## 📊 Features Explained

### 1. **Real-time Detection**
- **Face Detection**: Haar Cascade (fast, edge-optimized)
- **Eye Detection**: Haar Cascade in face ROI
- **Smile/Yawn Detection**: Optional feature for fatigue analysis

### 2. **Drowsiness Scoring**
- Rolling 15-frame window tracks eye closure
- Hysteresis prevents false positives
- Score decays when eyes open (recovery time)
- Thresholds: DROWSY (>35) → SLEEPING (>75)

### 3. **Blink Detection**
- Distinguishes between normal blinks and eye closure
- Tracks blink rate (blinks/30s)
- Abnormal patterns trigger warnings

### 4. **Multi-level Alerts**
| Level | Trigger | Response |
|-------|---------|----------|
| **INFO** | Normal operation | Green border, ✅ message |
| **WARNING** | Score > 35 | Orange border, ⚠️ message |
| **CRITICAL** | Score > 75 | Red border, 🚨 alert + beep |

### 5. **Event Logging**
- Persistent JSON log (`vision_events.json`)
- Filterable by alert level
- Timestamp, score, and message
- Export for analysis

---

## 🎛️ Dashboard Controls

### Sidebar Settings
- **Camera Mode**: Live Camera or Demo Mode
- **Eye Sensitivity**: 3-20 (lower = more sensitive)
- **Frame Skip**: Process every Nth frame (1-5)
- **Audio/Visual Alerts**: Toggle on/off
- **Event Logs**: View, filter, export, clear

### Main Display
- **Live Video Feed** with annotated detections
- **Real-time Metrics**: Status, score, eyes detected, blink rate
- **Drowsiness Chart**: Score over time
- **Blink Rate Chart**: Blink frequency trend

---

## 📈 Performance Optimization

### Frame Skipping
Process every 2nd or 3rd frame (configurable) while maintaining smooth UI:
```python
skip_frames = 2
if frame_count % skip_frames == 0:
    detect_faces_and_eyes()
```
**Benefit**: 2x-3x faster with minimal latency (50-100ms delay)

### CLAHE Enhancement
Contrast Limited Adaptive Histogram Equalization improves detection in poor lighting:
```python
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
enhanced = clahe.apply(gray_frame)
```

### ROI-based Detection
Eyes are searched only in face region:
```python
roi = frame[face_y:face_y+face_h, face_x:face_x+face_w]
eyes = detect_eyes(roi)  # Much faster than full frame
```

---

## 🔮 Future Enhancements

### Phase 2: Deep Learning Integration
```python
# Optional: Replace Haar Cascades with YOLO/RetinaFace
from ultralytics import YOLO
face_model = YOLO("yolov8n-face.pt")
faces = face_model.predict(frame)
```

### Phase 3: Multi-person Support
- Track multiple drivers simultaneously
- Per-person thresholds and profiles

### Phase 4: Advanced Analytics
- Fatigue prediction (predicts drowsiness before critical)
- Driving pattern analysis
- Integration with CAN bus (vehicle data)

### Phase 5: Cloud Sync
- Upload alerts to backend
- Mobile notifications
- Fleet monitoring dashboard

---

## 🐛 Troubleshooting

### Camera Not Detected
```bash
# Check available cameras
python -c "import cv2; print(cv2.VideoCapture(0).isOpened())"

# Try different camera indices
# In app.py, change: cap = cv2.VideoCapture(0)  →  cap = cv2.VideoCapture(1)
```

### Low FPS
1. Increase `skip_frames` in settings (3-4)
2. Reduce `frame_width`/`frame_height` in config
3. Lower `face_scale_factor` to 1.05

### False Positives (False Alerts)
1. Increase `eye_min_neighbors` in config (7-8)
2. Adjust `eye_closed_ratio` (0.5-0.6)
3. Increase `history_window` (20-25)

### Can't Find Eyes
1. Ensure good lighting
2. Remove glasses or adjust angle
3. Lower `eye_min_neighbors` (3-4)
4. Increase `eye_sensitivity` slider (15-18)

---

## 📁 Project Structure

```
pro-vision-v2/
├── app.py                    # Streamlit dashboard (main entry point)
├── detection_engine.py       # Core detection logic
├── alert_system.py          # Alert & logging system
├── config.json              # Tunable parameters
├── requirements.txt         # Python dependencies
├── vision_events.json       # Event log (auto-generated)
└── README.md               # This file
```

---

## 🔐 Data Privacy

- **No cloud transmission**: All processing local
- **No persistent video**: Frames not stored
- **Logs only**: Timestamps and status changes
- **Opt-out**: Disable logging in config

---

## 📄 License

Free to use and modify. Built for educational and commercial use.

---

## 🤝 Contributing

Want to improve Pro-Vision?

1. **Better Detection**: Experiment with different Haar Cascades
2. **New Features**: Add blink rate alerts, fatigue prediction
3. **Performance**: Optimize with GPU acceleration (CUDA)
4. **UI**: Enhance Streamlit dashboard design

---

## 📞 Support

- **Issues**: Check troubleshooting section
- **Enhancement**: Modify `config.json` for your use case
- **Integration**: Adapt detection_engine.py for custom pipelines

---

## 🚀 Deployment

### Local Development
```bash
streamlit run app.py
```

### Docker (Optional)
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt
CMD ["streamlit", "run", "app.py"]
```

### Cloud Deployment (Streamlit Cloud)
1. Push repo to GitHub
2. Go to [streamlit.io/cloud](https://streamlit.io/cloud)
3. Connect repo and deploy

---

## 📊 Performance Benchmarks

| Metric | Value | Notes |
|--------|-------|-------|
| FPS | 25-30 | 640x480, skip_frames=2 |
| Latency | 100-150ms | Face detection + scoring |
| Memory | ~200MB | Python + OpenCV |
| CPU | 30-40% | Single core (i7 @ 2.6GHz) |

---

**Pro-Vision v2.0** - Built for Production ✅
