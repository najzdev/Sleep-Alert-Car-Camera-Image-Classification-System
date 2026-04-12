"""
Pro-Vision v2.0 - Production Dashboard
Real-time driver monitoring with advanced analytics
"""

import cv2
import numpy as np
import streamlit as st
import time
import json
from pathlib import Path
from collections import deque
from datetime import datetime, timedelta

from detection_engine import ProVisionEngine, DrowsinessLevel
from alert_system import AlertManager, AlertLevel, EventLogger


# ============================================================================
# PAGE CONFIG
# ============================================================================

st.set_page_config(
    page_title="Pro-Vision v2.0",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
        .metric-card {
            background-color: #f0f2f6;
            padding: 15px;
            border-radius: 10px;
            margin: 5px 0;
        }
        .alert-critical {
            background-color: #ffcccc;
            border-left: 5px solid #ff0000;
        }
        .alert-warning {
            background-color: #fff3cd;
            border-left: 5px solid #ffc107;
        }
        .alert-info {
            background-color: #d1ecf1;
            border-left: 5px solid #17a2b8;
        }
    </style>
""", unsafe_allow_html=True)


# ============================================================================
# INITIALIZATION
# ============================================================================

@st.cache_resource
def initialize_system():
    """Load detection engine and alert system once"""
    engine = ProVisionEngine("config.json")
    logger = EventLogger(log_file="vision_events.json")
    alert_manager = AlertManager(engine.config, logger)
    return engine, alert_manager, logger


def init_session_state():
    """Initialize session state variables"""
    if 'engine' not in st.session_state:
        st.session_state.engine, st.session_state.alert_mgr, st.session_state.logger = initialize_system()
    
    if 'last_status' not in st.session_state:
        st.session_state.last_status = None
    
    if 'session_start' not in st.session_state:
        st.session_state.session_start = datetime.now()
    
    if 'history' not in st.session_state:
        st.session_state.history = {
            'scores': deque(maxlen=100),
            'timestamps': deque(maxlen=100),
            'blink_rates': deque(maxlen=100),
            'yawns': 0
        }


init_session_state()


# ============================================================================
# SIDEBAR: CONTROLS & SETTINGS
# ============================================================================

with st.sidebar:
    st.title("⚙️ Settings & Controls")
    
    # Mode selection
    mode = st.radio("Select Mode", ["Live Camera", "Demo Mode"])
    
    # Camera controls
    if mode == "Live Camera":
        st.subheader("📹 Camera Settings")
        sensitivity = st.slider("Eye Sensitivity", 3, 20, 12, 
                               help="Lower = more sensitive to eye closure")
        frame_skip = st.slider("Frame Skip", 1, 5, 2,
                              help="Process every Nth frame (faster but less responsive)")
    else:
        sensitivity = 12
        frame_skip = 2
    
    # Alert settings
    st.divider()
    st.subheader("🔔 Alert Settings")
    col1, col2 = st.columns(2)
    with col1:
        beep_enabled = st.checkbox("Enable Audio Alerts", value=True)
    with col2:
        visual_alerts = st.checkbox("Visual Alerts", value=True)
    
    # System stats
    st.divider()
    st.subheader("📊 System Stats")
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Session Duration", 
                 str(datetime.now() - st.session_state.session_start).split('.')[0])
    with col2:
        st.metric("Frames Processed", st.session_state.engine.frame_count)
    
    # Event logs
    st.divider()
    st.subheader("📋 Recent Events")
    
    if st.button("🔄 Refresh Logs"):
        st.rerun()
    
    log_filter = st.selectbox("Filter by Level",
                             ["All", "INFO", "WARNING", "CRITICAL"])
    
    recent_events = st.session_state.logger.get_recent(8)
    
    if recent_events:
        for event in reversed(recent_events):
            level = event.alert_level
            css_class = f"alert-{level.lower()}"
            timestamp = event.timestamp.split('T')[1][:8]
            
            st.markdown(f"""
                <div class="metric-card alert-{level.lower()}">
                    <b>[{timestamp}] {level}</b><br>
                    {event.message}
                </div>
            """, unsafe_allow_html=True)
    else:
        st.info("No events logged yet")
    
    # Export & Reset
    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        if st.button("📥 Export Logs"):
            logs_data = json.dumps(
                [e.to_dict() for e in st.session_state.logger.events],
                indent=2
            )
            st.download_button("Download JSON", logs_data, "vision_logs.json")
    with col2:
        if st.button("🗑️ Clear Logs"):
            st.session_state.logger.clear()
            st.success("Logs cleared!")


# ============================================================================
# MAIN DASHBOARD
# ============================================================================

st.title("🛡️ Pro-Vision v2.0: Edge Monitor")
st.markdown("Real-time driver drowsiness detection with advanced analytics")

# Layout: Main video + side metrics
col_video, col_metrics = st.columns([2, 1.2])

with col_video:
    st.subheader("📹 Live Feed")
    video_placeholder = st.empty()
    fps_placeholder = st.empty()

with col_metrics:
    st.subheader("📊 Real-time Metrics")
    status_placeholder = st.empty()
    score_placeholder = st.empty()
    eyes_placeholder = st.empty()
    blink_placeholder = st.empty()
    alert_placeholder = st.empty()


# Chart area (below video)
col_chart1, col_chart2 = st.columns(2)

with col_chart1:
    st.subheader("📈 Drowsiness Score Over Time")
    chart_placeholder = st.empty()

with col_chart2:
    st.subheader("👁️ Blink Rate Over Time")
    blink_chart_placeholder = st.empty()


# ============================================================================
# MAIN PROCESSING LOOP
# ============================================================================

if mode == "Live Camera":
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        st.error("❌ Cannot access webcam. Please ensure camera is connected.")
    else:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        placeholder_stop = st.empty()
        stop_button = placeholder_stop.button("⏹️ Stop Monitor")
        
        frame_start_time = time.time()
        frame_count = 0
        fps = 0
        
        while cap.isOpened() and not stop_button:
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to read frame from camera")
                break
            
            # Flip for selfie view
            frame = cv2.flip(frame, 1)
            
            # Process frame
            metrics = st.session_state.engine.process_frame(frame, sensitivity)
            
            # Update history
            st.session_state.history['scores'].append(metrics.drowsy_score)
            st.session_state.history['timestamps'].append(time.time())
            st.session_state.history['blink_rates'].append(metrics.blink_count)
            if metrics.yawn_detected:
                st.session_state.history['yawns'] += 1
            
            # Evaluate alerts
            alert_info = st.session_state.alert_mgr.evaluate(
                metrics.drowsy_score,
                metrics.status
            )
            
            # Draw alert border
            if visual_alerts and alert_info["should_alert"]:
                from detection_engine import FrameAnnotator
                FrameAnnotator.draw_alert_border(
                    metrics.frame_annotated,
                    color=alert_info["border_color"],
                    thickness=20
                )
            
            # Draw metrics on frame
            metrics_text = {
                f"Score: {metrics.drowsy_score:.0f}": "",
                f"Eyes: {metrics.eyes_detected}": "",
                f"Blinks/30s: {metrics.blink_count}": "",
                f"Status: {metrics.status.value}": ""
            }
            from detection_engine import FrameAnnotator
            FrameAnnotator.draw_metrics_text(metrics.frame_annotated, metrics_text)
            
            # Convert BGR to RGB for display
            rgb_frame = cv2.cvtColor(metrics.frame_annotated, cv2.COLOR_BGR2RGB)
            
            # Calculate FPS
            frame_count += 1
            elapsed = time.time() - frame_start_time
            if elapsed > 1.0:
                fps = frame_count / elapsed
                frame_count = 0
                frame_start_time = time.time()
            
            # Update UI - Video Feed
            # NEW
video_placeholder.image(rgb_frame, channels="RGB", use_container_width=True)
            fps_placeholder.metric("FPS", f"{fps:.1f}", delta=None)
            
            # Update UI - Metrics
            status_color = "🟢" if metrics.status == DrowsinessLevel.AWAKE else \
                          "🟡" if metrics.status == DrowsinessLevel.DROWSY else "🔴"
            status_placeholder.metric(
                "Status",
                f"{status_color} {metrics.status.value}",
                delta=None
            )
            
            score_placeholder.metric(
                "Drowsiness Score",
                f"{metrics.drowsy_score:.0f}/100",
                delta=None
            )
            
            eyes_placeholder.metric(
                "Eyes Detected",
                metrics.eyes_detected,
                delta=None
            )
            
            blink_placeholder.metric(
                "Blinks/30s",
                metrics.blink_count,
                delta=None
            )
            
            # Update UI - Alert
            alert_css = f"alert-{alert_info['alert_level'].value.lower()}"
            alert_placeholder.markdown(f"""
                <div class="metric-card {alert_css}">
                    <b>{alert_info['alert_message']}</b>
                </div>
            """, unsafe_allow_html=True)
            
            # Update Charts
            if len(st.session_state.history['scores']) > 1:
                chart_data = {
                    "Score": list(st.session_state.history['scores'])
                }
                chart_placeholder.line_chart(chart_data)
                
                blink_data = {
                    "Blinks/30s": list(st.session_state.history['blink_rates'])
                }
                blink_chart_placeholder.line_chart(blink_data)
            
            # Control refresh rate
            time.sleep(0.01)
        
        cap.release()

else:  # Demo Mode
    st.info("📌 Demo Mode: This shows sample data without camera access")
    
    st.subheader("📹 Live Feed")
    st.markdown("*(Simulated demo feed - no camera required)*")
    
    # Create fake image
    demo_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(demo_frame, "DEMO MODE", (150, 240),
               cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
    cv2.putText(demo_frame, "No camera connected", (120, 300),
               cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 200, 200), 2)
    
    video_placeholder.image(cv2.cvtColor(demo_frame, cv2.COLOR_BGR2RGB),
                           channels="RGB", use_column_width=True)
    
    st.markdown("""
    **Demo Features:**
    - Shows sample detection output
    - Demonstrates UI layout
    - For testing without camera
    
    To use **Live Camera Mode**, select it from the sidebar.
    """)


# ============================================================================
# FOOTER
# ============================================================================

st.divider()
st.markdown("""
    <div style='text-align: center; color: #666; font-size: 0.85em; margin-top: 20px;'>
        <p>Pro-Vision v2.0 | Advanced Driver Monitoring System</p>
        <p>Built with OpenCV, Streamlit, and Python</p>
    </div>
""", unsafe_allow_html=True)
