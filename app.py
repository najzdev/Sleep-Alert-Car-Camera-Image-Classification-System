"""
Pro-Vision v2.0 - Application Controller

Dual execution mode:
  1. Streamlit UI  : streamlit run app.py
  2. Standalone CV : python app.py

The execution mode is detected automatically via the STREAMLIT_SERVER_PORT
environment variable (set by Streamlit at startup). use_container_width=True
"""

import os
import sys
import cv2
import time
import json
import numpy as np
from collections import deque
from datetime import datetime
from pathlib import Path

from detection_engine import ProVisionEngine, DrowsinessLevel, FrameAnnotator
from alert_system import AlertManager, AlertLevel, EventLogger
from analytics import SessionAnalytics

# ---------------------------------------------------------------------------
# Mode detection
# ---------------------------------------------------------------------------

IS_STREAMLIT = "streamlit" in sys.modules or os.environ.get("STREAMLIT_SERVER_PORT")

if IS_STREAMLIT:
    import streamlit as st


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def build_hud_lines(metrics, alert_info, fps: float, analytics) -> list[str]:
    """Build the text lines drawn on the OpenCV frame."""
    snap = analytics.snapshot()
    return [
        f"Status : {metrics.status.value}",
        f"Score  : {metrics.drowsy_score:.0f}/100",
        f"Eyes   : {metrics.eyes_detected}",
        f"Blinks : {snap.blink_rate_per_minute:.1f}/min",
        f"FPS    : {fps:.1f}",
    ]


def annotate_frame(frame, metrics, alert_info, fps, analytics, visual_alerts=True):
    """Apply all overlays to a BGR frame in-place."""
    if visual_alerts and alert_info["level"] != AlertLevel.NORMAL:
        FrameAnnotator.draw_alert_border(
            frame, color=alert_info["border_color"], thickness=18
        )

    hud = build_hud_lines(metrics, alert_info, fps, analytics)
    FrameAnnotator.draw_hud(frame, hud)

    status_text = f"  {alert_info['message']}"
    FrameAnnotator.draw_status_banner(
        frame, status_text, bg_color=alert_info["border_color"]
    )

    if metrics.yawn_detected:
        h = frame.shape[0]
        cv2.putText(
            frame, "YAWN DETECTED",
            (10, h - 55),
            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 200, 255), 2,
        )


# ---------------------------------------------------------------------------
# STANDALONE MODE
# ---------------------------------------------------------------------------

def run_standalone():
    """
    OpenCV window loop.
    Press 'q' or ESC to quit.
    """
    config_path = "config.json"
    if not Path(config_path).exists():
        print(f"[ERROR] {config_path} not found.")
        sys.exit(1)

    with open(config_path) as fh:
        config = json.load(fh)

    engine    = ProVisionEngine(config_path)
    logger    = EventLogger(log_file="vision_events.json")
    alerter   = AlertManager(config, logger)
    analytics = SessionAnalytics()

    sensitivity = config["detection"].get("default_sensitivity", 12)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Cannot open camera.")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    fps = 0.0
    t0 = time.time()
    frame_counter = 0

    print("[Pro-Vision] Running standalone. Press 'q' or ESC to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Failed to read camera frame.")
            break

        frame = cv2.flip(frame, 1)

        metrics    = engine.process_frame(frame, sensitivity)
        alert_info = alerter.evaluate(metrics.drowsy_score)
        analytics.record_frame(
            eyes_open=metrics.eyes_detected > 0,
            is_blink=metrics.is_blink,
            drowsy_score=metrics.drowsy_score,
        )

        # FPS calculation
        frame_counter += 1
        elapsed = time.time() - t0
        if elapsed >= 1.0:
            fps = frame_counter / elapsed
            frame_counter = 0
            t0 = time.time()

        annotate_frame(metrics.frame_annotated, metrics, alert_info, fps, analytics)

        cv2.imshow("Pro-Vision v2.0 | Press Q to quit", metrics.frame_annotated)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), ord("Q"), 27):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[Pro-Vision] Session ended.")


# ---------------------------------------------------------------------------
# STREAMLIT MODE
# ---------------------------------------------------------------------------

def run_streamlit():
    """Full Streamlit dashboard."""

    st.set_page_config(
        page_title="Pro-Vision v2.0",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.markdown("""
        <style>
            .metric-card {
                background-color: #f0f2f6;
                padding: 12px 15px;
                border-radius: 8px;
                margin: 4px 0;
            }
            .level-normal  { border-left: 5px solid #28a745; background: #d4edda; }
            .level-warning { border-left: 5px solid #ffc107; background: #fff3cd; }
            .level-drowsy  { border-left: 5px solid #dc3545; background: #f8d7da; }
        </style>
    """, unsafe_allow_html=True)

    # --- Session state init ---
    if "engine" not in st.session_state:
        config_path = "config.json"
        with open(config_path) as fh:
            cfg = json.load(fh)
        st.session_state.config       = cfg
        st.session_state.engine       = ProVisionEngine(config_path)
        st.session_state.logger       = EventLogger(log_file="vision_events.json")
        st.session_state.alerter      = AlertManager(cfg, st.session_state.logger)
        st.session_state.analytics    = SessionAnalytics()
        st.session_state.session_start = datetime.now()

    engine    = st.session_state.engine
    alerter   = st.session_state.alerter
    analytics = st.session_state.analytics
    logger    = st.session_state.logger

    # ---- SIDEBAR ----
    with st.sidebar:
        st.title("Settings & Controls")

        st.subheader("Camera")
        sensitivity = st.slider(
            "Eye Sensitivity", 3, 20, 12,
            help="Lower = more sensitive to eye closure"
        )
        frame_skip = st.slider(
            "Frame Skip", 1, 5, 2,
            help="Process every Nth frame"
        )
        engine._skip = frame_skip

        st.divider()
        st.subheader("Alerts")
        visual_alerts = st.checkbox("Visual Border Alerts", value=True)

        st.divider()
        st.subheader("System Stats")
        c1, c2 = st.columns(2)
        c1.metric(
            "Session",
            str(datetime.now() - st.session_state.session_start).split(".")[0],
        )
        c2.metric("Frames", engine.frame_count)

        st.divider()
        st.subheader("Recent Events")
        if st.button("Refresh Logs"):
            st.rerun()
        recent = logger.get_recent(8)
        if recent:
            for ev in reversed(recent):
                ts = ev.timestamp.split("T")[1][:8]
                css = f"level-{ev.alert_level.lower()}"
                st.markdown(f"""
                    <div class="metric-card {css}">
                        <b>[{ts}] {ev.alert_level}</b><br>
                        {ev.message}
                    </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No events yet.")

        st.divider()
        c1, c2 = st.columns(2)
        with c1:
            if st.button("Export Logs"):
                data = json.dumps(
                    [e.to_dict() for e in logger.events], indent=2
                )
                st.download_button("Download JSON", data, "vision_logs.json")
        with c2:
            if st.button("Clear Logs"):
                logger.clear()
                st.success("Cleared.")

    # ---- MAIN AREA ----
    st.title("Pro-Vision v2.0: Driver Monitoring")
    st.markdown("Real-time drowsiness detection")

    col_video, col_metrics = st.columns([2, 1.2])

    with col_video:
        st.subheader("Live Feed")
        video_ph = st.empty()
        fps_ph   = st.empty()

    with col_metrics:
        st.subheader("Real-time Metrics")
        status_ph  = st.empty()
        score_ph   = st.empty()
        eyes_ph    = st.empty()
        blink_ph   = st.empty()
        closure_ph = st.empty()
        alert_ph   = st.empty()

    col_c1, col_c2 = st.columns(2)
    with col_c1:
        st.subheader("Drowsiness Score")
        score_chart_ph = st.empty()
    with col_c2:
        st.subheader("Blink Rate (per min)")
        blink_chart_ph = st.empty()

    # ---- CAMERA LOOP ----
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("Cannot access webcam. Ensure the camera is connected.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    stop_ph = st.empty()
    stop = stop_ph.button("Stop Monitor")

    fps = 0.0
    t0  = time.time()
    fc  = 0

    while cap.isOpened() and not stop:
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to read camera frame.")
            break

        frame = cv2.flip(frame, 1)

        metrics    = engine.process_frame(frame, sensitivity)
        alert_info = alerter.evaluate(metrics.drowsy_score)
        analytics.record_frame(
            eyes_open=metrics.eyes_detected > 0,
            is_blink=metrics.is_blink,
            drowsy_score=metrics.drowsy_score,
        )

        fc += 1
        elapsed = time.time() - t0
        if elapsed >= 1.0:
            fps = fc / elapsed
            fc = 0
            t0 = time.time()

        annotate_frame(
            metrics.frame_annotated, metrics, alert_info, fps, analytics, visual_alerts
        )

        rgb = cv2.cvtColor(metrics.frame_annotated, cv2.COLOR_BGR2RGB)
        video_ph.image(rgb, channels="RGB", width='stretch')
        fps_ph.metric("FPS", f"{fps:.1f}")

        snap = analytics.snapshot()

        # Status indicator
        color_label = {
            DrowsinessLevel.AWAKE:   "AWAKE",
            DrowsinessLevel.WARNING: "WARNING",
            DrowsinessLevel.DROWSY:  "DROWSY",
            DrowsinessLevel.OFFLINE: "OFFLINE",
        }
        status_ph.metric("Status", color_label.get(metrics.status, "---"))
        score_ph.metric("Drowsiness Score", f"{metrics.drowsy_score:.0f} / 100")
        eyes_ph.metric("Eyes Detected", metrics.eyes_detected)
        blink_ph.metric("Blink Rate", f"{snap.blink_rate_per_minute:.1f} /min")
        closure_ph.metric("Avg Closure", f"{snap.avg_eye_closure_ms:.0f} ms")

        css = f"level-{alert_info['level'].value.lower()}"
        alert_ph.markdown(f"""
            <div class="metric-card {css}">
                <b>{alert_info['message']}</b>
            </div>
        """, unsafe_allow_html=True)

        if len(analytics.score_history) > 1:
            score_chart_ph.line_chart({"Score": list(analytics.score_history)})
            blink_chart_ph.line_chart({"Blinks/min": list(analytics.blink_rate_history)})

        time.sleep(0.01)

    cap.release()

    # ---- FOOTER ----
    st.divider()
    st.markdown("""
        <div style='text-align:center; color:#666; font-size:0.85em; margin-top:20px;'>
            <p style='font-weight:bold; color:#444;'>
                LABBAALLI Hamza | ID-BOUBRIK Abdelouahed | MAAROUF Yassine | IDDAHA Soumaya
            </p>
            <p>Pro-Vision v2.0 | Advanced Driver Monitoring System</p>
            <p>Built with OpenCV, Streamlit, and Python</p>
        </div>
    """, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if IS_STREAMLIT:
    run_streamlit()
else:
    if __name__ == "__main__":
        run_standalone()
