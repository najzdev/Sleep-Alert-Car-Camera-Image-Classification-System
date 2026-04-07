# import cv2
# import numpy as np
# import streamlit as st
# import time
# import winsound
# import threading
# from collections import deque hhhhhhh

# # --- 1. THE PRECISION ENGINE ---
# class EmbeddedSystem:
#     def __init__(self):
#         self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
#         self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        
#         self.history = deque(maxlen=15)
#         self.drowsy_score = 0
#         self.last_beep_time = 0  # To prevent audio lag/overlap
#         self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

#     def run_inference(self, frame, sensitivity):
#         if frame is None: return "OFFLINE", 0
        
#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         gray = self.clahe.apply(gray)
#         faces = self.face_cascade.detectMultiScale(gray, 1.1, 6, minSize=(150, 150))
        
#         eyes_this_frame = 0
#         for (x, y, w, h) in faces:
#             cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
#             roi_gray = gray[y + int(h*0.22):y + int(h*0.52), x:x + w]
#             eyes = self.eye_cascade.detectMultiScale(roi_gray, 1.1, sensitivity)
#             eyes_this_frame = len(eyes)
            
#             for (ex, ey, ew, eh) in eyes:
#                 # Draw green eye trackers
#                 cv2.rectangle(frame[y+int(h*0.22):y+int(h*0.52), x:x+w], (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)

#         # Logic: 1 if eyes closed, 0 if eyes open
#         is_closed = 1 if (len(faces) > 0 and eyes_this_frame == 0) else 0
#         self.history.append(is_closed)
#         sleep_ratio = sum(self.history) / len(self.history) if self.history else 0
        
#         if sleep_ratio > 0.4: self.drowsy_score += 10
#         else: self.drowsy_score -= 15
            
#         self.drowsy_score = max(0, min(100, self.drowsy_score))
        
#         status = "AWAKE"
#         if self.drowsy_score > 75: status = "SLEEPING"
#         elif self.drowsy_score > 30: status = "DROWSY"
            
#         return status, frame

# # --- 2. ASYNC AUDIO ALERT ---
# def play_alarm_async():
#     winsound.Beep(2000, 200)

# # --- 3. STREAMLIT UI ---
# st.set_page_config(page_title="Pro-Vision v6.0", layout="wide")
# st.title("🛡️ Pro-Vision Edge Monitor")

# # Sidebar: Analytics Logs
# st.sidebar.header("📊 System Logs")
# log_container = st.sidebar.container(height=250)
# if 'logs' not in st.session_state: st.session_state.logs = []

# # Sidebar: Hardware Tuning
# st.sidebar.divider()
# sens = st.sidebar.slider("Eye Sensitivity", 3, 20, 12)
# bar_ui = st.sidebar.progress(0)
# chart_ui = st.sidebar.empty()
# history_plot = deque(maxlen=50)

# # Layout: Compact Dash
# col1, col2 = st.columns([1, 1.2])

# with col1:
#     st.subheader("Edge Camera Feed")
#     video_ui = st.empty()

# with col2:
#     st.subheader("Real-time Classification")
#     status_box = st.empty()
#     instruction_ui = st.info("System Calibrating...")

# # --- 4. EXECUTION ---
# if 'system' not in st.session_state:
#     st.session_state.system = EmbeddedSystem()
#     st.session_state.last_status = "AWAKE"

# cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret: continue
    
#     frame = cv2.flip(frame, 1)
#     label, frame = st.session_state.system.run_inference(frame, sens)
    
#     # --- LOGGING & UI COLORS ---
#     if label != st.session_state.last_status:
#         st.session_state.logs.insert(0, f"[{time.strftime('%H:%M:%S')}] -> {label}")
#         st.session_state.last_status = label
#         with log_container:
#             for l in st.session_state.logs[:10]: st.write(l)

#     # Color Logic for Display
#     if label == "SLEEPING":
#         status_box.error(f"### STATUS: {label}")
#         instruction_ui.warning("🚨 EMERGENCY: WAKE UP IMMEDIATELY!")
#         # THREADED SOUND WITH COOLDOWN (Fixes the delay/lag)
#         if time.time() - st.session_state.system.last_beep_time > 0.8:
#             threading.Thread(target=play_alarm_async, daemon=True).start()
#             st.session_state.system.last_beep_time = time.time()
#         # Red Border
#         cv2.rectangle(frame, (0,0), (frame.shape[1], frame.shape[0]), (0,0,255), 20)
#     elif label == "DROWSY":
#         status_box.warning(f"### STATUS: {label}")
#         instruction_ui.info("⚠️ CAUTION: Eyes closing frequently.")
#     else:
#         status_box.success(f"### STATUS: {label}")
#         instruction_ui.success("✅ Secure: Driver is attentive.")

#     # Telemetry Updates
#     score = st.session_state.system.drowsy_score
#     bar_ui.progress(score / 100)
#     history_plot.append(score)
#     chart_ui.line_chart(list(history_plot))

#     # Display (Small Window)
#     rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     video_ui.image(rgb, channels="RGB", width=420)
    
#     time.sleep(0.01)

# cap.release()




import cv2
import numpy as np
import streamlit as st
import time
import base64
import threading
from collections import deque

# --- 1. THE PRECISION ENGINE ---
class EmbeddedSystem:
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        self.history = deque(maxlen=15)
        self.drowsy_score = 0
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

    def run_inference(self, frame, sensitivity):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = self.clahe.apply(gray)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 6, minSize=(150, 150))
        
        eyes_this_frame = 0
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            roi_gray = gray[y + int(h*0.22):y + int(h*0.52), x:x + w]
            eyes = self.eye_cascade.detectMultiScale(roi_gray, 1.1, sensitivity)
            eyes_this_frame = len(eyes)
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(frame[y+int(h*0.22):y+int(h*0.52), x:x+w], (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)

        is_closed = 1 if (len(faces) > 0 and eyes_this_frame == 0) else 0
        self.history.append(is_closed)
        sleep_ratio = sum(self.history) / len(self.history) if self.history else 0
        
        if sleep_ratio > 0.4: self.drowsy_score += 10
        else: self.drowsy_score -= 15
            
        self.drowsy_score = max(0, min(100, self.drowsy_score))
        
        status = "AWAKE"
        if self.drowsy_score > 75: status = "SLEEPING"
        elif self.drowsy_score > 35: status = "DROWSY"
            
        return status, frame

# --- 2. WEB AUDIO PLAYER (Replaces Winsound) ---
def play_web_alarm():
    # This creates an invisible HTML audio tag that plays a beep
    # You can replace this with a link to an actual .mp3 file
    audio_html = """
        <audio autoplay>
            <source src="https://www.soundjay.com/buttons/beep-01a.mp3" type="audio/mpeg">
        </audio>
    """
    st.components.v1.html(audio_html, height=0)

# --- 3. STREAMLIT DASHBOARD ---
st.set_page_config(page_title="Pro-Vision Cloud", layout="wide")
st.title("🛡️ Pro-Vision: Cloud Edge Demo")

st.sidebar.header("📊 System Logs")
log_placeholder = st.sidebar.empty()
if 'logs' not in st.session_state: st.session_state.logs = []

sens = st.sidebar.slider("Eye Sensitivity", 3, 20, 12)
chart_ui = st.sidebar.empty()
history_plot = deque(maxlen=50)

col1, col2 = st.columns([1, 1.2])
with col1: video_ui = st.empty()
with col2:
    status_box = st.empty()
    alert_ui = st.empty()

# --- 4. CLOUD CAMERA INITIALIZATION ---
if 'system' not in st.session_state:
    st.session_state.system = EmbeddedSystem()
    st.session_state.last_status = "AWAKE"
    st.session_state.last_beep = 0

# NOTE: For local testing this works. 
# For true WebRTC (friends' cameras), you'd use 'streamlit_webrtc'
cap = cv2.VideoCapture(0) 

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    
    frame = cv2.flip(frame, 1)
    label, frame = st.session_state.system.run_inference(frame, sens)
    
    # Update Logs
    if label != st.session_state.last_status:
        st.session_state.logs.insert(0, f"[{time.strftime('%H:%M:%S')}] {label}")
        st.session_state.last_status = label
        log_placeholder.write("\n".join(st.session_state.logs[:8]))

    # UI Handling
    if label == "SLEEPING":
        status_box.error(f"### STATUS: {label}")
        alert_ui.warning("🚨 WAKE UP!")
        if time.time() - st.session_state.last_beep > 2:
            play_web_alarm()
            st.session_state.last_beep = time.time()
        cv2.rectangle(frame, (0,0), (frame.shape[1], frame.shape[0]), (0,0,255), 20)
    elif label == "DROWSY":
        status_box.warning(f"### STATUS: {label}")
    else:
        status_box.success(f"### STATUS: {label}")

    history_plot.append(st.session_state.system.drowsy_score)
    chart_ui.line_chart(list(history_plot))

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    video_ui.image(rgb, channels="RGB", width=400)
    time.sleep(0.01)