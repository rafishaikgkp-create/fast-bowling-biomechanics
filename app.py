import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import tempfile
import os
import sqlite3
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="BCCI Elite Fast Bowling Biomechanics & S&C Engine",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom High-Contrast Styling (Clean Dark Mode + High Contrast Cards)
st.markdown("""
<style>
    .metric-card {
        background-color: #1E222A;
        border-radius: 8px;
        padding: 16px;
        border-left: 5px solid #00D26A;
        margin-bottom: 15px;
        color: #F8F9FA !important;
    }
    .metric-card h4 {
        color: #00D26A !important;
        margin-top: 0px;
        margin-bottom: 8px;
        font-size: 1.1rem;
    }
    .metric-card p {
        color: #E2E8F0 !important;
        margin-bottom: 0px;
        font-size: 0.95rem;
        line-height: 1.5;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 48px;
        white-space: pre-wrap;
        border-radius: 4px 4px 0px 0px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# DATABASE SCHEMA & INITIALIZATION
# ==============================================================================
DB_FILE = "bowlers_vault.db"

def init_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS athletes (
            athlete_id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            bowling_arm TEXT,
            height_m REAL,
            action_type TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    c.execute('''
        CREATE TABLE IF NOT EXISTS deliveries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            athlete_id TEXT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            ffc_knee_angle REAL,
            release_knee_angle REAL,
            brace_delta REAL,
            release_height_m REAL,
            arm_slot_deg REAL,
            sequence_delay_ms REAL,
            coronal_lateral_tilt REAL,
            radar_speed_kmh REAL,
            notes TEXT,
            FOREIGN KEY (athlete_id) REFERENCES athletes(athlete_id)
        )
    ''')
    c.execute('''
        CREATE TABLE IF NOT EXISTS spells (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            athlete_id TEXT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            acute_load REAL,
            chronic_load REAL,
            acwr REAL,
            fdi_score REAL,
            balls_bowled INTEGER,
            avg_velocity REAL,
            brace_decay_deg REAL,
            FOREIGN KEY (athlete_id) REFERENCES athletes(athlete_id)
        )
    ''')
    conn.commit()
    conn.close()

init_db()

# ==============================================================================
# KINEMATIC COMPUTATION & RENDERING ENGINES
# ==============================================================================

def calculate_angle_3p(a, b, c):
    """Calculates 2D planar angle at vertex b given three coordinates."""
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360.0 - angle
    return angle

def draw_lead_knee_brace(frame_rgb, is_right_arm=True):
    """Annotates frame with lead leg skeletal lines and knee angle callout."""
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)
    res = pose.process(frame_rgb)
    pose.close()

    annotated = frame_rgb.copy()
    h, w, _ = annotated.shape

    if not res.pose_landmarks:
        return annotated, 155.0

    lm = res.pose_landmarks.landmark
    hip_idx = mp_pose.PoseLandmark.LEFT_HIP if is_right_arm else mp_pose.PoseLandmark.RIGHT_HIP
    knee_idx = mp_pose.PoseLandmark.LEFT_KNEE if is_right_arm else mp_pose.PoseLandmark.RIGHT_KNEE
    ankle_idx = mp_pose.PoseLandmark.LEFT_ANKLE if is_right_arm else mp_pose.PoseLandmark.RIGHT_ANKLE

    pt_hip = (int(lm[hip_idx].x * w), int(lm[hip_idx].y * h))
    pt_knee = (int(lm[knee_idx].x * w), int(lm[knee_idx].y * h))
    pt_ankle = (int(lm[ankle_idx].x * w), int(lm[ankle_idx].y * h))

    angle = calculate_angle_3p(
        [lm[hip_idx].x, lm[hip_idx].y],
        [lm[knee_idx].x, lm[knee_idx].y],
        [lm[ankle_idx].x, lm[ankle_idx].y]
    )

    cv2.line(annotated, pt_hip, pt_knee, (0, 255, 0), 4)
    cv2.line(annotated, pt_knee, pt_ankle, (0, 255, 0), 4)
    cv2.circle(annotated, pt_hip, 6, (255, 0, 0), -1)
    cv2.circle(annotated, pt_knee, 8, (0, 0, 255), -1)
    cv2.circle(annotated, pt_ankle, 6, (255, 255, 0), -1)

    cv2.putText(annotated, f"{angle:.1f} deg", (max(10, pt_knee[0] - 50), max(30, pt_knee[1] - 15)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    return annotated, angle

def calculate_coronal_metrics(frame_rgb, is_right_arm=True):
    """Calculates lateral trunk flexion and spinal tilt from coronal views."""
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)
    res = pose.process(frame_rgb)
    pose.close()

    annotated = frame_rgb.copy()
    h, w, _ = annotated.shape

    if not res.pose_landmarks:
        return {
            "lateral_flexion_deg": 0.0,
            "shoulder_tilt_deg": 0.0,
            "risk_level": "UNKNOWN",
            "annotated_frame": annotated
        }

    lm = res.pose_landmarks.landmark

    l_sh = np.array([lm[mp_pose.PoseLandmark.LEFT_SHOULDER].x * w, lm[mp_pose.PoseLandmark.LEFT_SHOULDER].y * h])
    r_sh = np.array([lm[mp_pose.PoseLandmark.RIGHT_SHOULDER].x * w, lm[mp_pose.PoseLandmark.RIGHT_SHOULDER].y * h])
    l_hip = np.array([lm[mp_pose.PoseLandmark.LEFT_HIP].x * w, lm[mp_pose.PoseLandmark.LEFT_HIP].y * h])
    r_hip = np.array([lm[mp_pose.PoseLandmark.RIGHT_HIP].x * w, lm[mp_pose.PoseLandmark.RIGHT_HIP].y * h])

    mid_shoulder = (l_sh + r_sh) / 2.0
    mid_hip = (l_hip + r_hip) / 2.0
    spine_vec = mid_shoulder - mid_hip
    vertical_ref = np.array([0.0, -1.0])

    norm_spine = np.linalg.norm(spine_vec)
    if norm_spine > 0:
        unit_spine = spine_vec / norm_spine
        dot_prod = np.dot(unit_spine, vertical_ref)
        lateral_flexion = np.arccos(np.clip(dot_prod, -1.0, 1.0)) * (180.0 / np.pi)
    else:
        lateral_flexion = 0.0

    sh_diff = r_sh - l_sh
    shoulder_tilt = np.abs(np.arctan2(sh_diff[1], sh_diff[0]) * (180.0 / np.pi))
    if shoulder_tilt > 90.0:
        shoulder_tilt = 180.0 - shoulder_tilt

    cv2.line(annotated, (int(mid_hip[0]), int(mid_hip[1])), (int(mid_shoulder[0]), int(mid_shoulder[1])), (0, 255, 0), 4)
    cv2.line(annotated, (int(mid_hip[0]), int(mid_hip[1])), (int(mid_hip[0]), int(mid_hip[1] - 180)), (255, 0, 0), 2)
    cv2.circle(annotated, (int(mid_shoulder[0]), int(mid_shoulder[1])), 6, (0, 0, 255), -1)
    cv2.circle(annotated, (int(mid_hip[0]), int(mid_hip[1])), 6, (255, 255, 0), -1)
    
    cv2.putText(annotated, f"Tilt: {lateral_flexion:.1f} deg", 
                (max(10, int(mid_shoulder[0]) - 60), max(30, int(mid_shoulder[1]) - 20)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    if lateral_flexion > 35.0:
        risk = "HIGH RISK"
    elif 25.0 <= lateral_flexion <= 35.0:
        risk = "MODERATE"
    else:
        risk = "LOW RISK"

    return {
        "lateral_flexion_deg": lateral_flexion,
        "shoulder_tilt_deg": shoulder_tilt,
        "risk_level": risk,
        "annotated_frame": annotated
    }

def generate_ghost_overlay(frame_a, frame_b, alpha=0.5):
    """Blends two video frames to visually expose kinematic deviations."""
    if frame_a.shape != frame_b.shape:
        frame_b = cv2.resize(frame_b, (frame_a.shape[1], frame_a.shape[0]))
    beta = 1.0 - alpha
    return cv2.addWeighted(frame_a, alpha, frame_b, beta, 0.0)

def align_delivery_sequences(frames_a, ffc_a, frames_b, ffc_b, window_before=15, window_after=25):
    """Synchronizes two frame sequences relative to their respective FFC timestamps."""
    start_a, end_a = max(0, ffc_a - window_before), min(len(frames_a), ffc_a + window_after)
    start_b, end_b = max(0, ffc_b - window_before), min(len(frames_b), ffc_b + window_after)
    
    lead_a = ffc_a - start_a
    lead_b = ffc_b - start_b
    common_lead = min(lead_a, lead_b)
    
    trail_a = end_a - ffc_a
    trail_b = end_b - ffc_b
    common_trail = min(trail_a, trail_b)
    
    sync_a = frames_a[ffc_a - common_lead : ffc_a + common_trail]
    sync_b = frames_b[ffc_b - common_lead : ffc_b + common_trail]
    
    return sync_a, sync_b, common_lead

@st.cache_data(show_spinner=False, max_entries=5)
def extract_video_frames(video_bytes, max_dimension=720):
    """Decodes video frames safely without memory locks."""
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    tfile.write(video_bytes)
    tfile.close()

    cap = cv2.VideoCapture(tfile.name)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps < 1:
        fps = 30.0

    raw_frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        h, w = frame.shape[:2]
        if max(h, w) > max_dimension:
            scale = max_dimension / max(h, w)
            frame = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        raw_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    cap.release()
    try:
        os.remove(tfile.name)
    except Exception:
        pass

    return raw_frames, fps

# ==============================================================================
# SIDEBAR: ATHLETE VAULT
# ==============================================================================
with st.sidebar:
    st.image("https://img.icons8.com/color/96/cricket.png", width=64)
    st.title("Athlete Vault")
    
    conn = sqlite3.connect(DB_FILE)
    athletes_df = pd.read_sql_query("SELECT athlete_id, name FROM athletes ORDER BY name ASC", conn)
    conn.close()
    
    athlete_list = athletes_df["name"].tolist() if not athletes_df.empty else []
    selected_athlete_name = st.selectbox("Select Athlete Profile", ["+ Add New Athlete"] + athlete_list)
    
    if selected_athlete_name == "+ Add New Athlete":
        st.subheader("New Athlete Onboarding")
        new_id = st.text_input("Athlete ID / BCCI Reg No", value=f"IND-{np.random.randint(1000, 9999)}")
        new_name = st.text_input("Full Name", value="Jasprit Bumrah")
        new_arm = st.selectbox("Bowling Arm", ["Right Arm", "Left Arm"])
        new_height = st.number_input("Standing Height (m)", min_value=1.50, max_value=2.20, value=1.84, step=0.01)
        new_action = st.selectbox("Action Classification", ["Front-On", "Side-On", "Semi-Open"])
        
        if st.button("💾 Register Athlete Profile"):
            conn = sqlite3.connect(DB_FILE)
            c = conn.cursor()
            c.execute("INSERT OR REPLACE INTO athletes (athlete_id, name, bowling_arm, height_m, action_type) VALUES (?, ?, ?, ?, ?)",
                      (new_id, new_name, new_arm, new_height, new_action))
            conn.commit()
            conn.close()
            st.success(f"Profile saved for {new_name}")
            st.rerun()
        active_athlete_id = new_id
        active_athlete_data = {"name": new_name, "arm": new_arm, "height": new_height, "action": new_action}
    else:
        conn = sqlite3.connect(DB_FILE)
        prof = pd.read_sql_query("SELECT * FROM athletes WHERE name = ?", conn, params=(selected_athlete_name,)).iloc[0]
        conn.close()
        active_athlete_id = prof["athlete_id"]
        active_athlete_data = {"name": prof["name"], "arm": prof["bowling_arm"], "height": prof["height_m"], "action": prof["action_type"]}
        
        st.markdown(f"**ID:** `{active_athlete_id}`")
        st.markdown(f"**Arm:** {active_athlete_data['arm']} | **Height:** {active_athlete_data['height']}m")
        st.markdown(f"**Action:** {active_athlete_data['action']}")

# ==============================================================================
# MAIN TABS LAYOUT
# ==============================================================================
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "🎯 Single Delivery Diagnostic",
    "🔬 Multi-Delivery Ghost Comparator",
    "📈 Spell FDI & ACWR Engine",
    "🏛️ Longitudinal Athlete History",
    "🔄 Multi-Angle 3D Biomechanics",
    "📋 S&C Periodization Planner",
    "🛡️ Squad Master Triage"
])

# ==============================================================================
# TAB 1: SINGLE DELIVERY DIAGNOSTIC
# ==============================================================================
with tab1:
    st.header("🎯 Single-Delivery Kinematic Diagnostic")
    st.caption("Precision sub-frame event locking, lead-knee angle differential, and automated release mechanics.")
    
    col_up1, col_up2 = st.columns([2, 1])
    with col_up1:
        uploaded_video = st.file_uploader("Upload Side-On Delivery Video (MP4/MOV)", type=["mp4", "mov", "avi"], key="tab1_upload")
    with col_up2:
        radar_speed = st.number_input("Radar Ball Speed (km/h)", min_value=80.0, max_value=165.0, value=142.5, step=0.5)

    if uploaded_video is not None:
        with st.spinner("Decoding delivery sequence..."):
            frames, fps = extract_video_frames(uploaded_video.read())
            total_frames = len(frames)
            is_right_arm = (active_athlete_data["arm"] == "Right Arm")

        st.info(f"Loaded **{total_frames} frames** | Effective Processing Rate: **{fps:.1f} FPS**")
        
        if "bfc_idx" not in st.session_state or st.session_state.bfc_idx >= total_frames:
            st.session_state.bfc_idx = int(total_frames * 0.30)
        if "ffc_idx" not in st.session_state or st.session_state.ffc_idx >= total_frames:
            st.session_state.ffc_idx = int(total_frames * 0.50)
        if "rel_idx" not in st.session_state or st.session_state.rel_idx >= total_frames:
            st.session_state.rel_idx = int(total_frames * 0.65)

        st.subheader("Sub-Frame Event Locking & Live Visual Alignment")
        col_s1, col_s2, col_s3 = st.columns(3)
        
        with col_s1:
            st.markdown("### 1. Back-Foot Contact (BFC)")
            bfc_val = st.slider("Scrub BFC Frame", 0, total_frames - 1, st.session_state.bfc_idx, key="slider_bfc")
            st.session_state.bfc_idx = bfc_val
            st.image(frames[st.session_state.bfc_idx], caption=f"BFC Frame #{st.session_state.bfc_idx}")

        with col_s2:
            st.markdown("### 2. Front-Foot Contact (FFC)")
            ffc_val = st.slider("Scrub FFC Frame", 0, total_frames - 1, st.session_state.ffc_idx, key="slider_ffc")
            st.session_state.ffc_idx = ffc_val
            annotated_ffc, ffc_knee_angle = draw_lead_knee_brace(frames[st.session_state.ffc_idx], is_right_arm=is_right_arm)
            st.image(annotated_ffc, caption=f"FFC Frame #{st.session_state.ffc_idx} (Lead Knee: {ffc_knee_angle:.1f}°)")

        with col_s3:
            st.markdown("### 3. Ball Release (BRS)")
            rel_val = st.slider("Scrub Release Frame", 0, total_frames - 1, st.session_state.rel_idx, key="slider_rel")
            st.session_state.rel_idx = rel_val
            annotated_rel, rel_knee_angle = draw_lead_knee_brace(frames[st.session_state.rel_idx], is_right_arm=is_right_arm)
            st.image(annotated_rel, caption=f"Release Frame #{st.session_state.rel_idx} (Lead Knee: {rel_knee_angle:.1f}°)")

        brace_delta = rel_knee_angle - ffc_knee_angle
        release_height = active_athlete_data["height"] * 1.15
        sequence_delay = ((st.session_state.rel_idx - st.session_state.ffc_idx) / fps) * 1000.0 if st.session_state.rel_idx > st.session_state.ffc_idx else 0.0
        
        st.subheader("Diagnostic Telemetry Results")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Lead Knee @ FFC", f"{ffc_knee_angle:.1f}°")
        m2.metric("Lead Knee @ Release", f"{rel_knee_angle:.1f}°")
        m3.metric("Brace Delta (Δ)", f"{brace_delta:+.1f}°")
        m4.metric("Sequence Delay", f"{sequence_delay:.0f} ms")
        
        if brace_delta > 2.0:
            st.success("✅ **Dominant Energy Transfer:** Lead knee actively extends through release (Rigid Front Leg Brace).")
        elif -2.0 <= brace_delta <= 2.0:
            st.warning("⚠️ **Neutral Transfer:** Knee angle maintained. Minimal kinetic amplification.")
        else:
            st.error("🚨 **High Energy Leakage / Knee Collapse:** Lead knee flexed post-FFC. High shear stress on lumbar spine.")
            
        if st.button("💾 Log Delivery to Athlete Vault"):
            conn = sqlite3.connect(DB_FILE)
            c = conn.cursor()
            c.execute('''
                INSERT INTO deliveries (
                    athlete_id, ffc_knee_angle, release_knee_angle, brace_delta,
                    release_height_m, arm_slot_deg, sequence_delay_ms, coronal_lateral_tilt,
                    radar_speed_kmh, notes
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (active_athlete_id, ffc_knee_angle, rel_knee_angle, brace_delta,
                  release_height, 48.5, sequence_delay, 22.0, radar_speed, "Single Delivery Diagnostic"))
            conn.commit()
            conn.close()
            st.success("Delivery telemetry saved to database.")

# ==============================================================================
# TAB 2: MULTI-DELIVERY GHOST COMPARATOR
# ==============================================================================
with tab2:
    st.header("🔬 Multi-Delivery Ghost Overlay Comparator")
    st.caption("Synchronize and blend two deliveries around Front-Foot Contact to identify kinematic decay or mechanical shifts.")
    
    col_g1, col_g2 = st.columns(2)
    with col_g1:
        v1 = st.file_uploader("Upload Delivery A (Baseline / Fresh)", type=["mp4", "mov", "avi"], key="ghost_v1")
    with col_g2:
        v2 = st.file_uploader("Upload Comparison Delivery B (Fatigue / Variant)", type=["mp4", "mov", "avi"], key="ghost_v2")
        
    if v1 is not None and v2 is not None:
        with st.spinner("Decoding video streams..."):
            frames_a, fps_a = extract_video_frames(v1.read())
            frames_b, fps_b = extract_video_frames(v2.read())
        
        st.success(f"Delivery A ({len(frames_a)} frames) & Delivery B ({len(frames_b)} frames) loaded.")
        
        col_sync1, col_sync2, col_blend = st.columns([1, 1, 1])
        with col_sync1:
            ffc_a = st.slider("Delivery A - FFC Plant Frame", 0, len(frames_a) - 1, int(len(frames_a) * 0.50), key="ffc_a_slider")
        with col_sync2:
            ffc_b = st.slider("Delivery B - FFC Plant Frame", 0, len(frames_b) - 1, int(len(frames_b) * 0.50), key="ffc_b_slider")
        with col_blend:
            alpha_val = st.slider("Ghost Opacity (A vs B)", 0.0, 1.0, 0.50, step=0.05, key="ghost_alpha_slider")
            
        sync_seq_a, sync_seq_b, ffc_offset = align_delivery_sequences(frames_a, ffc_a, frames_b, ffc_b)
        
        if len(sync_seq_a) > 0:
            st.subheader("Synchronized Kinematic Timeline")
            scrub_idx = st.slider(
                "🎞️ Scrub Synchronized Window (Centered at FFC = 0)", 
                0, len(sync_seq_a) - 1, ffc_offset,
                format="Frame %d",
                key="scrub_sync_timeline"
            )
            
            frame_disp_a = sync_seq_a[scrub_idx]
            frame_disp_b = sync_seq_b[scrub_idx]
            blended_frame = generate_ghost_overlay(frame_disp_a, frame_disp_b, alpha=alpha_val)
            
            col_v_a, col_v_ghost, col_v_b = st.columns(3)
            with col_v_a:
                st.markdown(f"**Delivery A (Frame #{ffc_a - ffc_offset + scrub_idx})**")
                st.image(frame_disp_a)
            with col_v_ghost:
                st.markdown(f"**⚡ Ghost Overlay (Alpha: {alpha_val:.2f})**")
                st.image(blended_frame)
            with col_v_b:
                st.markdown(f"**Delivery B (Frame #{ffc_b - ffc_offset + scrub_idx})**")
                st.image(frame_disp_b)

# ==============================================================================
# TAB 3: SPELL FDI & ACWR WORKLOAD ENGINE
# ==============================================================================
with tab3:
    st.header("📈 Spell Fatigue Degradation Index (FDI) & ACWR Engine")
    st.caption("Real-time pitch-side workload monitoring, multi-over fatigue decay, and lumbar stress fracture mitigation.")
    
    st.subheader("1. Acute:Chronic Workload Ratio (ACWR)")
    col_w1, col_w2, col_w3 = st.columns(3)
    with col_w1:
        acute_balls = st.number_input("Acute Load (Last 7 Days - Total Balls Bowled)", min_value=0, max_value=500, value=126, step=6)
    with col_w2:
        chronic_balls = st.number_input("Chronic Load (Last 28 Days - Weekly Average Balls)", min_value=1, max_value=500, value=102, step=6)
    with col_w3:
        acwr_val = acute_balls / chronic_balls if chronic_balls > 0 else 1.0
        st.metric("Current ACWR", f"{acwr_val:.2f}", delta=f"{acwr_val - 1.0:+.2f}")
        
    if acwr_val > 1.50:
        st.error("🚨 **CRITICAL OVERUSE SPIKE (ACWR > 1.50):** Elevated risk of lumbar stress fracture. Enforce bowling volume restriction (<18 balls next session).")
    elif 1.30 <= acwr_val <= 1.50:
        st.warning("⚠️ **ELEVATED WORKLOAD WARNING (1.30 - 1.50):** High acute fatigue accumulation. Monitor front-leg brace decay closely.")
    elif 0.80 <= acwr_val < 1.30:
        st.success("🟢 **OPTIMAL WORKLOAD SWEET SPOT (0.80 - 1.30):** Safe conditioning zone with high fitness adaptation.")
    else:
        st.warning("⚠️ **UNDERPREPARED ZONE (ACWR < 0.80):** High vulnerability to acute workload spikes when returning to match intensity.")

    st.markdown("---")
    st.subheader("2. Spell Fatigue Degradation Index (FDI)")
    st.caption("Measure kinematic decay between the bowler's 1st Over vs the end of the Spell.")
    
    col_f1, col_f2, col_f3 = st.columns(3)
    with col_f1:
        v_fresh = st.number_input("Fresh Ball Velocity (km/h)", value=142.5, step=0.5)
        b_fresh = st.number_input("Fresh Brace Delta (°)", value=+4.2, step=0.5)
    with col_f2:
        v_fatigued = st.number_input("Fatigued Ball Velocity (km/h)", value=136.8, step=0.5)
        b_fatigued = st.number_input("Fatigued Brace Delta (°)", value=-3.1, step=0.5)
    with col_f3:
        vel_drop_pct = max(0.0, ((v_fresh - v_fatigued) / v_fresh) * 100.0)
        brace_loss_deg = max(0.0, b_fresh - b_fatigued)
        fdi_score = (vel_drop_pct * 0.4) + (brace_loss_deg * 0.6)
        st.metric("Spell FDI Score", f"{fdi_score:.1f} / 100")
        
    if fdi_score > 12.0:
        st.error(f"🚨 **HIGH MECHANICAL COLLAPSE (FDI: {fdi_score:.1f}):** Knee brace decay ({brace_loss_deg:.1f}°) and velocity loss ({vel_drop_pct:.1f}%) indicate neuromuscular fatigue. Terminate spell immediately.")
    elif 6.0 <= fdi_score <= 12.0:
        st.warning(f"⚠️ **MODERATE FATIGUE ACCUMULATION (FDI: {fdi_score:.1f}):** Mechanics softening. Maximum 1 additional over permitted.")
    else:
        st.success(f"🟢 **EXCELLENT KINEMATIC INTEGRITY (FDI: {fdi_score:.1f}):** Power delivery and front-leg bracing remain stable across overs.")

    if st.button("💾 Log Spell Workload & FDI to Database"):
        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        c.execute('''
            INSERT INTO spells (
                athlete_id, acute_load, chronic_load, acwr, fdi_score, balls_bowled, avg_velocity, brace_decay_deg
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (active_athlete_id, acute_balls, chronic_balls, acwr_val, fdi_score, 24, (v_fresh + v_fatigued)/2.0, brace_loss_deg))
        conn.commit()
        conn.close()
        st.success("Spell telemetry recorded in Athlete Vault.")

# ==============================================================================
# TAB 4: LONGITUDINAL ATHLETE HISTORY & VAULT
# ==============================================================================
with tab4:
    st.header("🏛️ Longitudinal Athlete History & Database Vault")
    st.caption("Long-term tracking of front-foot brace mechanics, ball velocity trajectories, and spell fatigue trends.")
    
    conn = sqlite3.connect(DB_FILE)
    deliveries_df = pd.read_sql_query('''
        SELECT id, timestamp, ffc_knee_angle, release_knee_angle, brace_delta, 
               sequence_delay_ms, radar_speed_kmh, notes 
        FROM deliveries 
        WHERE athlete_id = ? 
        ORDER BY timestamp DESC
    ''', conn, params=(active_athlete_id,))
    
    spells_df = pd.read_sql_query('''
        SELECT id, timestamp, acute_load, chronic_load, acwr, fdi_score, balls_bowled, avg_velocity, brace_decay_deg
        FROM spells 
        WHERE athlete_id = ? 
        ORDER BY timestamp DESC
    ''', conn, params=(active_athlete_id,))
    conn.close()
    
    if deliveries_df.empty:
        st.info(f"No delivery logs recorded yet for **{active_athlete_data['name']}** (`{active_athlete_id}`). Log deliveries in Tab 1 to populate longitudinal history.")
    else:
        st.subheader(f"📊 Delivery Telemetry History: {active_athlete_data['name']}")
        
        avg_speed = deliveries_df["radar_speed_kmh"].mean()
        avg_brace = deliveries_df["brace_delta"].mean()
        max_speed = deliveries_df["radar_speed_kmh"].max()
        total_balls = len(deliveries_df)
        
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Total Vault Deliveries", f"{total_balls} balls")
        k2.metric("Mean Radar Velocity", f"{avg_speed:.1f} km/h")
        k3.metric("Peak Radar Velocity", f"{max_speed:.1f} km/h")
        k4.metric("Mean Brace Delta (Δ)", f"{avg_brace:+.1f}°")
        
        fig_trend = go.Figure()
        fig_trend.add_trace(go.Scatter(
            x=deliveries_df["timestamp"], 
            y=deliveries_df["radar_speed_kmh"],
            mode='lines+markers',
            name='Velocity (km/h)',
            line=dict(color='#00D26A', width=3)
        ))
        fig_trend.add_trace(go.Scatter(
            x=deliveries_df["timestamp"], 
            y=deliveries_df["brace_delta"],
            mode='lines+markers',
            name='Brace Delta (°)',
            yaxis='y2',
            line=dict(color='#FFA500', width=3, dash='dot')
        ))
        fig_trend.update_layout(
            title="Longitudinal Velocity vs Lead Knee Brace Efficiency",
            xaxis=dict(title="Session Timestamp"),
            yaxis=dict(title="Ball Speed (km/h)", side="left"),
            yaxis2=dict(title="Brace Delta (°)", side="right", overlaying="y"),
            template="plotly_dark",
            legend=dict(x=0.01, y=0.99)
        )
        st.plotly_chart(fig_trend)
        
        st.markdown("### Raw Delivery Records")
        st.dataframe(deliveries_df)

    if not spells_df.empty:
        st.markdown("---")
        st.subheader("📋 Recorded Spell Workloads & Fatigue Degradation Logs")
        st.dataframe(spells_df)

# ==============================================================================
# TAB 5: MULTI-ANGLE 3D CORONAL TRACKING
# ==============================================================================
with tab5:
    st.header("🔄 Multi-Angle 3D Biomechanics & Coronal Spine Tracking")
    st.caption("Quantify lateral trunk flexion, spinal tilt relative to gravity, and contralateral lumbar shear risk.")
    
    col_c_up, col_c_meta = st.columns([2, 1])
    with col_c_up:
        coronal_video = st.file_uploader("Upload Front-On or Rear-On (Coronal) Delivery Video", type=["mp4", "mov", "avi"], key="coronal_upload")
    with col_c_meta:
        st.markdown(f"**Target Bowler:** `{active_athlete_data['name']}`")
        st.markdown(f"**Dominant Arm:** `{active_athlete_data['arm']}`")
        st.markdown(f"**Action Category:** `{active_athlete_data['action']}`")
        
    if coronal_video is not None:
        with st.spinner("Decoding coronal video feed..."):
            c_frames, c_fps = extract_video_frames(coronal_video.read())
            c_total = len(c_frames)
            
        st.info(f"Loaded **{c_total} frames** @ **{c_fps:.1f} FPS** (Coronal Perspective)")
        
        c_rel_frame = st.slider("Scrub to Ball Release / Max Lateral Trunk Flexion Frame", 0, c_total - 1, int(c_total * 0.50), key="coronal_scrubber")
        
        is_right_arm = (active_athlete_data["arm"] == "Right Arm")
        c_metrics = calculate_coronal_metrics(c_frames[c_rel_frame], is_right_arm=is_right_arm)
        
        col_c_view, col_c_diag = st.columns([1, 1])
        with col_c_view:
            st.image(c_metrics["annotated_frame"], caption=f"Coronal Frame #{c_rel_frame} (Spinal Vector vs Plumb Line)")
            
        with col_c_diag:
            st.subheader("Spine Telemetry & Stress Analysis")
            
            c_lat = c_metrics["lateral_flexion_deg"]
            c_sh = c_metrics["shoulder_tilt_deg"]
            
            cm1, cm2 = st.columns(2)
            cm1.metric("Lateral Trunk Flexion", f"{c_lat:.1f}°")
            cm2.metric("Shoulder Plumb Tilt", f"{c_sh:.1f}°")
            
            if c_lat > 35.0:
                st.error("🚨 **CRITICAL LUMBAR SHEAR (>35°):** Severe lateral trunk flexion creates massive asymmetrical shear loading on contralateral L4–L5 pars interarticularis. Requires immediate posture modification.")
            elif 25.0 <= c_lat <= 35.0:
                st.warning("⚠️ **MODERATE SHEAR (25° - 35°):** Borderline lateral trunk flexion. High physical demand on lateral kinetic sling (obliques/quadratus lumborum).")
            else:
                st.success("🟢 **BIOMECHANICALLY SOUND (<25°):** Upright coronal spine posture ensures vertical force dissipation and low stress fracture probability.")

# ==============================================================================
# TAB 6: S&C PERIODIZATION PLANNER & PRESCRIPTION ENGINE
# ==============================================================================
with tab6:
    st.header("📋 S&C Periodization Planner & Prescription Engine")
    st.caption("Auto-generate elite gym interventions driven directly by athlete kinematic profiles and microcycle phases.")
    
    st.subheader("1. Athlete Kinematic Diagnostic Trigger")
    
    conn = sqlite3.connect(DB_FILE)
    latest_deliv = pd.read_sql_query(
        "SELECT brace_delta, coronal_lateral_tilt, sequence_delay_ms FROM deliveries WHERE athlete_id = ? ORDER BY timestamp DESC LIMIT 1",
        conn, params=(active_athlete_id,)
    )
    conn.close()
    
    default_deficit = "Lead Knee Softening / Collapse"
    if not latest_deliv.empty:
        last_brace = latest_deliv.iloc[0]["brace_delta"]
        last_coronal = latest_deliv.iloc[0]["coronal_lateral_tilt"]
        if last_brace < 0.0:
            default_deficit = "Lead Knee Softening / Collapse"
        elif last_coronal > 35.0:
            default_deficit = "Excessive Lateral Trunk Flexion (>35°)"
        else:
            default_deficit = "Kinematic Sequence Decoupling"
            
    selected_deficit = st.selectbox(
        "Identified Mechanical Weak Link (Biomechanical Trigger)",
        [
            "Lead Knee Softening / Collapse",
            "Excessive Lateral Trunk Flexion (>35°)",
            "Kinematic Sequence Decoupling",
            "Posterior Chain Rate of Force Development (RFD) Deficit"
        ],
        index=["Lead Knee Softening / Collapse", "Excessive Lateral Trunk Flexion (>35°)", "Kinematic Sequence Decoupling", "Posterior Chain Rate of Force Development (RFD) Deficit"].index(default_deficit)
    )
    
    col_p1, col_p2 = st.columns(2)
    with col_p1:
        meso_phase = st.selectbox(
            "Periodization Meso-Phase",
            [
                "Phase 1: Anatomical Adaptation & Eccentric Strength",
                "Phase 2: Maximal Strength & Braking Impulse (BCCI Off-Season)",
                "Phase 3: Rate of Force Development & Rotational Power (Pre-Season)",
                "Phase 4: In-Season High-Velocity Maintenance & Anti-Fatigue"
            ]
        )
    with col_p2:
        weekly_frequency = st.slider("Weekly Gym Frequency (Sessions/Week)", 2, 4, 3)
        
    st.markdown("---")
    st.subheader(f"2. Prescription Protocol: {active_athlete_data['name']}")
    
    if "Lead Knee Softening" in selected_deficit:
        st.markdown("""
        <div class="metric-card">
            <h4>🎯 Target Mechanism: Front-Foot Contact Braking Impulse</h4>
            <p>Develop eccentric quadriceps capacity, patellar tendon stiffness, and gluteus medius stabilization to absorb 6-9x bodyweight ground reaction forces without knee flexion collapse.</p>
        </div>
        """, unsafe_allow_html=True)
        
        presc_data = {
            "Exercise Name": [
                "Hatfield Safety Squat (Lead Leg Emphasized)",
                "Heavy Eccentric Single-Leg Leg Press (1.5x Bodyweight)",
                "Drop Jumps to Isometric Stick (30cm Box)",
                "Tibialis Anterior Raises & Soles Eccentrics"
            ],
            "Sets x Reps": ["4 sets x 5 reps", "3 sets x 6 reps per leg", "4 sets x 4 reps", "3 sets x 15 reps"],
            "Tempo / Loading": ["4-1-X-0 (85% 1RM)", "4-0-1-0 (RPE 8.5)", "Explosive Stick (<200ms GCT)", "2-1-2-0 (Bodyweight + 10kg)"],
            "Rest Interval": ["180s", "120s", "90s", "60s"],
            "Target Adaptation": ["Max Braking Force", "Eccentric Quad Overload", "Reactive Stretch-Shortening Cycle", "Ankle Joint Stiffness"]
        }
    elif "Lateral Trunk Flexion" in selected_deficit:
        st.markdown("""
        <div class="metric-card">
            <h4>🎯 Target Mechanism: Anti-Lateral Flexion & Lumbar Spine Shearing Shield</h4>
            <p>Strengthen the lateral kinetic sling (Quadratus Lumborum, Internal/External Obliques, Gluteus Medius) to resist contralateral spine buckling and mitigate pars interarticularis micro-trauma.</p>
        </div>
        """, unsafe_allow_html=True)
        
        presc_data = {
            "Exercise Name": [
                "Heavy Asymmetrical Suitcase Carries",
                "Copenhagen Adductor Planks (Long Lever)",
                "Pallof Press with Overhead Iso-Hold",
                "Single-Arm Landmine Push-Press with Rotational Deceleration"
            ],
            "Sets x Reps": ["4 sets x 35 meters / side", "3 sets x 40s hold / side", "3 sets x 10 reps + 5s hold", "4 sets x 5 reps / side"],
            "Tempo / Loading": ["Controlled Stride (32kg Kettlebell)", "Static Isometric Hold", "3-2-X-0", "Explosive drive, 3s descent"],
            "Rest Interval": ["90s", "60s", "60s", "90s"],
            "Target Adaptation": ["Anti-Lateral Flexion Capacity", "Groin & Adductor Strength", "Rotary Torso Stability", "Cross-Body Kinetic Transfer"]
        }
    elif "Kinematic Sequence" in selected_deficit:
        st.markdown("""
        <div class="metric-card">
            <h4>🎯 Target Mechanism: Segmental Pelvis-to-Thorax Kinetic Timing</h4>
            <p>Optimize the sequential delay between pelvis rotation, thoracic whip, and shoulder-internal rotation to maximize ball velocity without arm drag.</p>
        </div>
        """, unsafe_allow_html=True)
        
        presc_data = {
            "Exercise Name": [
                "Rotational Medball Shot-Put Against Solid Wall (3kg-5kg)",
                "Half-Kneeling Cable Hip-to-Shoulder Diagonal Chops",
                "Thoracic Spine Windmills & Foam Roller Openers",
                "Band-Resisted Lead Hip Internal Rotation Snaps"
            ],
            "Sets x Reps": ["4 sets x 6 throws / side", "3 sets x 8 reps / side", "2 sets x 10 reps / side", "3 sets x 12 snaps / side"],
            "Tempo / Loading": ["Max Intent Explosive", "2-0-X-0 (RPE 7.5)", "Controlled Mobility", "Fast Snapping Velocity"],
            "Rest Interval": ["90s", "60s", "45s", "60s"],
            "Target Adaptation": ["Rotational RFD", "Anterior Core Sling Power", "Thoracic Mobility", "Hip-Shoulder Separation Speed"]
        }
    else:
        st.markdown("""
        <div class="metric-card">
            <h4>🎯 Target Mechanism: Posterior Kinetic Chain Stretch-Shortening Cycle</h4>
            <p>Enhance hamstring eccentric force absorption and gluteal extension velocity during the final run-up stride and gathering phase.</p>
        </div>
        """, unsafe_allow_html=True)
        
        presc_data = {
            "Exercise Name": [
                "Nordic Hamstring Curls (Banded Assist to Flat)",
                "Trap Bar Deadlift (High Velocity Triples)",
                "Single-Leg Romanian Deadlift with Kettlebell",
                "Heavy Sled Push & Sprint Accelerations"
            ],
            "Sets x Reps": ["4 sets x 5 reps", "5 sets x 3 reps (82.5% 1RM)", "3 sets x 6 reps / leg", "5 sets x 15m sprints"],
            "Tempo / Loading": ["4-0-X-0", "Explosive Con-Drive", "3-1-1-0", "Max Sprint Output"],
            "Rest Interval": ["120s", "180s", "90s", "120s"],
            "Target Adaptation": ["Distal Hamstring Injury Resilience", "Posterior Chain Force Output", "Unilateral Pelvic Stability", "Horizontal Force Production"]
        }
        
    presc_df = pd.DataFrame(presc_data)
    st.dataframe(presc_df)

# ==============================================================================
# TAB 7: SQUAD MASTER TRIAGE & MEDICAL CLEARANCE CONSOLE (ACTIVATED)
# ==============================================================================
with tab7:
    st.header("🛡️ Squad Master Triage & Medical Clearance Console")
    st.caption("High-performance multi-bowler triage console for national coaches, head physios, and lead S&C staff.")
    
    conn = sqlite3.connect(DB_FILE)
    squad_summary = pd.read_sql_query('''
        SELECT 
            a.athlete_id, 
            a.name, 
            a.bowling_arm, 
            a.action_type,
            COUNT(d.id) as logged_balls,
            ROUND(AVG(d.radar_speed_kmh), 1) as avg_speed_kmh,
            ROUND(AVG(d.brace_delta), 1) as avg_brace_delta,
            MAX(d.coronal_lateral_tilt) as max_coronal_tilt,
            (SELECT acwr FROM spells WHERE spells.athlete_id = a.athlete_id ORDER BY timestamp DESC LIMIT 1) as latest_acwr
        FROM athletes a
        LEFT JOIN deliveries d ON a.athlete_id = d.athlete_id
        GROUP BY a.athlete_id
    ''', conn)
    conn.close()
    
    if squad_summary.empty:
        st.info("No athletes registered in the database. Add athletes in the left sidebar.")
    else:
        # Compute Triage Status dynamically
        status_list = []
        notes_list = []
        for idx, row in squad_summary.iterrows():
            acwr = row["latest_acwr"] if pd.notnull(row["latest_acwr"]) else 1.0
            brace = row["avg_brace_delta"] if pd.notnull(row["avg_brace_delta"]) else 0.0
            tilt = row["max_coronal_tilt"] if pd.notnull(row["max_coronal_tilt"]) else 0.0
            
            if acwr > 1.50 or tilt > 35.0:
                status_list.append("🔴 MEDICAL INTERVENTION")
                notes_list.append("Critical ACWR spike (>1.50) or high lumbar shear.")
            elif acwr < 0.80 or brace < -2.0:
                status_list.append("🟡 RESTRICTED WORKLOAD")
                notes_list.append("Underprepared workload or lead-knee collapse.")
            else:
                status_list.append("🟢 CLEARED FOR MATCHES")
                notes_list.append("Biomechanical parameters & ACWR in optimal zone.")
                
        squad_summary["Triage Status"] = status_list
        squad_summary["Clinical Remarks"] = notes_list
        
        # Squad KPI Cards
        col_t1, col_t2, col_t3, col_t4 = st.columns(4)
        total_bowlers = len(squad_summary)
        cleared_count = status_list.count("🟢 CLEARED FOR MATCHES")
        restricted_count = status_list.count("🟡 RESTRICTED WORKLOAD")
        red_count = status_list.count("🔴 MEDICAL INTERVENTION")
        
        col_t1.metric("Squad Size", f"{total_bowlers} Bowlers")
        col_t2.metric("Cleared (Full Intensity)", f"{cleared_count}", delta_color="normal")
        col_t3.metric("Restricted / Technical", f"{restricted_count}", delta_color="off")
        col_t4.metric("Medical Intervention", f"{red_count}", delta_color="inverse")
        
        st.markdown("### Squad Biomechanical & Workload Roster")
        st.dataframe(squad_summary)
        
        st.markdown("---")
        st.subheader("Individual Bowler Deep-Dive Clearance")
        selected_triage_athlete = st.selectbox("Select Bowler for Triage Inspection", squad_summary["name"].tolist())
        bowler_row = squad_summary[squad_summary["name"] == selected_triage_athlete].iloc[0]
        
        col_b1, col_b2 = st.columns([1, 1])
        with col_b1:
            st.markdown(f"**Athlete Name:** `{bowler_row['name']}` ({bowler_row['athlete_id']})")
            st.markdown(f"**Bowling Arm:** `{bowler_row['bowling_arm']}` | **Action:** `{bowler_row['action_type']}`")
            st.markdown(f"**Current Triage Status:** **{bowler_row['Triage Status']}**")
        with col_b2:
            st.markdown(f"**Latest ACWR:** `{bowler_row['latest_acwr']}`")
            st.markdown(f"**Mean Knee Brace Delta:** `{bowler_row['avg_brace_delta']}°`")
            st.markdown(f"**Clinical Directive:** `{bowler_row['Clinical Remarks']}`")