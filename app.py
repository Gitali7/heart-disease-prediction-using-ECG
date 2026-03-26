"""
app.py
=======
Main Streamlit application for AI-Based Heart Disease Detection.

Run with:
    streamlit run app.py

Architecture:
- Uses Streamlit session state to hold model across reruns (avoids reloading)
- All image processing happens in-memory (no disk writes)
- Demo mode activates automatically when no trained model is found
- Grad-CAM heatmap shows model attention regions for explainability
"""

import os
import sys
import numpy as np
from PIL import Image
import streamlit as st
import cv2
import io
import time

# ─── Page Config (MUST be first Streamlit call) ───────────────────────────────
st.set_page_config(
    page_title="CardioScan AI",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─── Module Imports ───────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from preprocessing.image_processing import preprocess_for_model, load_image
from prediction.predictor import (
    load_model, predict, get_gradcam, overlay_heatmap,
    demo_predict, demo_gradcam,
    demo_predict_ecg, demo_gradcam_ecg,
    demo_predict_record, demo_gradcam_record,
    demo_gradcam_ct, demo_gradcam_echo,
    load_zero_shot_model, zero_shot_predict
)
from recommendation.medical_advice import format_full_report, get_condition_advice
from utils.helpers import model_exists, get_model_path, validate_image_file, create_sample_xray

# ─── Custom CSS ───────────────────────────────────────────────────────────────
# Design philosophy: Clinical dark theme with warm accents.
# Inspired by medical imaging software (dark backgrounds reduce eye strain,
# improve image contrast perception). Red accent = cardiac / urgency.

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=Playfair+Display:wght@700&family=JetBrains+Mono:wght@400;600&display=swap');

/* ─── Reset & Base ─── */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif !important;
    background-color: #0a0d14 !important;
    color: #e2e8f0 !important;
}

.main { background-color: #0a0d14 !important; }
.block-container { padding: 0 2rem 2rem 2rem !important; max-width: 1400px !important; }

/* Hide default Streamlit elements */
#MainMenu, footer, .stDeployButton { visibility: hidden; }
header[data-testid="stHeader"] { background: transparent !important; }

/* ─── Header ─── */
.app-header {
    background: linear-gradient(135deg, #0d1117 0%, #161b27 50%, #0d1117 100%);
    border-bottom: 1px solid rgba(220, 38, 38, 0.3);
    padding: 2rem 2rem 1.5rem;
    margin: -2rem -2rem 2rem -2rem;
    position: relative;
    overflow: hidden;
}

.app-header::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0; bottom: 0;
    background: radial-gradient(ellipse at 20% 50%, rgba(220,38,38,0.08) 0%, transparent 60%),
                radial-gradient(ellipse at 80% 50%, rgba(59,130,246,0.05) 0%, transparent 60%);
    pointer-events: none;
}

.header-content { display: flex; align-items: center; gap: 1.5rem; position: relative; z-index: 1; }

.header-icon {
    font-size: 3.5rem;
    animation: pulse-icon 2s ease-in-out infinite;
}

@keyframes pulse-icon {
    0%, 100% { transform: scale(1); }
    50% { transform: scale(1.08); }
}

.header-title {
    font-family: 'Playfair Display', serif;
    font-size: 2.4rem;
    font-weight: 700;
    color: #fff;
    margin: 0;
    line-height: 1.1;
    letter-spacing: -0.5px;
}

.header-subtitle {
    font-size: 0.95rem;
    color: #94a3b8;
    margin: 0.3rem 0 0;
    font-weight: 400;
    letter-spacing: 0.3px;
}

.header-badge {
    margin-left: auto;
    background: rgba(220, 38, 38, 0.15);
    border: 1px solid rgba(220, 38, 38, 0.4);
    color: #f87171;
    padding: 0.3rem 0.9rem;
    border-radius: 999px;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 1.5px;
    text-transform: uppercase;
}

/* ─── Upload Zone ─── */
.upload-zone {
    border: 2px dashed rgba(99, 102, 241, 0.4);
    border-radius: 16px;
    background: rgba(15, 20, 35, 0.8);
    padding: 2.5rem;
    text-align: center;
    transition: all 0.3s ease;
    cursor: pointer;
}

.upload-zone:hover {
    border-color: rgba(99, 102, 241, 0.8);
    background: rgba(99, 102, 241, 0.05);
}

/* ─── Cards ─── */
.glass-card {
    background: rgba(15, 22, 36, 0.9);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 16px;
    padding: 1.5rem;
    margin-bottom: 1rem;
    backdrop-filter: blur(10px);
}

.result-card-positive {
    background: linear-gradient(135deg, rgba(220,38,38,0.08) 0%, rgba(15,22,36,0.95) 100%);
    border: 1px solid rgba(220, 38, 38, 0.35);
    border-radius: 16px;
    padding: 1.5rem;
    margin-bottom: 1rem;
}

.result-card-negative {
    background: linear-gradient(135deg, rgba(0,200,83,0.08) 0%, rgba(15,22,36,0.95) 100%);
    border: 1px solid rgba(0, 200, 83, 0.35);
    border-radius: 16px;
    padding: 1.5rem;
    margin-bottom: 1rem;
}

/* ─── Prediction Badge ─── */
.prediction-badge-positive {
    display: inline-block;
    background: rgba(220,38,38,0.2);
    border: 1.5px solid #dc2626;
    color: #fca5a5;
    padding: 0.5rem 1.5rem;
    border-radius: 999px;
    font-weight: 700;
    font-size: 1rem;
    letter-spacing: 1px;
    text-transform: uppercase;
}

.prediction-badge-negative {
    display: inline-block;
    background: rgba(0,200,83,0.15);
    border: 1.5px solid #00c853;
    color: #69f0ae;
    padding: 0.5rem 1.5rem;
    border-radius: 999px;
    font-weight: 700;
    font-size: 1rem;
    letter-spacing: 1px;
    text-transform: uppercase;
}

/* ─── Confidence Bar ─── */
.confidence-container { margin: 1rem 0; }
.confidence-label {
    display: flex;
    justify-content: space-between;
    font-size: 0.8rem;
    color: #94a3b8;
    margin-bottom: 0.4rem;
    font-weight: 500;
}
.confidence-bar-bg {
    width: 100%;
    height: 10px;
    background: rgba(255,255,255,0.08);
    border-radius: 999px;
    overflow: hidden;
}
.confidence-bar-fill {
    height: 100%;
    border-radius: 999px;
    transition: width 1s ease;
}

/* ─── Risk Badge ─── */
.risk-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    padding: 0.3rem 0.9rem;
    border-radius: 8px;
    font-size: 0.8rem;
    font-weight: 700;
    letter-spacing: 0.5px;
}

/* ─── Section Labels ─── */
.section-label {
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #64748b;
    margin-bottom: 0.8rem;
}

.condition-chip {
    display: inline-block;
    background: rgba(99, 102, 241, 0.12);
    border: 1px solid rgba(99, 102, 241, 0.3);
    color: #a5b4fc;
    padding: 0.25rem 0.8rem;
    border-radius: 8px;
    font-size: 0.82rem;
    font-weight: 500;
    margin: 0.2rem 0.2rem 0.2rem 0;
}

/* ─── Advice List ─── */
.advice-item {
    display: flex;
    align-items: flex-start;
    gap: 0.6rem;
    padding: 0.5rem 0;
    border-bottom: 1px solid rgba(255,255,255,0.04);
    font-size: 0.87rem;
    color: #cbd5e1;
    line-height: 1.5;
}
.advice-item:last-child { border-bottom: none; }
.advice-dot {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    margin-top: 7px;
    flex-shrink: 0;
}

/* ─── Demo Banner ─── */
.demo-banner {
    background: linear-gradient(135deg, rgba(245,158,11,0.1), rgba(245,158,11,0.05));
    border: 1px solid rgba(245,158,11,0.4);
    border-radius: 10px;
    padding: 0.7rem 1rem;
    font-size: 0.82rem;
    color: #fbbf24;
    margin-bottom: 1rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

/* ─── Step Indicator ─── */
.step-indicator {
    display: flex;
    align-items: center;
    gap: 1rem;
    margin-bottom: 2rem;
}
.step-dot {
    width: 32px; height: 32px;
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 0.8rem; font-weight: 700;
    flex-shrink: 0;
}
.step-dot-active { background: #dc2626; color: white; }
.step-dot-done { background: #16a34a; color: white; }
.step-dot-inactive { background: rgba(255,255,255,0.1); color: #64748b; }
.step-line { flex: 1; height: 1px; background: rgba(255,255,255,0.1); }

/* ─── Image Display ─── */
.img-container {
    border-radius: 12px;
    overflow: hidden;
    border: 1px solid rgba(255,255,255,0.08);
}

/* ─── Streamlit element overrides ─── */
.stButton > button {
    background: linear-gradient(135deg, #dc2626, #b91c1c) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 600 !important;
    font-size: 1rem !important;
    padding: 0.65rem 2.5rem !important;
    letter-spacing: 0.3px !important;
    transition: all 0.2s !important;
    box-shadow: 0 4px 15px rgba(220,38,38,0.3) !important;
    width: 100% !important;
}

.stButton > button:hover {
    background: linear-gradient(135deg, #ef4444, #dc2626) !important;
    box-shadow: 0 6px 20px rgba(220,38,38,0.45) !important;
    transform: translateY(-1px) !important;
}

div[data-testid="stFileUploader"] {
    background: rgba(15, 22, 36, 0.8) !important;
    border: 2px dashed rgba(99, 102, 241, 0.4) !important;
    border-radius: 16px !important;
    padding: 1rem !important;
}

div[data-testid="stFileUploader"] label {
    color: #94a3b8 !important;
    font-size: 0.9rem !important;
}

.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: #64748b !important;
    font-weight: 500 !important;
    font-size: 0.88rem !important;
    padding: 0.5rem 1.2rem !important;
}
.stTabs [aria-selected="true"] {
    color: #e2e8f0 !important;
    border-bottom: 2px solid #dc2626 !important;
}
.stTabs [data-baseweb="tab-list"] {
    background: transparent !important;
    border-bottom: 1px solid rgba(255,255,255,0.08) !important;
}

.stSpinner > div { border-top-color: #dc2626 !important; }

hr { border-color: rgba(255,255,255,0.06) !important; }

div[data-testid="metric-container"] {
    background: rgba(15,22,36,0.9) !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    border-radius: 12px !important;
    padding: 1rem !important;
}
</style>
""", unsafe_allow_html=True)


# ─── Session State Initialization ────────────────────────────────────────────

def init_session_state():
    """Initialize all session state variables."""
    defaults = {
        "model": None,
        "processor": None,
        "model_loaded": False,
        "analysis_done": False,
        "uploaded_img": None,
        "report": None,
        "cam_overlay": None,
        "original_img": None,
        "image_type": "Chest X-Ray",
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


init_session_state()


# ─── Model Loading (cached) ───────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def get_model():
    """
    Load CLIP model once and cache in Streamlit's resource cache.
    Provides Zero-Shot Capabilities for all medical images.
    """
    try:
        model, processor = load_zero_shot_model()
        return model, processor
    except ImportError:
        st.error("Transformers library not installed. Please install it to use the new prediction mode.")
        return None, None


# ─── Rendering Helpers ────────────────────────────────────────────────────────

def render_header():
    st.markdown("""
    <div class="app-header">
        <div class="header-content">
            <div class="header-icon">🫀</div>
            <div>
                <h1 class="header-title">CardioScan AI</h1>
                <p class="header-subtitle">Deep Learning Heart Disease Detection from Chest X-Ray Images</p>
            </div>
            <div class="header-badge">Research Tool</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_confidence_bar(confidence_pct: float, is_positive: bool):
    color = "#dc2626" if is_positive else "#00c853"
    st.markdown(f"""
    <div class="confidence-container">
        <div class="confidence-label">
            <span>Model Confidence</span>
            <span style="color: {color}; font-weight: 700;">{confidence_pct}%</span>
        </div>
        <div class="confidence-bar-bg">
            <div class="confidence-bar-fill"
                 style="width: {confidence_pct}%; background: linear-gradient(90deg, {color}88, {color});"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_risk_badge(risk: dict):
    color = risk["color"]
    level = risk["level"]
    st.markdown(f"""
    <div class="risk-badge" style="background: {color}20; border: 1px solid {color}60; color: {color};">
        ⬤ &nbsp; RISK LEVEL: {level}
    </div>
    """, unsafe_allow_html=True)


def render_advice_list(items: list, dot_color: str = "#6366f1"):
    html = ""
    for item in items:
        html += f"""
        <div class="advice-item">
            <div class="advice-dot" style="background: {dot_color};"></div>
            <span>{item}</span>
        </div>"""
    st.markdown(html, unsafe_allow_html=True)


def render_condition_chips(conditions: list):
    chips = "".join(f'<span class="condition-chip">{c}</span>' for c in conditions)
    st.markdown(chips, unsafe_allow_html=True)


def render_demo_banner():
    st.markdown("""
    <div class="demo-banner">
        ⚡ <strong>Demo Mode</strong> — No trained model found at <code>model/heart_xray_model.pth</code>.
        Running mock predictions for UI demonstration. Train the model to enable real analysis.
    </div>
    """, unsafe_allow_html=True)


# ─── Main Analysis Function ───────────────────────────────────────────────────

def run_analysis(uploaded_file, model, processor, image_type: str = "Chest X-Ray", patient_data: dict = None):
    """
    Core analysis pipeline:
    1. Load and preprocess image
    2. Run prediction (real or demo)
    3. Generate Grad-CAM heatmap
    4. Build medical report
    5. Store results in session state
    """
    with st.spinner(""):
        progress_placeholder = st.empty()
        stages = [
            ("🔬 Loading and preprocessing image...", 0.15),
            ("🧠 Running deep learning analysis...", 0.50),
            ("🌡️  Generating Grad-CAM heatmap...", 0.75),
            ("📋 Building medical report...", 0.90),
            ("✅ Analysis complete!", 1.0),
        ]

        progress_bar = st.progress(0)

        for msg, prog in stages:
            progress_placeholder.markdown(
                f'<p style="color:#94a3b8; font-size:0.85rem; font-style:italic;">{msg}</p>',
                unsafe_allow_html=True
            )
            progress_bar.progress(prog)
            time.sleep(0.4)

    progress_placeholder.empty()
    progress_bar.empty()

    # Load image
    img_rgb = load_image(uploaded_file)
    uploaded_file.seek(0)

    # Resize for display (keep original for overlay)
    display_img = cv2.resize(img_rgb, (224, 224))

    # Convert OpenCV image to PIL for CLIP
    pil_img = Image.fromarray(img_rgb)
    
    # Run ZERO-SHOT prediction (REAL AI Inference)
    prediction = zero_shot_predict(pil_img, image_type, model, processor)
    
    # Generate mock heatmap for demo visualization since extracting ViT attention maps is computationally expensive here
    if image_type == "ECG/EKG Pattern":
        cam = demo_gradcam_ecg(img_rgb)
    elif image_type == "Other Medical Record":
        cam = demo_gradcam_record(img_rgb)
    elif image_type == "CT Scan":
        cam = demo_gradcam_ct(img_rgb)
    elif image_type == "Echocardiogram":
        cam = demo_gradcam_echo(img_rgb)
    else:
        cam = demo_gradcam(img_rgb)

    # Overlay heatmap
    cam_overlay = overlay_heatmap(display_img, cam, alpha=0.4)

    # Build full report
    report = format_full_report(prediction, patient_data)

    # Store in session state
    st.session_state.original_img = display_img
    st.session_state.cam_overlay = cam_overlay
    st.session_state.report = report
    st.session_state.analysis_done = True
    st.session_state.image_type = image_type


# ─── Results Rendering ────────────────────────────────────────────────────────

def render_results():
    report = st.session_state.report
    if report is None:
        return

    is_positive = report["is_positive"]
    card_class = "result-card-positive" if is_positive else "result-card-negative"
    badge_class = "prediction-badge-positive" if is_positive else "prediction-badge-negative"

    # ── Main Result Card
    st.markdown(f'<div class="{card_class}">', unsafe_allow_html=True)

    # Top row: badge + risk
    col_badge, col_risk = st.columns([1, 1])
    with col_badge:
        st.markdown(f"""
        <div class="section-label">Prediction</div>
        <div class="{badge_class}">{report['prediction']}</div>
        """, unsafe_allow_html=True)
    with col_risk:
        st.markdown('<div class="section-label">Risk Level</div>', unsafe_allow_html=True)
        render_risk_badge(report["risk"])

    st.markdown("&nbsp;", unsafe_allow_html=True)

    # Confidence bar
    render_confidence_bar(report["confidence_pct"], is_positive)

    st.markdown("</div>", unsafe_allow_html=True)

    # ── Demo mode warning
    if report.get("demo_mode"):
        render_demo_banner()

    # ── Detected Conditions
    if is_positive:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-label">Likely Cardiac Conditions</div>', unsafe_allow_html=True)
        render_condition_chips(report["all_conditions"])
        st.markdown("</div>", unsafe_allow_html=True)

    # ── Medical Advice Tabs
    advice = report["advice"]

    tab_symptoms, tab_lifestyle, tab_medical, tab_diet, tab_precautions = st.tabs([
        "🚨 Watch For",
        "🏃 Lifestyle",
        "💊 Medical Actions",
        "🥗 Diet",
        "⚠️ Precautions",
    ])

    with tab_symptoms:
        st.markdown('<div class="section-label">Symptoms to Monitor</div>', unsafe_allow_html=True)
        render_advice_list(advice.symptoms_to_watch, "#ef4444")
        st.markdown(f"""
        <p style="margin-top:1rem; font-size:0.82rem; color: #64748b; font-style: italic;">
        🕐 Urgency: <strong style="color:#fbbf24;">{advice.urgency}</strong>
        </p>""", unsafe_allow_html=True)

    with tab_lifestyle:
        st.markdown('<div class="section-label">Lifestyle Recommendations</div>', unsafe_allow_html=True)
        render_advice_list(advice.lifestyle_advice, "#6366f1")

    with tab_medical:
        st.markdown('<div class="section-label">Recommended Medical Actions</div>', unsafe_allow_html=True)
        render_advice_list(advice.medical_actions, "#3b82f6")

    with tab_diet:
        st.markdown('<div class="section-label">Dietary Guidelines</div>', unsafe_allow_html=True)
        render_advice_list(advice.dietary_advice, "#10b981")

    with tab_precautions:
        st.markdown('<div class="section-label">Important Precautions</div>', unsafe_allow_html=True)
        render_advice_list(advice.precautions, "#f59e0b")
        st.markdown(f"""
        <p style="margin-top:1rem; font-size:0.82rem; color: #64748b;">
        📅 Follow-up: {advice.follow_up}
        </p>""", unsafe_allow_html=True)

    # ── Disclaimer
    st.markdown(f"""
    <div style="margin-top:1.5rem; background: rgba(99,102,241,0.06); border: 1px solid
         rgba(99,102,241,0.2); border-radius:10px; padding: 0.8rem 1rem;
         font-size: 0.78rem; color: #94a3b8; line-height: 1.6;">
        {advice.disclaimer}
    </div>
    """, unsafe_allow_html=True)


# ─── App Layout ───────────────────────────────────────────────────────────────

def main():
    render_header()

    # Load model
    model, processor = get_model()

    # ── Layout: Left panel (upload) | Right panel (results)
    left_col, spacer, right_col = st.columns([1.1, 0.05, 1.85])

    with left_col:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("""
        <div class="section-label">Upload Medical Image</div>
        <p style="font-size:0.85rem; color:#94a3b8; margin-bottom:1rem;">
        Upload a frontal chest X-ray, an ECG/EKG pattern, or a general medical record in JPG, PNG, or DICOM format.
        All processing is done locally — no data is stored.
        </p>
        """, unsafe_allow_html=True)

        # Let user select the image type
        selected_image_type = st.selectbox(
            "Select Image Type:",
            ["Chest X-Ray", "CT Scan", "Echocardiogram", "ECG/EKG Pattern", "Other Medical Record"],
            index=0
        )

        uploaded_file = st.file_uploader(
            "Drop your image here",
            type=["jpg", "jpeg", "png", "dcm"],
            label_visibility="collapsed",
        )

        if uploaded_file is not None:
            # Preview
            try:
                preview_img = load_image(uploaded_file)
                uploaded_file.seek(0)
                preview_pil = Image.fromarray(cv2.resize(preview_img, (280, 280)))
                st.image(preview_pil, caption="Uploaded Image", use_column_width=True)

                file_size = len(uploaded_file.getvalue()) / 1024
                st.markdown(f"""
                <p style="font-size:0.75rem; color:#64748b; margin-top:0.4rem;">
                📁 {uploaded_file.name} &nbsp;·&nbsp; {file_size:.0f} KB
                </p>""", unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Cannot preview image: {e}")

        st.markdown("</div>", unsafe_allow_html=True)

        # Patient Vitals (Optional)
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-label">Patient Vitals (Optional)</div>', unsafe_allow_html=True)
        v_col1, v_col2 = st.columns(2)
        with v_col1:
            p_age = st.number_input("Age (yrs)", min_value=0, max_value=120, value=0, step=1)
            p_bp_sys = st.number_input("Systolic BP", min_value=0, max_value=300, value=120, step=1)
            p_fat = st.number_input("Body Fat %", min_value=0.0, max_value=70.0, value=0.0, step=0.5)
        with v_col2:
            p_sugar = st.selectbox("Blood Sugar", ["Unknown", "Normal", "Pre-diabetic", "Diabetic"])
            p_bp_dia = st.number_input("Diastolic BP", min_value=0, max_value=200, value=80, step=1)

        patient_data = {
            "age": p_age if p_age > 0 else None,
            "bp_sys": p_bp_sys,
            "bp_dia": p_bp_dia,
            "sugar": p_sugar,
            "body_fat": p_fat if p_fat > 0 else None
        }
        st.markdown("</div>", unsafe_allow_html=True)

        # Analyze button
        analyze_disabled = uploaded_file is None
        if st.button("🔬 Analyze Health Record", disabled=analyze_disabled):
            if uploaded_file is not None and model is not None:
                run_analysis(uploaded_file, model, processor, selected_image_type, patient_data)
                st.rerun()

        # Info section
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("""
        <div class="section-label">About This AI Model</div>
        <div style="font-size:0.82rem; color:#94a3b8; line-height:1.8;">
        <div style="display:flex; align-items:center; gap:0.5rem; margin-bottom:0.4rem;">
            <span style="color:#6366f1;">◆</span> HuggingFace CLIP Vision Transformer
        </div>
        <div style="display:flex; align-items:center; gap:0.5rem; margin-bottom:0.4rem;">
            <span style="color:#6366f1;">◆</span> Zero-Shot multi-modal classification
        </div>
        <div style="display:flex; align-items:center; gap:0.5rem; margin-bottom:0.4rem;">
            <span style="color:#6366f1;">◆</span> Real-time cross-image type analysis
        </div>
        <div style="display:flex; align-items:center; gap:0.5rem; margin-bottom:0.4rem;">
            <span style="color:#6366f1;">◆</span> Pre-trained on 400M text/image pairs
        </div>
        <div style="display:flex; align-items:center; gap:0.5rem;">
            <span style="color:#6366f1;">◆</span> Zero data retention policy
        </div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with right_col:
        if st.session_state.analysis_done and st.session_state.report is not None:

            # Image comparison tabs
            img_tab1, img_tab2 = st.tabs(["🔬 Grad-CAM Analysis", "📷 Original X-ray"])

            with img_tab1:
                if st.session_state.cam_overlay is not None:
                    overlay_pil = Image.fromarray(st.session_state.cam_overlay)
                    st.image(overlay_pil, caption="Grad-CAM Heatmap — Red regions = high model attention",
                             use_column_width=True)
                    st.markdown("""
                    <p style="font-size:0.75rem; color:#64748b; line-height:1.5;">
                    🔴 Red = regions most influential to prediction &nbsp;·&nbsp;
                    🔵 Blue = less influential &nbsp;·&nbsp;
                    Generated using gradient-weighted class activation mapping
                    </p>""", unsafe_allow_html=True)

            with img_tab2:
                if st.session_state.original_img is not None:
                    original_pil = Image.fromarray(st.session_state.original_img)
                    st.image(original_pil, caption="Original Medical Image (preprocessed)",
                             use_column_width=True)

            st.markdown("<br>", unsafe_allow_html=True)
            render_results()

            # Reset button
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("↺ Analyze Another Image"):
                st.session_state.analysis_done = False
                st.session_state.report = None
                st.session_state.cam_overlay = None
                st.session_state.original_img = None
                st.rerun()

        else:
            # Placeholder state
            st.markdown("""
            <div style="
                height: 100%;
                min-height: 500px;
                display: flex;
                flex-direction: column;
                align-items: center;
                justify-content: center;
                color: #334155;
                text-align: center;
                border: 1px dashed rgba(255,255,255,0.06);
                border-radius: 16px;
                padding: 3rem;
            ">
                <div style="font-size: 5rem; margin-bottom: 1rem; opacity:0.3;">🫀</div>
                <div style="font-size: 1.1rem; font-weight: 600; color: #475569; margin-bottom: 0.5rem;">
                    No Analysis Yet
                </div>
                <div style="font-size: 0.85rem; color: #334155; max-width: 300px; line-height: 1.6;">
                    Upload a medical image and click <strong style="color:#64748b;">Analyze Health Record</strong>
                    to see the AI prediction and Grad-CAM heatmap here.
                </div>
                <div style="margin-top:2rem; display:flex; gap:2rem; color:#334155; font-size:0.75rem;">
                    <div style="text-align:center;">
                        <div style="font-size:1.5rem; color:#475569;">📊</div>
                        <div>Confidence Score</div>
                    </div>
                    <div style="text-align:center;">
                        <div style="font-size:1.5rem; color:#475569;">🗺️</div>
                        <div>Grad-CAM Map</div>
                    </div>
                    <div style="text-align:center;">
                        <div style="font-size:1.5rem; color:#475569;">📋</div>
                        <div>Medical Report</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    # ── Footer
    st.markdown("""
    <hr>
    <div style="display:flex; justify-content:space-between; align-items:center;
                padding: 0.5rem 0 1rem; font-size: 0.72rem; color: #334155;">
        <span>CardioScan AI &nbsp;·&nbsp; DenseNet121 + Grad-CAM &nbsp;·&nbsp; MIT License</span>
        <span>⚠️ Not for clinical use &nbsp;·&nbsp; For research and education only</span>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
