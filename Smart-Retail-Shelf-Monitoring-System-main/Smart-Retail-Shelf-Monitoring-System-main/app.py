import streamlit as st
from pathlib import Path
from PIL import Image
import sys
import os
import io
import time
from datetime import datetime
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Shelvex AI — Enterprise Retail Shelf Monitoring",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# PATH CONFIGURATION
# ============================================================

CURRENT_FILE = Path(__file__).resolve()
BASE_DIR = CURRENT_FILE.parent

# Support running from root workspace or nested directory
if (BASE_DIR / "src").exists():
    PROJECT_ROOT = BASE_DIR
elif (BASE_DIR / "Smart-Retail-Shelf-Monitoring-System-main" / "src").exists():
    PROJECT_ROOT = BASE_DIR / "Smart-Retail-Shelf-Monitoring-System-main"
else:
    PROJECT_ROOT = BASE_DIR

SRC_DIR = PROJECT_ROOT / "src"
SAMPLES_DIR = PROJECT_ROOT / "samples"
if not SAMPLES_DIR.exists() and (BASE_DIR.parent / "samples").exists():
    SAMPLES_DIR = BASE_DIR.parent / "samples"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# ============================================================
# IMPORT AI ENGINE & VISUALIZER
# ============================================================

from predict import run_all_models
from shelf_analysis import analyze_shelf
from visualizer import render_annotated_shelf

# ============================================================
# MODERN ENTERPRISE CSS
# ============================================================

st.markdown("""
<style>
    /* Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Top Brand Bar */
    .brand-container {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 0.8rem 1.4rem;
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.7) 0%, rgba(15, 23, 42, 0.85) 100%);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        margin-bottom: 1.5rem;
        backdrop-filter: blur(12px);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.12);
    }
    
    .brand-left {
        display: flex;
        align-items: center;
        gap: 14px;
    }

    .brand-logo {
        font-size: 2.2rem;
        filter: drop-shadow(0 2px 8px rgba(56, 189, 248, 0.4));
    }

    .brand-title {
        font-size: 1.55rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        background: linear-gradient(135deg, #38BDF8 0%, #818CF8 50%, #C084FC 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
        line-height: 1.2;
    }

    .brand-subtitle {
        font-size: 0.82rem;
        color: #94A3B8;
        font-weight: 500;
        margin: 0;
    }

    .badge-pill {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.78rem;
        font-weight: 600;
        border: 1px solid rgba(255, 255, 255, 0.1);
        background: rgba(255, 255, 255, 0.05);
    }

    .badge-online {
        background: rgba(16, 185, 129, 0.12);
        color: #34D399;
        border-color: rgba(16, 185, 129, 0.3);
    }

    .badge-location {
        background: rgba(56, 189, 248, 0.12);
        color: #38BDF8;
        border-color: rgba(56, 189, 248, 0.3);
    }

    /* KPI Stat Cards */
    .kpi-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 16px;
        margin-bottom: 1.5rem;
    }

    .kpi-card {
        background: linear-gradient(145deg, rgba(30, 41, 59, 0.6) 0%, rgba(15, 23, 42, 0.75) 100%);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px;
        padding: 1.1rem 1.2rem;
        position: relative;
        overflow: hidden;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }

    .kpi-card:hover {
        transform: translateY(-2px);
        border-color: rgba(255, 255, 255, 0.18);
    }

    .kpi-card-emerald { border-top: 3px solid #10B981; }
    .kpi-card-sky { border-top: 3px solid #0EA5E9; }
    .kpi-card-purple { border-top: 3px solid #8B5CF6; }
    .kpi-card-rose { border-top: 3px solid #F43F5E; }

    .kpi-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 8px;
    }

    .kpi-title {
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        font-weight: 600;
        color: #94A3B8;
        margin: 0;
    }

    .kpi-icon {
        font-size: 1.25rem;
        opacity: 0.85;
    }

    .kpi-value {
        font-size: 1.85rem;
        font-weight: 800;
        color: #F8FAFC;
        margin: 0;
        line-height: 1.1;
    }

    .kpi-meta {
        font-size: 0.76rem;
        color: #64748B;
        margin-top: 6px;
        font-weight: 500;
    }

    /* Executive Alert Banner */
    .executive-banner {
        border-radius: 14px;
        padding: 1.1rem 1.4rem;
        margin-bottom: 1.5rem;
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 16px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
    }

    .banner-full {
        background: linear-gradient(135deg, rgba(6, 78, 59, 0.85) 0%, rgba(6, 95, 70, 0.7) 100%);
        border: 1px solid rgba(16, 185, 129, 0.4);
        color: #ECFDF5;
    }

    .banner-low {
        background: linear-gradient(135deg, rgba(120, 53, 15, 0.85) 0%, rgba(146, 64, 14, 0.7) 100%);
        border: 1px solid rgba(245, 158, 11, 0.4);
        color: #FEF3C7;
    }

    .banner-empty {
        background: linear-gradient(135deg, rgba(136, 19, 55, 0.85) 0%, rgba(159, 18, 57, 0.7) 100%);
        border: 1px solid rgba(244, 63, 94, 0.4);
        color: #FFE4E6;
    }

    .banner-title {
        font-size: 1.15rem;
        font-weight: 700;
        margin: 0 0 4px 0;
    }

    .banner-desc {
        font-size: 0.88rem;
        margin: 0;
        opacity: 0.9;
    }

    /* Preset Showcase Card */
    .preset-card {
        background: rgba(30, 41, 59, 0.5);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 12px;
        text-align: center;
        transition: all 0.2s ease;
    }

    .preset-card:hover {
        border-color: #38BDF8;
        background: rgba(30, 41, 59, 0.8);
    }

    /* Visualizer Toolbar */
    .studio-toolbar {
        background: rgba(15, 23, 42, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 10px 16px;
        margin-bottom: 12px;
    }

    /* Subheader enhancements */
    h3 {
        font-weight: 700 !important;
        letter-spacing: -0.01em !important;
    }

    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        border-bottom: 1px solid rgba(255, 255, 255, 0.08);
        padding-bottom: 4px;
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        padding: 8px 16px;
        font-weight: 600;
        font-size: 0.92rem;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# SESSION STATE INITIALIZATION
# ============================================================

if "analysis_result" not in st.session_state:
    st.session_state.analysis_result = None

if "uploaded_image" not in st.session_state:
    st.session_state.uploaded_image = None

if "current_image_name" not in st.session_state:
    st.session_state.current_image_name = "Untitled Shelf"

if "analysis_history" not in st.session_state:
    st.session_state.analysis_history = []

if "show_products" not in st.session_state:
    st.session_state.show_products = True

if "show_skus" not in st.session_state:
    st.session_state.show_skus = True

if "show_voids" not in st.session_state:
    st.session_state.show_voids = True

if "show_confidence" not in st.session_state:
    st.session_state.show_confidence = True

if "highlight_voids" not in st.session_state:
    st.session_state.highlight_voids = True

if "box_thickness" not in st.session_state:
    st.session_state.box_thickness = 2

if "target_layer_only" not in st.session_state:
    st.session_state.target_layer_only = "All"

if "store_id" not in st.session_state:
    st.session_state.store_id = "Store #104 (Downtown Flagship)"

if "aisle_id" not in st.session_state:
    st.session_state.aisle_id = "Aisle 4B — Beverages & Snacks"

if "inference_time_ms" not in st.session_state:
    st.session_state.inference_time_ms = 0

# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    # Enterprise Branding Header
    st.markdown("""
    <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 6px;">
        <span style="font-size: 2rem;">🛒</span>
        <div>
            <h2 style="margin: 0; font-size: 1.35rem; font-weight: 800; background: linear-gradient(135deg, #38BDF8 0%, #818CF8 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">SHELVEX AI</h2>
            <p style="margin: 0; font-size: 0.72rem; color: #94A3B8; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em;">Enterprise Shelf Vision</p>
        </div>
    </div>
    <div style="display: flex; gap: 6px; margin-bottom: 14px;">
        <span class="badge-pill badge-online">● 3 Models Active</span>
        <span class="badge-pill" style="color: #CBD5E1;">v2.4 LTS</span>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # Audit Control Actions
    col_new, col_clear = st.columns(2)
    with col_new:
        if st.button("＋ New Audit", use_container_width=True, type="primary"):
            st.session_state.analysis_result = None
            st.session_state.uploaded_image = None
            st.session_state.current_image_name = "Untitled Shelf"
            st.rerun()

    with col_clear:
        if st.button("🗑️ Clear Log", use_container_width=True):
            st.session_state.analysis_history = []
            st.rerun()

    # Store Location Metadata
    with st.expander("📍 Store & Aisle Context", expanded=False):
        st.session_state.store_id = st.text_input("Store Location", value=st.session_state.store_id)
        st.session_state.aisle_id = st.text_input("Aisle / Bay Section", value=st.session_state.aisle_id)
        st.caption("Metadata will be attached to generated audit reports.")

    # Detection Sensitivity Controls
    with st.expander("⚙️ Detection Sensitivity", expanded=True):
        st.markdown("<p style='font-size:0.8rem; color:#94A3B8; margin-bottom:8px;'>Adjust model threshold filters in real time:</p>", unsafe_allow_html=True)
        
        # Sensitivity Preset Selector
        preset_choice = st.radio(
            "Confidence Preset",
            ["Balanced", "High Precision", "High Recall"],
            horizontal=True,
            label_visibility="collapsed"
        )
        
        if preset_choice == "High Precision":
            def_prod, def_sku, def_void = 0.40, 0.35, 0.40
        elif preset_choice == "High Recall":
            def_prod, def_sku, def_void = 0.15, 0.15, 0.15
        else:
            def_prod, def_sku, def_void = 0.25, 0.20, 0.25

        product_conf = st.slider("Product Confidence", 0.05, 0.95, def_prod, 0.05, help="Filter for product presence")
        sku_conf = st.slider("SKU Confidence", 0.05, 0.95, def_sku, 0.05, help="Filter for specific SKU classification")
        void_conf = st.slider("Void Gap Confidence", 0.05, 0.95, def_void, 0.05, help="Filter for out-of-stock empty spaces")

    # Recent Audits Session Log
    st.markdown("<h4 style='font-size:0.92rem; font-weight:700; color:#E2E8F0; margin-top:16px; margin-bottom:8px;'>📋 Session Audit Log</h4>", unsafe_allow_html=True)
    if st.session_state.analysis_history:
        for idx, item in enumerate(reversed(st.session_state.analysis_history[-5:])):
            status_color = "#34D399" if item["status"] == "FULL" else ("#FBBF24" if item["status"] == "LOW" else "#F87171")
            with st.container():
                st.markdown(f"""
                <div style="background: rgba(30, 41, 59, 0.4); border: 1px solid rgba(255,255,255,0.06); border-radius: 8px; padding: 8px 10px; margin-bottom: 6px;">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <span style="font-weight: 600; font-size: 0.82rem; color: #F1F5F9;">{item['name']}</span>
                        <span style="font-size: 0.72rem; color: {status_color}; font-weight: 700;">● {item['status']}</span>
                    </div>
                    <div style="font-size: 0.72rem; color: #94A3B8; margin-top: 3px;">
                        {item['timestamp']} • {item['products']} Products • {item['empty']} Voids
                    </div>
                </div>
                """, unsafe_allow_html=True)
    else:
        st.markdown("<p style='font-size: 0.78rem; color: #64748B;'>No audits executed in this session yet.</p>", unsafe_allow_html=True)

    st.divider()

    # System Architecture Info
    with st.expander("🤖 Neural Architecture Details", expanded=False):
        st.markdown("""
        **Pipeline Stack:**
        - **Product Detection:** `product_best.pt` (YOLOv8)
        - **SKU Categorization:** `sku_best.pt` (148 Retail SKUs)
        - **Void Space Detection:** `void_best.pt` (Shelf Gap Detection)
        - **Inference Resolution:** `640 x 640 px`
        - **Acceleration:** Torch CUDA / Optimized CPU
        """)

# ============================================================
# TOP EXECUTIVE BRAND HEADER
# ============================================================

current_status_badge = '<span class="badge-pill badge-online">● System Ready</span>'
if st.session_state.analysis_result is not None:
    status_label = st.session_state.analysis_result["analysis"]["summary"]["overall_status"]
    if status_label == "FULL":
        current_status_badge = '<span class="badge-pill" style="background: rgba(16,185,129,0.15); color:#34D399; border-color: rgba(16,185,129,0.3);">● Optimal Stock (100% Planogram)</span>'
    elif status_label == "LOW":
        current_status_badge = '<span class="badge-pill" style="background: rgba(245,158,11,0.15); color:#FBBF24; border-color: rgba(245,158,11,0.3);">⚠️ Low Stock Warning</span>'
    else:
        current_status_badge = '<span class="badge-pill" style="background: rgba(244,63,94,0.15); color:#F87171; border-color: rgba(244,63,94,0.3);">🚨 Critical Stockout Detected</span>'

st.markdown(f"""
<div class="brand-container">
    <div class="brand-left">
        <div class="brand-logo">🛒</div>
        <div>
            <h1 class="brand-title">SHELVEX ENTERPRISE</h1>
            <p class="brand-subtitle">Autonomous Shelf Health, Planogram Verification & Real-Time Restock Intelligence</p>
        </div>
    </div>
    <div style="display: flex; gap: 8px; align-items: center;">
        <span class="badge-pill badge-location">📍 {st.session_state.store_id}</span>
        {current_status_badge}
    </div>
</div>
""", unsafe_allow_html=True)

# ============================================================
# INPUT HUB: DEMO PRESETS / FILE UPLOAD / CAMERA
# ============================================================

input_image_to_process = None
run_inference_trigger = False

# Only show the input ingestion panel if no analysis is present or when user requests new upload
if st.session_state.analysis_result is None:
    st.markdown("<h3 style='margin-bottom: 4px;'>📸 Ingest Shelf Imagery</h3>", unsafe_allow_html=True)
    st.markdown("<p style='color: #94A3B8; font-size: 0.88rem; margin-bottom: 16px;'>Select a pre-loaded retail shelf scenario, upload your own store photo, or capture directly using a mobile tablet camera.</p>", unsafe_allow_html=True)

    input_tab1, input_tab2, input_tab3 = st.tabs([
        "⚡ 1-Click Demo Scenarios",
        "📁 Upload Image File",
        "📷 Live Camera Audit"
    ])

    # Tab 1: Instant Demo Scenarios
    with input_tab1:
        st.markdown("<p style='font-size: 0.82rem; color: #94A3B8;'>Click any scenario below to immediately load and evaluate the multi-model vision pipeline:</p>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)

        sample1_path = SAMPLES_DIR / "sample_shelf_1.jpg"
        sample2_path = SAMPLES_DIR / "sample_shelf_2.jpg"
        sample3_path = SAMPLES_DIR / "sample_shelf_3.jpg"

        with col1:
            if sample1_path.exists():
                st.image(str(sample1_path), use_container_width=True)
            st.markdown("**Scenario A: Beverage Cooler**")
            st.caption("Bottled beverages with isolated gap voids.")
            if st.button("Load Scenario A", key="btn_sample_1", use_container_width=True):
                if sample1_path.exists():
                    st.session_state.uploaded_image = Image.open(sample1_path).convert("RGB")
                    st.session_state.current_image_name = "Beverage Cooler Shelf"
                    run_inference_trigger = True

        with col2:
            if sample2_path.exists():
                st.image(str(sample2_path), use_container_width=True)
            st.markdown("**Scenario B: Snack Gondola**")
            st.caption("Dense multi-facing retail snack display.")
            if st.button("Load Scenario B", key="btn_sample_2", use_container_width=True):
                if sample2_path.exists():
                    st.session_state.uploaded_image = Image.open(sample2_path).convert("RGB")
                    st.session_state.current_image_name = "Snack Gondola"
                    run_inference_trigger = True

        with col3:
            if sample3_path.exists():
                st.image(str(sample3_path), use_container_width=True)
            st.markdown("**Scenario C: Packaged Goods**")
            st.caption("Multi-tier shelf with visible restock voids.")
            if st.button("Load Scenario C", key="btn_sample_3", use_container_width=True):
                if sample3_path.exists():
                    st.session_state.uploaded_image = Image.open(sample3_path).convert("RGB")
                    st.session_state.current_image_name = "Packaged Goods Shelf"
                    run_inference_trigger = True

    # Tab 2: File Upload
    with input_tab2:
        uploaded_file = st.file_uploader(
            "Choose a retail shelf photograph (JPG, PNG, WEBP)",
            type=["jpg", "jpeg", "png", "webp"],
            key="file_uploader"
        )
        if uploaded_file is not None:
            st.session_state.uploaded_image = Image.open(uploaded_file).convert("RGB")
            st.session_state.current_image_name = uploaded_file.name
            st.success(f"Image `{uploaded_file.name}` loaded successfully.")

    # Tab 3: Camera Capture
    with input_tab3:
        camera_file = st.camera_input("Take a photo of the retail shelf bay")
        if camera_file is not None:
            st.session_state.uploaded_image = Image.open(camera_file).convert("RGB")
            st.session_state.current_image_name = f"Camera_Capture_{datetime.now().strftime('%H%M%S')}"

    # Analyze Button if image is loaded but not yet processed
    if st.session_state.uploaded_image is not None and not run_inference_trigger:
        st.markdown("<div style='margin-top: 20px;'></div>", unsafe_allow_html=True)
        col_preview, col_action = st.columns([1, 2], vertical_alignment="center")
        with col_preview:
            st.image(st.session_state.uploaded_image, caption=st.session_state.current_image_name, width=280)
        with col_action:
            st.markdown(f"#### Ready to audit: **{st.session_state.current_image_name}**")
            st.markdown(f"**Target Location:** {st.session_state.aisle_id}")
            if st.button("🚀 Run Neural Shelf Inspection", type="primary", use_container_width=True):
                run_inference_trigger = True

# ============================================================
# RUN INFERENCE PIPELINE
# ============================================================

if run_inference_trigger and st.session_state.uploaded_image is not None:
    target_img = st.session_state.uploaded_image
    
    with st.spinner("🤖 Running YOLO models (Product, SKU, Void)..."):
        t_start = time.time()
        try:
            results = run_all_models(
                target_img,
                product_conf=product_conf,
                sku_conf=sku_conf,
                void_conf=void_conf
            )
            analysis = analyze_shelf(results)
            t_elapsed_ms = int((time.time() - t_start) * 1000)
            
            st.session_state.inference_time_ms = t_elapsed_ms
            st.session_state.analysis_result = {
                "results": results,
                "analysis": analysis
            }

            # Append to history
            summary = analysis["summary"]
            st.session_state.analysis_history.append({
                "timestamp": datetime.now().strftime("%H:%M:%S"),
                "name": st.session_state.current_image_name,
                "products": summary["total_products"],
                "empty": summary["empty_spaces"],
                "status": summary["overall_status"]
            })
            st.rerun()

        except Exception as e:
            st.error("Inference pipeline encountered an error.")
            st.exception(e)

# ============================================================
# RESULTS DASHBOARD
# ============================================================

if st.session_state.analysis_result is not None and st.session_state.uploaded_image is not None:
    results = st.session_state.analysis_result["results"]
    analysis = st.session_state.analysis_result["analysis"]
    summary = analysis["summary"]
    
    total_products = summary["total_products"]
    empty_spaces = summary["empty_spaces"]
    unique_skus = summary["unique_skus"]
    unique_types = summary["unique_product_types"]
    overall_status = summary["overall_status"]
    
    # Calculate derived retail metrics
    total_facings = total_products + empty_spaces
    fill_rate = (total_products / total_facings * 100) if total_facings > 0 else 0.0
    void_rate = (empty_spaces / total_facings * 100) if total_facings > 0 else 0.0
    
    # Stock health index (0 to 100)
    health_score = int(min(100, max(0, fill_rate)))

    # Executive Status Banner
    if overall_status == "FULL":
        banner_class = "banner-full"
        banner_icon = "🟢"
        banner_title = "OPTIMAL SHELF HEALTH — FULLY STOCKED"
        banner_text = f"Shelf compliance is at <b>{fill_rate:.1f}%</b>. All product facings conform to planogram capacity. No critical replenishments required."
    elif overall_status == "LOW":
        banner_class = "banner-low"
        banner_icon = "🟡"
        banner_title = "WARNING: RESTOCK REQUIRED"
        banner_text = f"Shelf capacity at <b>{fill_rate:.1f}%</b>. {empty_spaces} empty shelf space(s) detected. Restocking task dispatched to floor associates."
    else:
        banner_class = "banner-empty"
        banner_icon = "🔴"
        banner_title = "CRITICAL OUT-OF-STOCK ALERT"
        banner_text = f"Shelf capacity is critically low at <b>{fill_rate:.1f}%</b>. {empty_spaces} empty positions require immediate replenishment from backroom inventory."

    st.markdown(f"""
    <div class="executive-banner {banner_class}">
        <div>
            <h3 class="banner-title">{banner_icon} {banner_title}</h3>
            <p class="banner-desc">{banner_text}</p>
        </div>
        <div style="text-align: right; min-width: 140px;">
            <span style="font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.05em; opacity: 0.85;">Shelf Health Index</span>
            <div style="font-size: 2rem; font-weight: 800; line-height: 1;">{health_score}%</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # 4 KPI Stat Cards
    kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)
    with kpi_col1:
        st.markdown(f"""
        <div class="kpi-card kpi-card-sky">
            <div class="kpi-header">
                <span class="kpi-title">Detected Products</span>
                <span class="kpi-icon">📦</span>
            </div>
            <div class="kpi-value">{total_products}</div>
            <div class="kpi-meta">{unique_types} distinct category facing(s)</div>
        </div>
        """, unsafe_allow_html=True)

    with kpi_col2:
        st.markdown(f"""
        <div class="kpi-card kpi-card-purple">
            <div class="kpi-header">
                <span class="kpi-title">Identified SKUs</span>
                <span class="kpi-icon">🏷️</span>
            </div>
            <div class="kpi-value">{unique_skus}</div>
            <div class="kpi-meta">{summary['total_sku_detections']} SKU tag detection(s)</div>
        </div>
        """, unsafe_allow_html=True)

    with kpi_col3:
        st.markdown(f"""
        <div class="kpi-card kpi-card-rose">
            <div class="kpi-header">
                <span class="kpi-title">Empty Void Gaps</span>
                <span class="kpi-icon">🕳️</span>
            </div>
            <div class="kpi-value">{empty_spaces}</div>
            <div class="kpi-meta">{void_rate:.1f}% of total facings are vacant</div>
        </div>
        """, unsafe_allow_html=True)

    with kpi_col4:
        st.markdown(f"""
        <div class="kpi-card kpi-card-emerald">
            <div class="kpi-header">
                <span class="kpi-title">Fill Availability</span>
                <span class="kpi-icon">📈</span>
            </div>
            <div class="kpi-value">{fill_rate:.1f}%</div>
            <div class="kpi-meta">Latency: {st.session_state.inference_time_ms} ms (YOLOv8)</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='margin-top: 14px;'></div>", unsafe_allow_html=True)

    # Main Tabbed Operations Dashboard
    tab_vision, tab_analytics, tab_ops, tab_report = st.tabs([
        "👁️ Vision Intelligence Studio",
        "📊 Stock & Planogram Analytics",
        "📋 Restock Action Checklist",
        "🏢 Store Manager Audit Report"
    ])

    # --------------------------------------------------------
    # TAB 1: VISION INTELLIGENCE STUDIO
    # --------------------------------------------------------
    with tab_vision:
        st.markdown("#### 🔬 Neural Vision Studio & Layer Inspector")
        st.markdown("<p style='font-size:0.84rem; color:#94A3B8; margin-bottom:12px;'>Inspect multi-model detections with interactive layer filters, confidence badges, and translucent void overlays.</p>", unsafe_allow_html=True)

        # Visualizer Control Toolbar
        toolbar_c1, toolbar_c2, toolbar_c3, toolbar_c4, toolbar_c5 = st.columns([2, 2, 2, 2, 2])
        
        with toolbar_c1:
            st.session_state.show_products = st.checkbox("📦 Products Layer", value=st.session_state.show_products)
        with toolbar_c2:
            st.session_state.show_skus = st.checkbox("🏷️ SKUs Layer", value=st.session_state.show_skus)
        with toolbar_c3:
            st.session_state.show_voids = st.checkbox("🕳️ Void Gaps Layer", value=st.session_state.show_voids)
        with toolbar_c4:
            st.session_state.highlight_voids = st.checkbox("Highlight Voids", value=st.session_state.highlight_voids, help="Fill empty void gaps with translucent red tint")
        with toolbar_c5:
            st.session_state.show_confidence = st.checkbox("Show % Badges", value=st.session_state.show_confidence)

        # Layer focus filter
        fcol1, fcol2 = st.columns([2, 4])
        with fcol1:
            st.session_state.target_layer_only = st.selectbox(
                "Filter Active Layer",
                ["All", "Voids", "Products", "SKUs"],
                index=["All", "Voids", "Products", "SKUs"].index(st.session_state.target_layer_only)
            )
        with fcol2:
            st.session_state.box_thickness = st.slider("Bounding Box Thickness", 1, 4, st.session_state.box_thickness)

        # Generate annotated image
        annotated_image = render_annotated_shelf(
            image=st.session_state.uploaded_image,
            results=results,
            show_products=st.session_state.show_products,
            show_skus=st.session_state.show_skus,
            show_voids=st.session_state.show_voids,
            box_thickness=st.session_state.box_thickness,
            show_confidence=st.session_state.show_confidence,
            highlight_voids=st.session_state.highlight_voids,
            target_layer_only=st.session_state.target_layer_only
        )

        # Dual Image View
        img_col_left, img_col_right = st.columns(2, gap="medium")
        
        with img_col_left:
            st.markdown("<p style='font-weight:600; font-size:0.88rem; color:#CBD5E1;'>AI Annotated Detections (Multi-Model)</p>", unsafe_allow_html=True)
            if annotated_image is not None:
                st.image(annotated_image, use_container_width=True)
                
                # Download annotated image button
                buf = io.BytesIO()
                annotated_image.save(buf, format="PNG")
                byte_im = buf.getvalue()
                st.download_button(
                    label="📥 Download Annotated Frame (PNG)",
                    data=byte_im,
                    file_name=f"Shelvex_{st.session_state.current_image_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                    mime="image/png",
                    use_container_width=True
                )

        with img_col_right:
            st.markdown("<p style='font-weight:600; font-size:0.88rem; color:#CBD5E1;'>Raw Optical Capture (Store Floor)</p>", unsafe_allow_html=True)
            st.image(st.session_state.uploaded_image, use_container_width=True)

    # --------------------------------------------------------
    # TAB 2: STOCK & PLANOGRAM ANALYTICS (PLOTLY)
    # --------------------------------------------------------
    with tab_analytics:
        st.markdown("#### 📊 Planogram Capacity & Inventory Distribution")
        
        chart_col1, chart_col2 = st.columns(2)

        # Donut Chart: Space Allocation
        with chart_col1:
            labels = ['Stocked Products', 'Empty Void Gaps']
            values = [total_products, empty_spaces]
            colors = ['#0EA5E9', '#EF4444']

            fig_donut = go.Figure(data=[go.Pie(
                labels=labels,
                values=values,
                hole=.62,
                marker_colors=colors,
                textinfo='label+percent',
                insidetextorientation='radial'
            )])
            fig_donut.update_layout(
                title_text="Shelf Capacity Allocation",
                title_font_size=16,
                annotations=[dict(
                    text=f"<b>{fill_rate:.0f}%</b><br><span style='font-size:11px;'>Fill Rate</span>",
                    x=0.5, y=0.5,
                    font_size=18,
                    showarrow=False
                )],
                showlegend=False,
                margin=dict(t=40, b=20, l=20, r=20),
                height=320,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig_donut, use_container_width=True)

        # Bar Chart: SKU Breakdown
        with chart_col2:
            sku_analysis = analysis["skus"]["analysis"]
            sku_counts = sku_analysis.get("counts", {})

            if sku_counts:
                sku_names = [f"SKU #{k}" for k in sku_counts.keys()]
                sku_quantities = list(sku_counts.values())
                
                df_sku_chart = pd.DataFrame({
                    "SKU": sku_names,
                    "Count": sku_quantities
                }).sort_values(by="Count", ascending=True)

                fig_bar = px.bar(
                    df_sku_chart,
                    x="Count",
                    y="SKU",
                    orientation='h',
                    title="SKU Inventory On-Shelf Count",
                    color="Count",
                    color_continuous_scale="Purples"
                )
                fig_bar.update_layout(
                    margin=dict(t=40, b=20, l=20, r=20),
                    height=320,
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    coloraxis_showscale=False
                )
                st.plotly_chart(fig_bar, use_container_width=True)
            else:
                st.info("No specific SKU markers detected on this shelf.")

        # Product Class Breakdown
        st.divider()
        st.markdown("##### 📦 Product Class Distribution")
        prod_counts = analysis["products"].get("counts", {})
        if prod_counts:
            p_cols = st.columns(len(prod_counts) + 1)
            for i, (cid, ccount) in enumerate(prod_counts.items()):
                with p_cols[i]:
                    st.metric(f"Product Category {cid}", f"{ccount} units", f"{ccount / total_products * 100:.0f}% of shelf" if total_products > 0 else "")
            with p_cols[-1]:
                st.metric("Total Empty Voids", f"{empty_spaces} units", f"-{void_rate:.0f}% vacancy", delta_color="inverse")

    # --------------------------------------------------------
    # TAB 3: RESTOCK ACTION CHECKLIST & TABLE
    # --------------------------------------------------------
    with tab_ops:
        st.markdown("#### 📋 Prioritized Associate Restock Plan")
        st.markdown("<p style='font-size:0.84rem; color:#94A3B8;'>Actionable replenishment schedule generated by computer vision intelligence:</p>", unsafe_allow_html=True)

        inventory_rows = []

        # Product Rows
        prod_counts = analysis["products"].get("counts", {})
        for class_id, count in prod_counts.items():
            if count >= 3:
                status = "🟢 FULL"
                urgency = "Low"
                action = "Adequate stock — maintain facing"
            elif count > 0:
                status = "🟡 LOW"
                urgency = "Medium"
                action = f"Replenish 4+ units from backroom"
            else:
                status = "🔴 OUT OF STOCK"
                urgency = "Critical"
                action = "Urgent: Expedite warehouse restock"

            inventory_rows.append({
                "Type": "Product Category",
                "Identifier": f"Product Class {class_id}",
                "Count": count,
                "Status": status,
                "Urgency": urgency,
                "Recommended Action": action
            })

        # SKU Rows
        sku_counts = analysis["skus"]["analysis"].get("counts", {})
        sku_status_data = analysis["skus"].get("status", {})

        for sku_id, count in sku_counts.items():
            sku_info = sku_status_data.get(sku_id, {})
            sku_status = sku_info.get("status", "FULL" if count >= 3 else "LOW")
            
            if sku_status == "FULL":
                status = "🟢 FULL"
                urgency = "Low"
                action = "Facing aligned — stock optimal"
            elif sku_status == "LOW":
                status = "🟡 LOW"
                urgency = "High"
                action = f"Replenish SKU #{sku_id} facing (+3 units)"
            else:
                status = "🔴 OUT OF STOCK"
                urgency = "Critical"
                action = f"Empty SKU #{sku_id} position — restock immediately"

            inventory_rows.append({
                "Type": "SKU Item",
                "Identifier": f"SKU #{sku_id}",
                "Count": count,
                "Status": status,
                "Urgency": urgency,
                "Recommended Action": action
            })

        # Void row if empty spaces exist
        if empty_spaces > 0:
            inventory_rows.append({
                "Type": "Void Space",
                "Identifier": "Unoccupied Shelf Slots",
                "Count": empty_spaces,
                "Status": "🔴 EMPTY",
                "Urgency": "Critical",
                "Recommended Action": f"Replenish {empty_spaces} empty shelf slot(s)"
            })

        if inventory_rows:
            inv_df = pd.DataFrame(inventory_rows)

            # Filter controls
            filter_col1, filter_col2 = st.columns([2, 4])
            with filter_col1:
                filter_status = st.selectbox("Filter Status", ["All Items", "Urgent / Low Stock", "Optimal Only"])
            
            if filter_status == "Urgent / Low Stock":
                inv_df = inv_df[inv_df["Urgency"].isin(["Medium", "High", "Critical"])]
            elif filter_status == "Optimal Only":
                inv_df = inv_df[inv_df["Urgency"] == "Low"]

            st.dataframe(
                inv_df,
                column_config={
                    "Count": st.column_config.ProgressColumn(
                        "On-Shelf Stock",
                        help="Current item count on shelf",
                        format="%d",
                        min_value=0,
                        max_value=max(10, max([r["Count"] for r in inventory_rows] if inventory_rows else [10]))
                    ),
                    "Identifier": st.column_config.TextColumn("Item / Facings"),
                    "Type": st.column_config.TextColumn("Category"),
                    "Status": st.column_config.TextColumn("Health"),
                    "Urgency": st.column_config.TextColumn("Priority"),
                    "Recommended Action": st.column_config.TextColumn("Associate Action")
                },
                use_container_width=True,
                hide_index=True
            )

            # CSV Export
            csv_data = inv_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Export Restock Audit CSV",
                data=csv_data,
                file_name=f"Restock_Audit_{st.session_state.current_image_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        else:
            st.info("No items detected in current view.")

    # --------------------------------------------------------
    # TAB 4: STORE MANAGER AUDIT REPORT
    # --------------------------------------------------------
    with tab_report:
        st.markdown("#### 🏢 Executive Store Audit Certificate")
        
        audit_time_str = datetime.now().strftime("%B %d, %Y • %H:%M:%S")

        st.markdown(f"""
        <div style="background: rgba(30, 41, 59, 0.5); border: 1px solid rgba(255,255,255,0.1); border-radius: 12px; padding: 24px; margin-top: 12px;">
            <div style="display: flex; justify-content: space-between; align-items: flex-start; border-bottom: 1px solid rgba(255,255,255,0.08); padding-bottom: 16px; margin-bottom: 16px;">
                <div>
                    <h3 style="margin: 0; font-size: 1.3rem; color: #F8FAFC;">Shelvex Retail Audit Certificate</h3>
                    <p style="margin: 4px 0 0 0; color: #94A3B8; font-size: 0.85rem;">Autonomous Computer Vision Compliance Verification</p>
                </div>
                <div style="text-align: right;">
                    <span style="font-size: 0.78rem; color: #94A3B8;">AUDIT TIMESTAMP</span><br>
                    <span style="font-size: 0.9rem; font-weight: 700; color: #E2E8F0;">{audit_time_str}</span>
                </div>
            </div>

            <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px; margin-bottom: 20px;">
                <div>
                    <span style="font-size: 0.76rem; color: #94A3B8; text-transform: uppercase;">Facility & Location</span><br>
                    <span style="font-size: 0.95rem; font-weight: 600; color: #F1F5F9;">{st.session_state.store_id}</span>
                </div>
                <div>
                    <span style="font-size: 0.76rem; color: #94A3B8; text-transform: uppercase;">Aisle & Section</span><br>
                    <span style="font-size: 0.95rem; font-weight: 600; color: #F1F5F9;">{st.session_state.aisle_id}</span>
                </div>
                <div>
                    <span style="font-size: 0.76rem; color: #94A3B8; text-transform: uppercase;">Neural Inference Latency</span><br>
                    <span style="font-size: 0.95rem; font-weight: 600; color: #F1F5F9;">{st.session_state.inference_time_ms} ms</span>
                </div>
            </div>

            <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; background: rgba(15, 23, 42, 0.6); padding: 14px; border-radius: 8px; margin-bottom: 20px;">
                <div>
                    <span style="font-size: 0.74rem; color: #94A3B8;">Products Present</span><br>
                    <b style="font-size: 1.2rem; color: #38BDF8;">{total_products}</b>
                </div>
                <div>
                    <span style="font-size: 0.74rem; color: #94A3B8;">SKU Variants</span><br>
                    <b style="font-size: 1.2rem; color: #A855F7;">{unique_skus}</b>
                </div>
                <div>
                    <span style="font-size: 0.74rem; color: #94A3B8;">Empty Slots</span><br>
                    <b style="font-size: 1.2rem; color: #F43F5E;">{empty_spaces}</b>
                </div>
                <div>
                    <span style="font-size: 0.74rem; color: #94A3B8;">Compliance Rate</span><br>
                    <b style="font-size: 1.2rem; color: #10B981;">{fill_rate:.1f}%</b>
                </div>
            </div>

            <div style="font-size: 0.85rem; color: #CBD5E1; line-height: 1.6;">
                <b>Executive Recommendation:</b><br>
                {
                    "Shelf facings conform to standard merchandising guidelines. Maintain normal inspection intervals." if overall_status == "FULL" else
                    f"Action Required: {empty_spaces} vacant facings detected. Associate team should replenish depleted lines to prevent lost sales." if overall_status == "LOW" else
                    f"CRITICAL DEFICIT: Urgent restock required immediately. Multiple empty facings are causing significant out-of-stock exposure."
                }
            </div>
        </div>
        """, unsafe_allow_html=True)