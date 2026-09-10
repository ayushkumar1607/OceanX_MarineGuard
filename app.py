"""
MarineGuard — Investigation UI
3-input upload (SAR image + environmental + AIS) with live preview.
Drift map uses matplotlib for maximum compatibility.
"""

import os
import io
import json

import numpy as np
from PIL import Image
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
import plotly.graph_objects as go

st.set_page_config(
    page_title="MarineGuard — Oil Spill Investigation",
    page_icon="🛢️",
    layout="wide",
    initial_sidebar_state="expanded",
)

from engines.detection_engine import DetectionEngine
from engines.drift_engine import DriftEngine
from engines.ais_engine import AISEngine
from engines.attribution_engine import AttributionEngine
from utils.visualization import (
    create_investigation_map,
    create_attribution_radar_chart,
    create_score_comparison_chart,
    create_confidence_gauge,
    create_factor_breakdown_chart,
    get_risk_color,
)
from config import SCORING_WEIGHTS

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAMPLES_DIR = os.path.join(BASE_DIR, "data", "test_samples")


# ═══════════════════════════════════════════════════════════════════════════
# CACHED ENGINES
# ═══════════════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner=False)
def get_detection_engine():
    return DetectionEngine()

@st.cache_resource(show_spinner=False)
def get_drift_engine():
    return DriftEngine()

@st.cache_resource(show_spinner=False)
def get_ais_engine():
    return AISEngine()

@st.cache_resource(show_spinner=False)
def get_attribution_engine():
    return AttributionEngine()


# ═══════════════════════════════════════════════════════════════════════════
# CSS
# ═══════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
    .stApp { background: linear-gradient(135deg, #0a0e17 0%, #111827 50%, #0f172a 100%); }
    .main-header {
        background: linear-gradient(135deg, rgba(30,58,95,0.8), rgba(15,23,42,0.9));
        border: 1px solid rgba(78,205,196,0.3); border-radius: 16px;
        padding: 24px 32px; margin-bottom: 24px;
    }
    .main-header h1 {
        background: linear-gradient(135deg, #4ECDC4, #45B7D1, #96E6A1);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        font-size: 2.2rem; font-weight: 800; margin: 0;
    }
    .main-header p { color: rgba(255,255,255,0.6); margin: 4px 0 0 0; }
    .metric-card {
        background: linear-gradient(135deg, rgba(30,41,59,0.8), rgba(15,23,42,0.6));
        border: 1px solid rgba(255,255,255,0.08); border-radius: 12px;
        padding: 20px; text-align: center;
    }
    .metric-value { font-size: 2rem; font-weight: 700; color: #4ECDC4; line-height: 1; }
    .metric-label { font-size: 0.85rem; color: rgba(255,255,255,0.5);
                    margin-top: 6px; text-transform: uppercase; letter-spacing: 1px; }
    .input-card {
        background: linear-gradient(135deg, rgba(30,41,59,0.7), rgba(15,23,42,0.5));
        border: 1px solid rgba(78,205,196,0.25); border-radius: 12px;
        padding: 16px; margin-bottom: 8px;
    }
    .input-card h4 { color: #4ECDC4; margin: 0 0 8px 0; font-size: 0.95rem; }
    .evidence-panel {
        background: rgba(30,41,59,0.6); border-left: 3px solid #4ECDC4;
        border-radius: 0 8px 8px 0; padding: 16px 20px; margin: 8px 0;
    }
    .vessel-card {
        background: linear-gradient(135deg, rgba(30,41,59,0.7), rgba(15,23,42,0.5));
        border: 1px solid rgba(255,255,255,0.08); border-radius: 12px;
        padding: 20px; margin: 12px 0;
    }
    .status-badge { display: inline-block; padding: 4px 12px; border-radius: 20px;
                    font-size: 0.8rem; font-weight: 600; }
    .status-critical { background: rgba(255,59,48,0.2); color: #FF3B30; }
    .status-high { background: rgba(255,149,0,0.2); color: #FF9500; }
    .status-medium { background: rgba(255,204,0,0.2); color: #FFCC00; }
    .status-low { background: rgba(52,199,89,0.2); color: #34C759; }
    .status-none { background: rgba(142,142,147,0.2); color: #8E8E93; }
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
        border-right: 1px solid rgba(78,205,196,0.15);
    }
    header[data-testid="stHeader"] { background: transparent; }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════
def list_sample_cases():
    if not os.path.isdir(SAMPLES_DIR):
        return []
    out = []
    for name in sorted(os.listdir(SAMPLES_DIR)):
        folder = os.path.join(SAMPLES_DIR, name)
        if not os.path.isdir(folder):
            continue
        req = ["sar.png", "environmental.json", "ais.json"]
        if all(os.path.exists(os.path.join(folder, r)) for r in req):
            out.append({"name": name, "path": folder})
    return out


def load_image_from_bytes(b):
    img = Image.open(io.BytesIO(b)).convert("L")
    if img.size[0] > 512 or img.size[1] > 512:
        img = img.resize((512, 512))
    return np.array(img)


def load_json_from_bytes(b):
    if isinstance(b, bytes):
        b = b.decode("utf-8")
    return json.loads(b)


def build_scenario(name, image_array, env, ais):
    return {
        "scenario_name": name,
        "sar_data": {
            "image_array": image_array,
            "image_height": image_array.shape[0],
            "image_width": image_array.shape[1],
        },
        "sar_metadata": {
            "satellite": env.get("satellite", "Sentinel-1A"),
            "mode": env.get("mode", "IW"),
            "polarization": env.get("polarization", "VV"),
            "resolution_m": env.get("resolution_m", 10),
            "acquisition_time": env.get("acquisition_time", "2026-01-01T00:00:00Z"),
            "slick_centroid_lat": env.get("slick_centroid_lat", 0.0),
            "slick_centroid_lon": env.get("slick_centroid_lon", 0.0),
            "slick_orientation": env.get("slick_orientation", 0),
        },
        "environmental_data": {
            "wind_speed_ms": env.get("wind_speed_ms", 5.0),
            "wind_direction_deg": env.get("wind_direction_deg", 180),
            "current_speed_ms": env.get("current_speed_ms", 0.3),
            "current_direction_deg": env.get("current_direction_deg", 180),
            "sea_state": env.get("sea_state", "moderate"),
            "wave_height_m": env.get("wave_height_m", 1.0),
            "source_wind": env.get("source_wind", "ERA5 Reanalysis"),
            "source_current": env.get("source_current", "CMEMS"),
        },
        "ais_data": ais,
    }


def run_pipeline(scenario):
    detection = get_detection_engine().detect(scenario)
    origin = None
    ais_results = None
    attribution = None
    if detection.get("detected"):
        try:
            origin = get_drift_engine().trace(detection, scenario["environmental_data"])
        except Exception as e:
            st.warning(f"Drift engine failed: {e}")
        try:
            if origin:
                ais_results = get_ais_engine().correlate(origin, scenario["ais_data"])
        except Exception as e:
            st.warning(f"AIS engine failed: {e}")
        try:
            if origin and ais_results:
                attribution = get_attribution_engine().attribute(origin, ais_results)
        except Exception as e:
            st.warning(f"Attribution engine failed: {e}")
    return detection, origin, ais_results, attribution


# ═══════════════════════════════════════════════════════════════════════════
# DRIFT MAP — matplotlib (works on every plotly/matplotlib version)
# ═══════════════════════════════════════════════════════════════════════════
def create_drift_matplotlib_figure(detection, origin):
    """Return a matplotlib Figure showing the drift path."""
    if not origin or not detection:
        return None

    det_lat = float(detection.get("centroid_lat", 0))
    det_lon = float(detection.get("centroid_lon", 0))
    orig_lat = float(origin.get("origin_lat", 0))
    orig_lon = float(origin.get("origin_lon", 0))
    drift_path = origin.get("drift_path", []) or []
    forward_path = origin.get("forward_prediction_path", []) or []
    zone = origin.get("origin_zone_polygon", []) or []

    fig, ax = plt.subplots(figsize=(11, 6.5), facecolor="#0a0e17")
    ax.set_facecolor("#0f172a")

    # Origin zone (filled polygon)
    if len(zone) > 2:
        poly_xy = [(p[1], p[0]) for p in zone]  # matplotlib uses (x=lon, y=lat)
        ax.add_patch(MplPolygon(
            poly_xy, closed=True,
            facecolor="#4ECDC4", edgecolor="#4ECDC4",
            alpha=0.22, linewidth=1.8, linestyle="--",
            label="Origin Uncertainty Zone",
        ))

    # Backward drift path
    if drift_path:
        lons = [p[1] for p in drift_path]
        lats = [p[0] for p in drift_path]
        ax.plot(lons, lats, "-o", color="#45B7D1",
                linewidth=2.5, markersize=5,
                label="Backward Drift Path", zorder=3)
        # Arrows showing direction (detection → origin)
        if len(lons) > 3:
            step = max(1, len(lons) // 6)
            for i in range(0, len(lons) - 1, step):
                ax.annotate(
                    "", xy=(lons[i + 1], lats[i + 1]),
                    xytext=(lons[i], lats[i]),
                    arrowprops=dict(arrowstyle="->", color="#45B7D1",
                                    lw=1.5, alpha=0.6),
                    zorder=4,
                )

    # Forward prediction
    if forward_path:
        flons = [p[1] for p in forward_path]
        flats = [p[0] for p in forward_path]
        ax.plot(flons, flats, "--", color="#FF9500",
                linewidth=2, alpha=0.85,
                label="Forward Prediction (24h)", zorder=3)

    # Detection point (orange circle)
    ax.plot(det_lon, det_lat, "o",
            color="#FF6B35", markersize=14,
            markeredgecolor="white", markeredgewidth=1.8,
            label="Slick Detection", zorder=5)
    ax.annotate(
        "Detection", (det_lon, det_lat),
        xytext=(10, 10), textcoords="offset points",
        color="#FF6B35", fontsize=10, fontweight="bold",
    )

    # Origin point (teal star)
    ax.plot(orig_lon, orig_lat, "*",
            color="#4ECDC4", markersize=22,
            markeredgecolor="white", markeredgewidth=1.5,
            label="Estimated Origin", zorder=5)
    ax.annotate(
        "Origin", (orig_lon, orig_lat),
        xytext=(10, -18), textcoords="offset points",
        color="#4ECDC4", fontsize=10, fontweight="bold",
    )

    # Styling
    ax.set_xlabel("Longitude (°E)", color="white", fontsize=11)
    ax.set_ylabel("Latitude (°N)", color="white", fontsize=11)
    ax.set_title("Drift Analysis — Backward Reconstruction & Forward Prediction",
                 color="white", fontsize=13, fontweight="bold", pad=12)
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_color("#334155")

    ax.grid(True, alpha=0.15, color="#94a3b8", linestyle=":", linewidth=0.7)
    ax.set_aspect("equal", adjustable="datalim")

    # Legend
    legend = ax.legend(loc="best", framealpha=0.85,
                       facecolor="#1e293b", edgecolor="#4ECDC4",
                       fontsize=9)
    for text in legend.get_texts():
        text.set_color("white")

    # Info box
    info = (
        f"Drift distance: {origin.get('drift_distance_km', 0):.1f} km\n"
        f"Drift duration: {origin.get('drift_duration_hours', 0):.1f} h\n"
        f"Uncertainty radius: {origin.get('uncertainty_radius_km', 0):.1f} km"
    )
    ax.text(
        0.02, 0.98, info,
        transform=ax.transAxes,
        va="top", ha="left",
        color="white", fontsize=9,
        bbox=dict(boxstyle="round,pad=0.5",
                  facecolor="#1e293b", edgecolor="#4ECDC4", alpha=0.85),
    )

    fig.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# HEADER
# ═══════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="main-header">
    <h1>🛢️ MarineGuard Investigation Dashboard</h1>
    <p>Sentinel-1 SAR + ERA5/CMEMS + Historical AIS → Vessel Attribution</p>
</div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 8px 0;">
        <h2 style="background: linear-gradient(135deg, #4ECDC4, #45B7D1);
                   -webkit-background-clip: text; -webkit-text-fill-color: transparent;
                   margin: 0;">MarineGuard</h2>
        <p style="color: rgba(255,255,255,0.5); font-size: 0.75rem;">
            Team Ocean X · SIH 2026
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 📥 Input Data")

    mode = st.radio(
        "input_mode",
        ["📤 Upload files", "📂 Load sample"],
        label_visibility="collapsed",
        key="input_mode_radio",
    )

    sar_bytes = env_bytes = ais_bytes = None
    case_name = None

    if mode == "📂 Load sample":
        samples = list_sample_cases()
        if not samples:
            st.error("No test samples found.\n\nRun:\n`python data/generate_test_samples.py`")
        else:
            case_name = st.selectbox("Test case",
                                     [s["name"] for s in samples],
                                     key="case_select")
            case_dir = next(s["path"] for s in samples if s["name"] == case_name)
            with open(os.path.join(case_dir, "sar.png"), "rb") as f:
                sar_bytes = f.read()
            with open(os.path.join(case_dir, "environmental.json"), "rb") as f:
                env_bytes = f.read()
            with open(os.path.join(case_dir, "ais.json"), "rb") as f:
                ais_bytes = f.read()
    else:
        sar_file = st.file_uploader(
            "1️⃣ SAR image (PNG/JPG/TIFF)",
            type=["png", "jpg", "jpeg", "tif", "tiff"],
            key="sar_upload",
        )
        env_file = st.file_uploader(
            "2️⃣ Environmental data (JSON)",
            type=["json"],
            key="env_upload",
        )
        ais_file = st.file_uploader(
            "3️⃣ Historical AIS traffic (JSON)",
            type=["json"],
            key="ais_upload",
        )
        if sar_file: sar_bytes = sar_file.read()
        if env_file: env_bytes = env_file.read()
        if ais_file: ais_bytes = ais_file.read()

        uploaded = sum(x is not None for x in [sar_bytes, env_bytes, ais_bytes])
        st.progress(uploaded / 3.0, text=f"{uploaded}/3 inputs uploaded")

    st.markdown("---")
    ready = all(x is not None for x in [sar_bytes, env_bytes, ais_bytes])
    run_clicked = st.button("🚀 Run Full Pipeline", type="primary",
                            use_container_width=True, disabled=not ready)

    st.markdown("---")
    with st.expander("ℹ️ Pipeline"):
        st.markdown("""
        1. **Detect** — U-Net on SAR
        2. **Trace** — Backward drift
        3. **Correlate** — AIS filtering
        4. **Rank** — Multi-factor scoring
        5. **Explain** — Evidence
        """)


# ═══════════════════════════════════════════════════════════════════════════
# UPLOADED INPUT PREVIEW
# ═══════════════════════════════════════════════════════════════════════════
if sar_bytes and env_bytes and ais_bytes:
    try:
        preview_img = load_image_from_bytes(sar_bytes)
        preview_env = load_json_from_bytes(env_bytes)
        preview_ais = load_json_from_bytes(ais_bytes)
        preview_vessels = preview_ais.get("vessels", []) if isinstance(preview_ais, dict) else []

        st.markdown("### 📥 Uploaded Input Preview")
        p1, p2, p3 = st.columns([1.2, 1, 1])

        with p1:
            st.markdown('<div class="input-card"><h4>1️⃣ SAR Image</h4></div>',
                        unsafe_allow_html=True)
            st.image(preview_img,
                     caption=f"{preview_img.shape[1]}×{preview_img.shape[0]} px",
                     use_container_width=True, clamp=True)

        with p2:
            st.markdown('<div class="input-card"><h4>2️⃣ Environmental Data</h4></div>',
                        unsafe_allow_html=True)
            for k in ["location_name", "acquisition_time", "satellite",
                      "wind_speed_ms", "wind_direction_deg",
                      "current_speed_ms", "current_direction_deg", "sea_state"]:
                if k in preview_env:
                    st.markdown(f"**{k}**: `{preview_env[k]}`")

        with p3:
            st.markdown('<div class="input-card"><h4>3️⃣ AIS Traffic</h4></div>',
                        unsafe_allow_html=True)
            st.markdown(f"**{len(preview_vessels)} vessels**")
            for v in preview_vessels[:6]:
                st.markdown(f"- {v.get('name', '—')} ({v.get('vessel_type', '')})")
            if len(preview_vessels) > 6:
                st.caption(f"+{len(preview_vessels) - 6} more")

        st.markdown("---")
    except Exception as e:
        st.warning(f"Could not preview uploaded inputs: {e}")


# ═══════════════════════════════════════════════════════════════════════════
# RUN PIPELINE
# ═══════════════════════════════════════════════════════════════════════════
if run_clicked and ready:
    with st.spinner("Running MarineGuard pipeline..."):
        try:
            image_array = load_image_from_bytes(sar_bytes)
            env_data = load_json_from_bytes(env_bytes)
            ais_data = load_json_from_bytes(ais_bytes)
            scenario = build_scenario(case_name or "Custom upload",
                                      image_array, env_data, ais_data)
            detection, origin, ais_results, attribution = run_pipeline(scenario)
            st.session_state["results"] = {
                "scenario": scenario,
                "image_array": image_array,
                "env_data": env_data,
                "ais_data": ais_data,
                "detection": detection,
                "origin": origin,
                "ais_results": ais_results,
                "attribution": attribution,
                "case_name": case_name or "Custom upload",
            }
            st.success("✅ Pipeline complete. Scroll down to view results.")
        except Exception as e:
            st.error(f"Pipeline failed: {e}")
            st.exception(e)


# ═══════════════════════════════════════════════════════════════════════════
# WELCOME
# ═══════════════════════════════════════════════════════════════════════════
if "results" not in st.session_state:
    if not (sar_bytes and env_bytes and ais_bytes):
        st.markdown("""
        <div style="text-align:center; padding: 60px 40px;">
            <h2 style="color: rgba(255,255,255,0.85); font-weight: 300;">
                Welcome to MarineGuard
            </h2>
            <p style="color: rgba(255,255,255,0.5); font-size: 1.05rem;
                      max-width: 720px; margin: 16px auto;">
                Upload a <strong>SAR image</strong>,
                <strong>environmental JSON</strong>, and
                <strong>AIS JSON</strong> — or pick a built-in test case —
                then click <strong style="color:#4ECDC4;">Run Full Pipeline</strong>.
            </p>
        </div>
        """, unsafe_allow_html=True)
    st.stop()


# ═══════════════════════════════════════════════════════════════════════════
# RESULTS
# ═══════════════════════════════════════════════════════════════════════════
R = st.session_state["results"]
detection = R.get("detection") or {}
origin = R.get("origin")
ais_results = R.get("ais_results")
attribution = R.get("attribution")
scenario = R.get("scenario", {})
image_array = R.get("image_array")
env_data = R.get("env_data", {})
ais_data = R.get("ais_data", {"vessels": []})

st.markdown(f"### 🔍 Investigation: `{R.get('case_name', 'Custom upload')}`")

detected = bool(detection.get("detected", False))
confidence = float(detection.get("confidence", 0))
area_km2 = float(detection.get("area_km2", 0))
vessels_scored = int(attribution.get("total_scored", 0)) if isinstance(attribution, dict) else 0

cols = st.columns(5)
for i, (label, value) in enumerate([
    ("Detected", "YES" if detected else "NO"),
    ("Confidence", f"{confidence:.0%}"),
    ("Area (km²)", f"{area_km2:.2f}"),
    ("Vessels Analyzed", str(vessels_scored)),
    ("Case", R.get("case_name", "—")[:18]),
]):
    with cols[i]:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{value}</div>
            <div class="metric-label">{label}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("---")

tab_in, tab_det, tab_drift, tab_map, tab_attr, tab_evi = st.tabs([
    "📥 Input Data",
    "📡 Detection",
    "🌊 Drift Analysis",
    "🗺️ Investigation Map",
    "📊 Attribution",
    "📋 Evidence",
])


# ─── Input tab ──────────────────────────────────────────────────────────────
with tab_in:
    st.markdown("### 📥 Input Data Sources")
    c1, c2, c3 = st.columns([1.2, 1, 1])
    with c1:
        st.markdown("#### 1️⃣ Sentinel-1 SAR Image")
        if image_array is not None:
            st.image(image_array,
                     caption=f"{image_array.shape[0]}×{image_array.shape[1]} grayscale",
                     use_container_width=True, clamp=True)
    with c2:
        st.markdown("#### 2️⃣ Environmental Data")
        for k in ["location_name", "acquisition_time", "satellite", "mode",
                  "polarization", "resolution_m", "wind_speed_ms",
                  "wind_direction_deg", "current_speed_ms",
                  "current_direction_deg", "sea_state", "wave_height_m",
                  "source_wind", "source_current"]:
            if k in env_data:
                st.markdown(f"- **{k}**: `{env_data[k]}`")
    with c3:
        st.markdown("#### 3️⃣ Historical AIS Traffic")
        vessels = ais_data.get("vessels", []) if isinstance(ais_data, dict) else []
        st.markdown(f"**{len(vessels)} vessels**")
        for v in vessels:
            st.markdown(f"- **{v.get('name', '—')}** · "
                        f"{v.get('vessel_type', '').title()} · "
                        f"{v.get('flag', '')} · "
                        f"{len(v.get('track', []))} pts")


# ─── Detection tab ──────────────────────────────────────────────────────────
with tab_det:
    st.markdown("### 📡 Oil Slick Detection")
    st.markdown(f"> {detection.get('explanation', '')}")
    ca, cb = st.columns(2)
    with ca:
        st.markdown("#### SAR Metadata")
        for k, v in (detection.get("sar_metadata") or {}).items():
            st.markdown(f"- **{k}**: `{v}`")
        st.markdown("#### Slick Properties")
        st.markdown(f"- **Type**: `{detection.get('slick_type', '—')}`")
        st.markdown(f"- **Location**: `{detection.get('centroid_lat', 0):.4f}, "
                    f"{detection.get('centroid_lon', 0):.4f}`")
        st.markdown(f"- **Area**: `{area_km2:.3f} km²`")
    with cb:
        try:
            st.plotly_chart(create_confidence_gauge(confidence),
                            use_container_width=True)
        except Exception:
            st.metric("Confidence", f"{confidence:.0%}")


# ─── Drift tab (matplotlib map) ─────────────────────────────────────────────
with tab_drift:
    if not detected or not origin:
        st.info("No spill detected — drift analysis not run.")
    else:
        st.markdown("### 🌊 Backward Drift Analysis")
        st.caption(
            "Tracing the slick backward through wind and ocean currents to "
            "estimate the probable origin zone and release time window."
        )

        c1, c2 = st.columns(2)
        with c1:
            st.metric("Origin latitude", f"{origin.get('origin_lat', 0):.4f}")
            st.metric("Origin longitude", f"{origin.get('origin_lon', 0):.4f}")
            st.metric("Uncertainty radius",
                      f"{origin.get('uncertainty_radius_km', 0):.1f} km")
        with c2:
            st.metric("Drift distance",
                      f"{origin.get('drift_distance_km', 0):.1f} km")
            st.metric("Drift duration",
                      f"{origin.get('drift_duration_hours', 0):.1f} h")
            st.markdown("**Release window:**")
            st.markdown(f"`{origin.get('release_time_earliest', '—')}`")
            st.markdown(f"`{origin.get('release_time_latest', '—')}`")

        st.markdown("---")
        st.markdown("#### 🗺️ Drift Path Visualization")
        st.caption(
            "🟠 Detection point · 🔵 Backward drift path · "
            "🟠 Forward prediction · ⭐ Estimated origin · "
            "◻ Uncertainty zone"
        )

        drift_fig = create_drift_matplotlib_figure(detection, origin)
        if drift_fig is not None:
            st.pyplot(drift_fig, use_container_width=True)
        else:
            st.info("Drift map unavailable.")

        st.markdown("---")
        st.markdown("#### Environmental Conditions")
        ec1, ec2 = st.columns(2)
        with ec1:
            st.metric("Wind Speed", f"{env_data.get('wind_speed_ms', 0):.1f} m/s")
            st.metric("Wind Direction", f"{env_data.get('wind_direction_deg', 0):.0f}°")
            st.caption(f"Source: {env_data.get('source_wind', 'ERA5')}")
        with ec2:
            st.metric("Current Speed", f"{env_data.get('current_speed_ms', 0):.2f} m/s")
            st.metric("Current Direction", f"{env_data.get('current_direction_deg', 0):.0f}°")
            st.caption(f"Source: {env_data.get('source_current', 'CMEMS')}")


# ─── Investigation Map tab ──────────────────────────────────────────────────
with tab_map:
    if not detected or not attribution:
        st.info("No spill detected — investigation map not available.")
    else:
        st.markdown("### 🗺️ Layered Evidence Map")
        st.caption(
            "Complete view: slick + origin zone + drift path + vessel tracks + AIS gaps."
        )
        try:
            from streamlit_folium import st_folium
            m = create_investigation_map(
                center=[detection.get("centroid_lat", 0),
                        detection.get("centroid_lon", 0)],
                zoom=9,
                slick_polygon=detection.get("polygon"),
                origin_zone=origin.get("origin_zone_polygon") if origin else None,
                drift_path=origin.get("drift_path") if origin else None,
                vessels=attribution.get("ranked_vessels", []),
                detection_point=[detection.get("centroid_lat", 0),
                                 detection.get("centroid_lon", 0)],
                origin_point=[origin.get("origin_lat", 0),
                              origin.get("origin_lon", 0)] if origin else None,
            )
            st_folium(m, width=None, height=600,
                      returned_objects=[],
                      key="evidence_map")
        except Exception as e:
            st.warning(f"Map render failed: {e}")


# ─── Attribution tab ────────────────────────────────────────────────────────
with tab_attr:
    if not attribution:
        st.info("Attribution not available — no spill detected.")
    else:
        st.markdown("### 📊 Multi-Factor Attribution")
        st.markdown(f"> {attribution.get('conclusion', '')}")
        ranked = attribution.get("ranked_vessels", [])
        if ranked:
            try:
                st.plotly_chart(create_score_comparison_chart(ranked),
                                use_container_width=True)
            except Exception:
                pass
            try:
                factors = {v["name"]: v.get("weighted_scores", {}) for v in ranked}
                st.plotly_chart(create_factor_breakdown_chart(factors),
                                use_container_width=True)
            except Exception:
                pass


# ─── Evidence tab ───────────────────────────────────────────────────────────
with tab_evi:
    if not detected:
        st.markdown("### 📋 Result")
        st.markdown(f"""
        <div class="evidence-panel">
            <p><strong>No oil spill detected</strong></p>
            <p>{detection.get('explanation', '')}</p>
        </div>
        """, unsafe_allow_html=True)
    elif not attribution or not attribution.get("ranked_vessels"):
        st.info("No vessel attributions available.")
    else:
        st.markdown("### 📋 Why Ranked? — Evidence Panel")
        for v in attribution["ranked_vessels"]:
            score = v.get("attribution_score", 0)
            risk = v.get("risk_level", "NONE").lower()
            color = get_risk_color(score)
            st.markdown(f"""
            <div class="vessel-card" style="border-left: 4px solid {color};">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <div>
                        <h3 style="margin:0;color:white;">#{v.get('rank','?')} — {v.get('name','—')}</h3>
                        <p style="color: rgba(255,255,255,0.5); margin:4px 0;">
                            MMSI {v.get('mmsi','—')} · {v.get('vessel_type','').title()} · {v.get('flag','')}
                        </p>
                    </div>
                    <div style="text-align:right;">
                        <span style="font-size:1.8rem;font-weight:700;color:{color};">{score:.0%}</span><br>
                        <span class="status-badge status-{risk}">{v.get('risk_level','NONE')}</span>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            with st.expander(f"Why ranked #{v.get('rank','?')}? — {v.get('name','—')}",
                             expanded=(v.get("rank", 99) <= 2)):
                ca, cb = st.columns([1, 1])
                with ca:
                    try:
                        st.plotly_chart(
                            create_attribution_radar_chart(
                                v.get("factor_scores", {}), v.get("name", "vessel")),
                            use_container_width=True)
                    except Exception:
                        pass
                with cb:
                    for f, s in v.get("factor_scores", {}).items():
                        w = SCORING_WEIGHTS.get(f, 0)
                        st.markdown(f"- **{f.replace('_',' ').title()}** `{s:.2f}` × {w:.0%} = `{s*w:.3f}`")
                for e in v.get("evidence", []):
                    st.markdown(f'<div class="evidence-panel"><p>{e}</p></div>',
                                unsafe_allow_html=True)


# ─── Footer ─────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center; padding: 30px 0 10px 0; margin-top: 40px;
            border-top: 1px solid rgba(255,255,255,0.05);">
    <p style="color: rgba(255,255,255,0.3); font-size: 0.8rem;">
        MarineGuard · Team Ocean X · SIH 2026 · PS 26143 (NTRO)
    </p>
</div>
""", unsafe_allow_html=True)