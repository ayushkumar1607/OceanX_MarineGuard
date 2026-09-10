# MarineGuard — From Slick Detection to Maritime Evidence
# Global Configuration & Constants

import os

# ─── Project Paths ───────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
TEST_SCENARIOS_DIR = os.path.join(DATA_DIR, "test_scenarios")
ASSETS_DIR = os.path.join(BASE_DIR, "assets")

# ─── Detection Engine ────────────────────────────────────────────────────────
DETECTION_CONFIDENCE_THRESHOLD = 0.50   # Below this → "No spill detected"
MIN_SLICK_AREA_KM2 = 0.1               # Minimum area to consider a valid slick
SAR_RESOLUTION_M = 10                   # Sentinel-1 IW mode ground resolution

# ─── Drift Engine ────────────────────────────────────────────────────────────
DRIFT_TIMESTEP_HOURS = 1                # Time step for Lagrangian tracking
DRIFT_MAX_HOURS_BACK = 48               # Maximum backward hindcast window
WIND_DRIFT_FACTOR = 0.03                # ~3% of wind speed contributes to drift
CURRENT_DRIFT_FACTOR = 1.0              # Ocean current contributes fully
ORIGIN_UNCERTAINTY_KM = 15              # Radius of uncertainty around origin

# ─── AIS Engine ──────────────────────────────────────────────────────────────
AIS_SPATIAL_BUFFER_KM = 50              # Search radius around origin zone
AIS_TEMPORAL_BUFFER_HOURS = 6           # Extra time buffer beyond release window
AIS_GAP_THRESHOLD_MINUTES = 30          # Gap > this is considered suspicious
VESSEL_SPEED_KNOTS_MAX = 25             # Filter out unrealistic speeds

# ─── Attribution Engine ──────────────────────────────────────────────────────
# Multi-factor scoring weights (must sum to 1.0)
SCORING_WEIGHTS = {
    "proximity":   0.30,   # Distance from vessel to origin zone
    "temporal":    0.25,   # Timing alignment with release window
    "trajectory":  0.20,   # Course/heading alignment with drift
    "ais_gap":     0.15,   # Suspicious transponder gaps
    "behavioral":  0.10,   # Speed changes, loitering, course deviation
}

# Attribution confidence thresholds
ATTRIBUTION_HIGH_CONFIDENCE = 0.75
ATTRIBUTION_MEDIUM_CONFIDENCE = 0.50
ATTRIBUTION_LOW_CONFIDENCE = 0.30

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
CARTO_API_KEY = os.getenv("CARTO_API_KEY")

# ─── UI / Map Configuration ─────────────────────────────────────────────────
MAP_DEFAULT_CENTER = [15.0, 72.0]       # Default center (Arabian Sea)
MAP_DEFAULT_ZOOM = 8

# Switched to Esri Dark Gray Base since CARTO's CDN is aggressively rejecting connections/keys
MAP_TILE_STYLE = "https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Dark_Gray_Base/MapServer/tile/{z}/{y}/{x}"
MAP_ATTR = 'Tiles &copy; Esri &mdash; Esri, DeLorme, NAVTEQ'

# Color palette for vessel risk levels
RISK_COLORS = {
    "critical": "#FF3B30",   # Red
    "high":     "#FF9500",   # Orange
    "medium":   "#FFCC00",   # Yellow
    "low":      "#34C759",   # Green
    "none":     "#8E8E93",   # Gray
}

# Slick visualization
SLICK_COLOR = "#FF6B35"
ORIGIN_ZONE_COLOR = "#4ECDC4"
DRIFT_PATH_COLOR = "#45B7D1"

# ─── Vessel Types ────────────────────────────────────────────────────────────
VESSEL_TYPES = {
    "tanker":       "Oil/Chemical Tanker",
    "cargo":        "General Cargo",
    "bulk":         "Bulk Carrier",
    "container":    "Container Ship",
    "fishing":      "Fishing Vessel",
    "passenger":    "Passenger Ship",
    "tug":          "Tug/Supply Vessel",
    "other":        "Other",
}
