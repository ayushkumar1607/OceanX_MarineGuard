"""
MarineGuard — Data Loading Utilities
JSON scenario loader, AIS data parser, result serialization.
"""

import json
import os
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field, asdict
from datetime import datetime


# ─── Data Classes ────────────────────────────────────────────────────────────

@dataclass
class SlickDetection:
    """Result from the Detection Engine."""
    detected: bool
    confidence: float
    centroid_lat: float
    centroid_lon: float
    area_km2: float
    polygon: List[List[float]]          # [[lat, lon], ...]
    estimated_age_hours: float
    slick_type: str                     # "oil", "biogenic", "look-alike"
    detection_time: str                 # ISO format UTC
    sar_metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OriginEstimate:
    """Result from the Drift Engine."""
    origin_lat: float
    origin_lon: float
    origin_zone_polygon: List[List[float]]  # Uncertainty ellipse
    release_time_earliest: str              # ISO format
    release_time_latest: str                # ISO format
    drift_path: List[List[float]]           # [[lat, lon], ...] backward path
    drift_distance_km: float
    drift_duration_hours: float
    uncertainty_radius_km: float
    wind_data: Dict[str, Any] = field(default_factory=dict)
    current_data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AISPosition:
    """Single AIS position report."""
    timestamp: str
    lat: float
    lon: float
    speed_knots: float
    heading: float
    course: float


@dataclass
class AISGap:
    """Detected AIS transponder gap."""
    start_time: str
    end_time: str
    duration_min: float
    start_pos: Dict[str, float]         # {"lat": ..., "lon": ...}
    end_pos: Dict[str, float]


@dataclass
class VesselCandidate:
    """Vessel candidate from the AIS Engine."""
    mmsi: str
    name: str
    vessel_type: str
    flag: str
    imo: str
    track: List[Dict[str, Any]]         # List of position dicts
    ais_gaps: List[Dict[str, Any]]      # List of gap dicts
    min_distance_to_origin_km: float
    time_in_zone: bool                  # Was the vessel in the zone during release window?


@dataclass
class AttributionResult:
    """Result from the Attribution Engine for one vessel."""
    mmsi: str
    name: str
    vessel_type: str
    flag: str
    attribution_score: float            # Overall weighted score [0, 1]
    rank: int
    risk_level: str                     # "CRITICAL", "HIGH", "MEDIUM", "LOW", "NONE"
    factor_scores: Dict[str, float]     # Individual factor scores
    weighted_scores: Dict[str, float]   # Factor scores × weights
    evidence: List[str]                 # Human-readable evidence statements
    anomalies: List[str]                # Detected behavioral anomalies
    track: List[Dict[str, Any]]
    ais_gaps: List[Dict[str, Any]]


@dataclass
class InvestigationResult:
    """Complete investigation result combining all engine outputs."""
    scenario_name: str
    detection: Dict[str, Any]
    origin: Dict[str, Any]
    vessels_analyzed: int
    vessels_filtered: int
    ranked_vessels: List[Dict[str, Any]]
    timeline_events: List[Dict[str, Any]]
    conclusion: str
    processing_time_sec: float


# ─── Loaders ─────────────────────────────────────────────────────────────────

def load_scenario(scenario_path: str) -> Dict[str, Any]:
    """Load a test scenario JSON file."""
    with open(scenario_path, "r", encoding="utf-8") as f:
        return json.load(f)


def list_scenarios(scenarios_dir: str) -> List[Dict[str, str]]:
    """List all available test scenario files."""
    scenarios = []
    if not os.path.isdir(scenarios_dir):
        return scenarios
    
    for filename in sorted(os.listdir(scenarios_dir)):
        if filename.endswith(".json"):
            filepath = os.path.join(scenarios_dir, filename)
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                scenarios.append({
                    "filename": filename,
                    "filepath": filepath,
                    "name": data.get("scenario_name", filename),
                    "description": data.get("description", ""),
                    "type": data.get("scenario_type", "unknown"),
                })
            except (json.JSONDecodeError, KeyError):
                continue
    
    return scenarios


def dataclass_to_dict(obj) -> Dict[str, Any]:
    """Convert a dataclass to a serializable dict."""
    if hasattr(obj, "__dataclass_fields__"):
        return asdict(obj)
    return obj


def save_results(results: InvestigationResult, output_path: str):
    """Save investigation results to JSON."""
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(dataclass_to_dict(results), f, indent=2, default=str)
