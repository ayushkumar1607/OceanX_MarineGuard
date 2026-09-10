"""
MarineGuard — AIS Engine
Vessel filtering + trajectory analysis + ML anomaly detection.

Pipeline:
  1. Ingest historical AIS traffic
  2. Spatial-temporal filtering
  3. Trajectory reconstruction
  4. AIS gap detection
  5. ML anomaly prediction (Random Forest)
"""

import os
import math
import joblib
import numpy as np
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.geo_utils import haversine_km, bearing_degrees, angular_difference
from config import (
    AIS_SPATIAL_BUFFER_KM, AIS_TEMPORAL_BUFFER_HOURS,
    AIS_GAP_THRESHOLD_MINUTES, VESSEL_SPEED_KNOTS_MAX,
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ANOMALY_MODEL_PATH = os.path.join(BASE_DIR, "models", "ais_anomaly_model.pkl")


class AISEngine:
    def __init__(self):
        self.spatial_buffer_km = AIS_SPATIAL_BUFFER_KM
        self.temporal_buffer_hours = AIS_TEMPORAL_BUFFER_HOURS
        self.gap_threshold_min = AIS_GAP_THRESHOLD_MINUTES

        # Load ML anomaly model if available
        self.anomaly_model = None
        self.anomaly_features = None
        if os.path.exists(ANOMALY_MODEL_PATH):
            try:
                bundle = joblib.load(ANOMALY_MODEL_PATH)
                self.anomaly_model = bundle["model"]
                self.anomaly_features = bundle["features"]
                print(f"[AISEngine] Loaded ML anomaly model from {ANOMALY_MODEL_PATH}")
            except Exception as e:
                print(f"[AISEngine] Failed to load anomaly model: {e}")
        else:
            print(f"[AISEngine] No ML anomaly model found at {ANOMALY_MODEL_PATH}. "
                  f"Run: python train_ais_anomaly.py")

    def parse_ais_timestamp(self, ts: str) -> Optional[datetime]:
        try:
            return datetime.fromisoformat(ts.replace("Z", "+00:00"))
        except (ValueError, AttributeError):
            return None

    def spatial_filter(self, vessels, origin_lat, origin_lon, buffer_km=None):
        if buffer_km is None:
            buffer_km = self.spatial_buffer_km
        filtered = []
        for vessel in vessels:
            track = vessel.get("track", [])
            min_dist = float("inf")
            passed_near = False
            for pos in track:
                dist = haversine_km(origin_lat, origin_lon,
                                    pos.get("lat", 0), pos.get("lon", 0))
                min_dist = min(min_dist, dist)
                if dist <= buffer_km:
                    passed_near = True
            if passed_near:
                vessel["min_distance_to_origin_km"] = round(min_dist, 2)
                filtered.append(vessel)
        return filtered

    def temporal_filter(self, vessels, release_earliest, release_latest,
                        buffer_hours=None):
        if buffer_hours is None:
            buffer_hours = self.temporal_buffer_hours
        try:
            t_early = datetime.fromisoformat(release_earliest.replace("Z", "+00:00"))
            t_late = datetime.fromisoformat(release_latest.replace("Z", "+00:00"))
        except (ValueError, AttributeError):
            return vessels

        window_start = t_early - timedelta(hours=buffer_hours)
        window_end = t_late + timedelta(hours=buffer_hours)

        filtered = []
        for vessel in vessels:
            track = vessel.get("track", [])
            in_window = False
            for pos in track:
                pos_time = self.parse_ais_timestamp(pos.get("timestamp", ""))
                if pos_time and window_start <= pos_time <= window_end:
                    in_window = True
                    break
            vessel["time_in_zone"] = in_window
            if in_window:
                filtered.append(vessel)
        return filtered

    def detect_ais_gaps(self, track, gap_threshold_min=None):
        if gap_threshold_min is None:
            gap_threshold_min = self.gap_threshold_min
        gaps = []
        sorted_track = sorted(track, key=lambda p: p.get("timestamp", ""))
        for i in range(len(sorted_track) - 1):
            t1 = self.parse_ais_timestamp(sorted_track[i].get("timestamp", ""))
            t2 = self.parse_ais_timestamp(sorted_track[i + 1].get("timestamp", ""))
            if t1 and t2:
                gap_minutes = (t2 - t1).total_seconds() / 60.0
                if gap_minutes > gap_threshold_min:
                    gaps.append({
                        "start_time": sorted_track[i]["timestamp"],
                        "end_time": sorted_track[i + 1]["timestamp"],
                        "duration_min": round(gap_minutes, 1),
                        "start_pos": {"lat": sorted_track[i]["lat"],
                                      "lon": sorted_track[i]["lon"]},
                        "end_pos": {"lat": sorted_track[i + 1]["lat"],
                                    "lon": sorted_track[i + 1]["lon"]},
                        "distance_km": round(haversine_km(
                            sorted_track[i]["lat"], sorted_track[i]["lon"],
                            sorted_track[i + 1]["lat"], sorted_track[i + 1]["lon"]
                        ), 2),
                    })
        return gaps

    def analyze_trajectory(self, track, origin_lat, origin_lon):
        if len(track) < 2:
            return {
                "avg_speed_knots": 0, "avg_heading": 0,
                "heading_alignment": 0, "speed_variance": 0,
                "loitering_detected": False, "course_changes": 0,
                "max_speed_knots": 0, "min_speed_knots": 0,
            }
        speeds = [p.get("speed_knots", 0) for p in track]
        headings = [p.get("heading", 0) for p in track]
        avg_speed = float(np.mean(speeds))
        speed_variance = float(np.var(speeds))

        alignments = []
        for pos in track:
            b = bearing_degrees(origin_lat, origin_lon,
                                pos.get("lat", 0), pos.get("lon", 0))
            a = 180 - angular_difference(pos.get("heading", 0), b)
            alignments.append(max(0, a) / 180.0)

        course_changes = 0
        for i in range(1, len(headings)):
            if angular_difference(headings[i - 1], headings[i]) > 30:
                course_changes += 1

        loitering = avg_speed < 3 and speed_variance < 2

        return {
            "avg_speed_knots": round(avg_speed, 1),
            "avg_heading": round(float(np.mean(headings)), 1),
            "heading_alignment": round(float(np.mean(alignments)), 3),
            "speed_variance": round(speed_variance, 2),
            "loitering_detected": bool(loitering),
            "course_changes": course_changes,
            "max_speed_knots": round(float(max(speeds)), 1),
            "min_speed_knots": round(float(min(speeds)), 1),
        }

    def predict_anomaly(self, vessel):
        """Use ML model to predict suspicious behavior probability."""
        if self.anomaly_model is None:
            return {"probability": None, "label": None,
                    "note": "ML model not loaded"}

        traj = vessel.get("trajectory_analysis", {})
        gaps = vessel.get("ais_gaps", [])

        num_gaps = len(gaps)
        max_gap_dur = max((g.get("duration_min", 0) for g in gaps), default=0.0)
        total_gap_min = sum(g.get("duration_min", 0) for g in gaps)

        feat = {
            "avg_speed_knots": traj.get("avg_speed_knots", 0),
            "speed_variance": traj.get("speed_variance", 0),
            "max_speed_knots": traj.get("max_speed_knots", 0),
            "min_speed_knots": traj.get("min_speed_knots", 0),
            "course_changes": traj.get("course_changes", 0),
            "loitering": int(traj.get("loitering_detected", False)),
            "num_ais_gaps": num_gaps,
            "max_gap_duration_min": float(max_gap_dur),
            "total_gap_minutes": float(total_gap_min),
            "heading_alignment": traj.get("heading_alignment", 0),
        }
        x = np.array([[feat[f] for f in self.anomaly_features]])
        proba = float(self.anomaly_model.predict_proba(x)[0, 1])
        label = int(proba > 0.5)
        return {"probability": round(proba, 4), "label": label,
                "features": feat}

    def correlate(self, origin, ais_data):
        vessels = ais_data.get("vessels", [])
        origin_lat = origin["origin_lat"]
        origin_lon = origin["origin_lon"]
        release_earliest = origin.get("release_time_earliest", "")
        release_latest = origin.get("release_time_latest", "")

        total_vessels = len(vessels)
        spatially_filtered = self.spatial_filter(vessels, origin_lat, origin_lon)
        candidates = self.temporal_filter(spatially_filtered,
                                          release_earliest, release_latest)

        analyzed = []
        for vessel in candidates:
            track = vessel.get("track", [])
            vessel["ais_gaps"] = self.detect_ais_gaps(track)
            vessel["trajectory_analysis"] = self.analyze_trajectory(
                track, origin_lat, origin_lon)
            if "min_distance_to_origin_km" not in vessel:
                md = float("inf")
                for pos in track:
                    md = min(md, haversine_km(origin_lat, origin_lon,
                                              pos.get("lat", 0),
                                              pos.get("lon", 0)))
                vessel["min_distance_to_origin_km"] = round(md, 2)
            vessel["ml_anomaly"] = self.predict_anomaly(vessel)
            analyzed.append(vessel)

        return {
            "total_vessels_in_data": total_vessels,
            "spatially_filtered": len(spatially_filtered),
            "temporally_filtered": len(candidates),
            "candidates": analyzed,
            "ml_anomaly_model_loaded": self.anomaly_model is not None,
            "filter_params": {
                "spatial_buffer_km": self.spatial_buffer_km,
                "temporal_buffer_hours": self.temporal_buffer_hours,
                "gap_threshold_min": self.gap_threshold_min,
            },
        }