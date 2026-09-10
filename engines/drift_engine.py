"""
MarineGuard — Drift Engine
Backward drift hindcasting using wind + ocean currents → Origin Zone estimation.

Inspired by OpenDrift / OpenOil (simplified Lagrangian particle tracking).

Pipeline (from slides):
1. Take slick detection location + time
2. Ingest wind vectors (ERA5) + ocean current vectors (CMEMS)
3. Backward Lagrangian tracking (reverse time)
4. Compute probable origin zone (uncertainty ellipse)
5. Estimate release time window
"""

import math
import numpy as np
from typing import Dict, Any, List, Tuple
from datetime import datetime, timedelta

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.geo_utils import (
    destination_point, bearing_degrees, haversine_km,
    compute_drift_vector, create_ellipse_polygon, create_circle_polygon
)
from config import (
    DRIFT_TIMESTEP_HOURS, DRIFT_MAX_HOURS_BACK,
    WIND_DRIFT_FACTOR, CURRENT_DRIFT_FACTOR, ORIGIN_UNCERTAINTY_KM
)


class DriftEngine:
    """
    Drift Engine: Backward drift modeling using wind & ocean currents.
    
    Computes the probable origin zone of an oil slick by tracing it
    backward in time using Lagrangian particle tracking.
    
    In production, would use OpenDrift/OpenOil with real ERA5 wind fields
    and CMEMS ocean current fields. For demo, uses simplified physics
    with synthetic environmental data.
    """

    def __init__(self):
        self.timestep_hours = DRIFT_TIMESTEP_HOURS
        self.max_hours_back = DRIFT_MAX_HOURS_BACK
        self.wind_drift_factor = WIND_DRIFT_FACTOR
        self.origin_uncertainty_km = ORIGIN_UNCERTAINTY_KM

    def get_environmental_data(self, lat: float, lon: float, time_str: str,
                                env_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get wind and current data for a given position and time.
        
        In production: query ERA5 (wind) and CMEMS (currents) APIs.
        In demo: use provided synthetic data with slight variation.
        """
        # Base values from scenario
        base_wind_speed = env_data.get("wind_speed_ms", 8.0)
        base_wind_dir = env_data.get("wind_direction_deg", 225)
        base_current_speed = env_data.get("current_speed_ms", 0.3)
        base_current_dir = env_data.get("current_direction_deg", 180)

        # Add realistic temporal/spatial variation
        np.random.seed(int(abs(lat * 1000 + lon * 100)) % 10000)
        wind_var = np.random.normal(0, 0.5)
        dir_var = np.random.normal(0, 10)

        return {
            "wind_speed_ms": max(0, base_wind_speed + wind_var),
            "wind_direction_deg": (base_wind_dir + dir_var) % 360,
            "current_speed_ms": max(0, base_current_speed + np.random.normal(0, 0.05)),
            "current_direction_deg": (base_current_dir + np.random.normal(0, 5)) % 360,
        }

    def backward_track(self, slick_lat: float, slick_lon: float,
                        detection_time: str, age_hours: float,
                        env_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform backward Lagrangian particle tracking.
        
        Starting from the slick detection location, trace backward in time
        using reversed wind and current vectors to estimate the origin.
        
        Returns:
            Dict with drift_path, origin coordinates, timing info
        """
        # Parse detection time
        if isinstance(detection_time, str):
            try:
                det_time = datetime.fromisoformat(detection_time.replace("Z", "+00:00"))
            except ValueError:
                det_time = datetime.utcnow()
        else:
            det_time = detection_time

        # Number of backward steps
        hours_back = min(age_hours * 1.5, self.max_hours_back)  # Go a bit further than estimated age
        n_steps = int(hours_back / self.timestep_hours)

        # Initialize tracking
        current_lat = slick_lat
        current_lon = slick_lon
        drift_path = [[current_lat, current_lon]]
        total_distance = 0.0

        for step in range(n_steps):
            step_time = det_time - timedelta(hours=(step + 1) * self.timestep_hours)
            step_time_str = step_time.isoformat()

            # Get environmental data at current position and time
            local_env = self.get_environmental_data(
                current_lat, current_lon, step_time_str, env_data
            )

            # Compute drift vector
            drift_speed, drift_dir = compute_drift_vector(
                wind_speed_ms=local_env["wind_speed_ms"],
                wind_dir_deg=local_env["wind_direction_deg"],
                current_speed_ms=local_env["current_speed_ms"],
                current_dir_deg=local_env["current_direction_deg"],
                wind_factor=self.wind_drift_factor,
            )

            # REVERSE the direction for backward tracking
            backward_dir = (drift_dir + 180) % 360

            # Distance traveled in this timestep
            distance_km = drift_speed * self.timestep_hours * 3.6  # m/s → km/h → km

            # Move to new position
            new_lat, new_lon = destination_point(
                current_lat, current_lon, backward_dir, distance_km
            )

            total_distance += distance_km
            current_lat = new_lat
            current_lon = new_lon
            drift_path.append([current_lat, current_lon])

        # Origin is the final position of backward tracking
        origin_lat = current_lat
        origin_lon = current_lon

        # Release time window (use naive UTC to avoid double-timezone suffix)
        earliest_dt = det_time.replace(tzinfo=None) - timedelta(hours=hours_back)
        latest_dt = det_time.replace(tzinfo=None) - timedelta(hours=max(1, age_hours * 0.5))
        release_earliest = earliest_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        release_latest = latest_dt.strftime("%Y-%m-%dT%H:%M:%SZ")

        return {
            "origin_lat": round(origin_lat, 6),
            "origin_lon": round(origin_lon, 6),
            "drift_path": drift_path,
            "drift_distance_km": round(total_distance, 2),
            "drift_duration_hours": round(hours_back, 1),
            "release_time_earliest": release_earliest,
            "release_time_latest": release_latest,
            "n_steps": n_steps,
        }

    def compute_origin_zone(self, origin_lat: float, origin_lon: float,
                             drift_distance_km: float,
                             wind_speed_ms: float) -> Dict[str, Any]:
        """
        Compute the uncertainty zone around the estimated origin.
        
        Uncertainty increases with:
        - Drift distance (longer drift → more error accumulation)
        - Wind speed (higher wind → more variable drift)
        
        Returns an uncertainty ellipse (origin zone polygon).
        """
        # Base uncertainty
        base_uncertainty = self.origin_uncertainty_km

        # Scale with drift distance (error accumulates)
        distance_factor = 1.0 + (drift_distance_km / 100) * 0.3

        # Scale with wind (higher wind → more uncertainty)
        wind_factor = 1.0 + max(0, (wind_speed_ms - 5)) * 0.1

        uncertainty_km = base_uncertainty * distance_factor * wind_factor

        # Create elliptical origin zone (elongated along drift direction)
        semi_major = uncertainty_km * 1.3
        semi_minor = uncertainty_km * 0.7
        rotation = np.random.uniform(0, 180)  # Based on drift direction in real implementation

        origin_zone_polygon = create_ellipse_polygon(
            origin_lat, origin_lon,
            semi_major, semi_minor,
            rotation_deg=rotation,
            num_points=48
        )

        return {
            "origin_zone_polygon": origin_zone_polygon,
            "uncertainty_radius_km": round(uncertainty_km, 2),
            "semi_major_km": round(semi_major, 2),
            "semi_minor_km": round(semi_minor, 2),
        }

    def forward_predict(self, origin_lat: float, origin_lon: float,
                         hours_forward: float, env_data: Dict[str, Any]) -> List[List[float]]:
        """
        Forward drift prediction for impact forecasting.
        
        Given a release point, predict where the oil will travel.
        Used for emergency response planning.
        """
        current_lat = origin_lat
        current_lon = origin_lon
        forward_path = [[current_lat, current_lon]]
        n_steps = int(hours_forward / self.timestep_hours)

        for step in range(n_steps):
            local_env = self.get_environmental_data(
                current_lat, current_lon, "", env_data
            )

            drift_speed, drift_dir = compute_drift_vector(
                wind_speed_ms=local_env["wind_speed_ms"],
                wind_dir_deg=local_env["wind_direction_deg"],
                current_speed_ms=local_env["current_speed_ms"],
                current_dir_deg=local_env["current_direction_deg"],
                wind_factor=self.wind_drift_factor,
            )

            distance_km = drift_speed * self.timestep_hours * 3.6

            new_lat, new_lon = destination_point(
                current_lat, current_lon, drift_dir, distance_km
            )

            current_lat = new_lat
            current_lon = new_lon
            forward_path.append([current_lat, current_lon])

        return forward_path

    def trace(self, detection: Dict[str, Any],
              env_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main drift engine entry point.
        
        Takes detection results + environmental data → origin estimate.
        """
        slick_lat = detection["centroid_lat"]
        slick_lon = detection["centroid_lon"]
        detection_time = detection["detection_time"]
        age_hours = detection.get("estimated_age_hours", 12)

        # Step 1: Backward tracking
        tracking = self.backward_track(
            slick_lat, slick_lon,
            detection_time, age_hours,
            env_data
        )

        # Step 2: Compute origin zone (uncertainty)
        wind_speed = env_data.get("wind_speed_ms", 5.0)
        origin_zone = self.compute_origin_zone(
            tracking["origin_lat"],
            tracking["origin_lon"],
            tracking["drift_distance_km"],
            wind_speed
        )

        # Step 3: Forward prediction (24h)
        forward_path = self.forward_predict(
            slick_lat, slick_lon, 24, env_data
        )

        return {
            "origin_lat": tracking["origin_lat"],
            "origin_lon": tracking["origin_lon"],
            "origin_zone_polygon": origin_zone["origin_zone_polygon"],
            "release_time_earliest": tracking["release_time_earliest"],
            "release_time_latest": tracking["release_time_latest"],
            "drift_path": tracking["drift_path"],
            "drift_distance_km": tracking["drift_distance_km"],
            "drift_duration_hours": tracking["drift_duration_hours"],
            "uncertainty_radius_km": origin_zone["uncertainty_radius_km"],
            "forward_prediction_path": forward_path,
            "wind_data": {
                "speed_ms": wind_speed,
                "direction_deg": env_data.get("wind_direction_deg", 0),
                "source": "ERA5 Reanalysis (simulated)",
            },
            "current_data": {
                "speed_ms": env_data.get("current_speed_ms", 0),
                "direction_deg": env_data.get("current_direction_deg", 0),
                "source": "CMEMS Global Ocean (simulated)",
            },
        }
