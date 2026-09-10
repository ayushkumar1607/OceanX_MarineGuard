"""
MarineGuard — Geospatial Utility Functions
Haversine distance, point-in-polygon, coordinate transforms, drift vectors.
"""

import math
import numpy as np
from typing import List, Tuple


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate the great-circle distance between two points on Earth (in km).
    Uses the Haversine formula.
    """
    R = 6371.0  # Earth's radius in km
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)

    a = math.sin(dphi / 2) ** 2 + \
        math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return R * c


def bearing_degrees(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate the initial bearing (forward azimuth) from point 1 to point 2.
    Returns bearing in degrees [0, 360).
    """
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dlambda = math.radians(lon2 - lon1)

    x = math.sin(dlambda) * math.cos(phi2)
    y = math.cos(phi1) * math.sin(phi2) - \
        math.sin(phi1) * math.cos(phi2) * math.cos(dlambda)

    theta = math.atan2(x, y)
    return (math.degrees(theta) + 360) % 360


def destination_point(lat: float, lon: float, bearing_deg: float, distance_km: float) -> Tuple[float, float]:
    """
    Calculate the destination point given a start point, bearing, and distance.
    Uses the spherical law of cosines.
    """
    R = 6371.0
    d = distance_km / R
    brng = math.radians(bearing_deg)
    phi1 = math.radians(lat)
    lambda1 = math.radians(lon)

    phi2 = math.asin(
        math.sin(phi1) * math.cos(d) +
        math.cos(phi1) * math.sin(d) * math.cos(brng)
    )
    lambda2 = lambda1 + math.atan2(
        math.sin(brng) * math.sin(d) * math.cos(phi1),
        math.cos(d) - math.sin(phi1) * math.sin(phi2)
    )

    return math.degrees(phi2), math.degrees(lambda2)


def point_in_polygon(lat: float, lon: float, polygon: List[List[float]]) -> bool:
    """
    Ray-casting algorithm to check if a point (lat, lon) is inside a polygon.
    Polygon is a list of [lat, lon] pairs.
    """
    n = len(polygon)
    inside = False
    j = n - 1

    for i in range(n):
        yi, xi = polygon[i]
        yj, xj = polygon[j]

        if ((yi > lon) != (yj > lon)) and \
                (lat < (xj - xi) * (lon - yi) / (yj - yi) + xi):
            inside = not inside
        j = i

    return inside


def create_circle_polygon(center_lat: float, center_lon: float,
                           radius_km: float, num_points: int = 36) -> List[List[float]]:
    """
    Create a circular polygon approximation around a center point.
    Returns list of [lat, lon] pairs.
    """
    polygon = []
    for i in range(num_points):
        angle = (360.0 / num_points) * i
        lat, lon = destination_point(center_lat, center_lon, angle, radius_km)
        polygon.append([lat, lon])
    polygon.append(polygon[0])  # Close the polygon
    return polygon


def create_ellipse_polygon(center_lat: float, center_lon: float,
                            semi_major_km: float, semi_minor_km: float,
                            rotation_deg: float = 0, num_points: int = 36) -> List[List[float]]:
    """
    Create an elliptical polygon around a center point.
    Used for origin zone uncertainty visualization.
    """
    polygon = []
    rot = math.radians(rotation_deg)

    for i in range(num_points):
        theta = (2 * math.pi / num_points) * i
        # Parametric ellipse
        x = semi_major_km * math.cos(theta)
        y = semi_minor_km * math.sin(theta)
        # Rotate
        x_rot = x * math.cos(rot) - y * math.sin(rot)
        y_rot = x * math.sin(rot) + y * math.cos(rot)
        # Convert to distance and bearing
        distance = math.sqrt(x_rot ** 2 + y_rot ** 2)
        bearing = math.degrees(math.atan2(x_rot, y_rot))
        lat, lon = destination_point(center_lat, center_lon, bearing, distance)
        polygon.append([lat, lon])

    polygon.append(polygon[0])  # Close
    return polygon


def compute_drift_vector(wind_speed_ms: float, wind_dir_deg: float,
                          current_speed_ms: float, current_dir_deg: float,
                          wind_factor: float = 0.03) -> Tuple[float, float]:
    """
    Compute the combined drift vector from wind and ocean current.
    Returns (drift_speed_ms, drift_direction_deg).
    
    Wind contributes ~3% of its speed (Ekman drift), current contributes fully.
    Directions follow meteorological convention (direction FROM which it blows/flows).
    """
    # Wind contribution (3% of wind, deflected ~15° to the right in Northern Hemisphere)
    wind_drift_speed = wind_speed_ms * wind_factor
    wind_drift_dir = (wind_dir_deg + 180 + 15) % 360  # Convert "from" to "to" + Ekman

    # Convert to cartesian components
    wind_vx = wind_drift_speed * math.sin(math.radians(wind_drift_dir))
    wind_vy = wind_drift_speed * math.cos(math.radians(wind_drift_dir))

    current_vx = current_speed_ms * math.sin(math.radians(current_dir_deg))
    current_vy = current_speed_ms * math.cos(math.radians(current_dir_deg))

    # Sum vectors
    total_vx = wind_vx + current_vx
    total_vy = wind_vy + current_vy

    drift_speed = math.sqrt(total_vx ** 2 + total_vy ** 2)
    drift_dir = (math.degrees(math.atan2(total_vx, total_vy)) + 360) % 360

    return drift_speed, drift_dir


def angular_difference(angle1: float, angle2: float) -> float:
    """
    Calculate the minimum angular difference between two bearings (0-180).
    """
    diff = abs(angle1 - angle2) % 360
    return min(diff, 360 - diff)


def knots_to_ms(knots: float) -> float:
    """Convert speed from knots to meters/second."""
    return knots * 0.514444


def ms_to_knots(ms: float) -> float:
    """Convert speed from meters/second to knots."""
    return ms / 0.514444


def km_to_nautical_miles(km: float) -> float:
    """Convert kilometers to nautical miles."""
    return km / 1.852


def nautical_miles_to_km(nm: float) -> float:
    """Convert nautical miles to kilometers."""
    return nm * 1.852
