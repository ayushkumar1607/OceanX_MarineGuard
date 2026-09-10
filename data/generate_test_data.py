"""
MarineGuard — Test Data Generator
Generates 3 synthetic test scenarios:
  1. Major Oil Spill (Arabian Sea, clear attribution)
  2. No Spill / False Positive (SAR look-alike)
  3. Minor/Very Little Spill (busy shipping lane, ambiguous attribution)

Each scenario includes:
  - SAR detection data (slick params, image metadata)
  - Environmental data (wind, ocean currents)
  - AIS vessel traffic (positions, headings, speeds, vessel info)
"""

import json
import os
from datetime import datetime, timedelta
import random
import math

# Output directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "test_scenarios")


def generate_vessel_track(start_lat, start_lon, start_time, num_points=24,
                           speed_knots=12, heading=180, speed_variance=1.0,
                           heading_variance=5.0, gap_start=None, gap_duration_min=0):
    """Generate a realistic AIS vessel track."""
    track = []
    lat, lon = start_lat, start_lon
    current_time = datetime.fromisoformat(start_time)

    for i in range(num_points):
        # Skip points during AIS gap
        if gap_start is not None and gap_start <= i < gap_start + max(1, gap_duration_min // 30):
            # Still update position (vessel moves during gap, but no AIS)
            spd = speed_knots + random.gauss(0, speed_variance)
            hdg = (heading + random.gauss(0, heading_variance)) % 360
            dt_hours = 0.5  # 30-minute intervals
            dist_km = spd * 1.852 * dt_hours  # nautical miles to km
            lat += dist_km * math.cos(math.radians(hdg)) / 111.0
            lon += dist_km * math.sin(math.radians(hdg)) / (111.0 * math.cos(math.radians(lat)))
            current_time += timedelta(minutes=30)
            continue

        spd = max(0.5, speed_knots + random.gauss(0, speed_variance))
        hdg = (heading + random.gauss(0, heading_variance)) % 360
        cog = (hdg + random.gauss(0, 3)) % 360

        track.append({
            "timestamp": current_time.isoformat() + "Z",
            "lat": round(lat, 6),
            "lon": round(lon, 6),
            "speed_knots": round(spd, 1),
            "heading": round(hdg, 1),
            "course": round(cog, 1),
        })

        # Move vessel
        dt_hours = 0.5
        dist_km = spd * 1.852 * dt_hours
        lat += dist_km * math.cos(math.radians(hdg)) / 111.0
        lon += dist_km * math.sin(math.radians(hdg)) / (111.0 * math.cos(math.radians(lat)))
        current_time += timedelta(minutes=30)

    return track


def scenario_major_spill():
    """
    Scenario 1: Major Oil Spill
    - Location: Arabian Sea, west of Mumbai (18.8°N, 71.5°E)
    - Large slick (~15 km²), high confidence (0.92)
    - 8 vessels in area, 1 tanker with AIS gap is the culprit
    - Clear attribution to MT OCEAN CARRIER (87% confidence)
    """
    base_time = "2026-09-09T06:00:00"
    detection_time = "2026-09-09T14:30:00"

    scenario = {
        "scenario_name": "Major Oil Spill — Arabian Sea",
        "scenario_type": "major_spill",
        "description": (
            "A large oil slick (~15 km²) detected in the Arabian Sea west of Mumbai. "
            "Strong SAR signature with high confidence. An oil tanker (MT OCEAN CARRIER) "
            "shows a 90-minute AIS gap near the origin zone during the release window, "
            "with trajectory alignment and behavioral anomalies."
        ),
        "location": "Arabian Sea, west of Mumbai, India",

        # ── SAR Data ──
        "sar_data": {
            "has_slick": True,
            "image_height": 512,
            "image_width": 512,
            "seed": 42,
            "slick_params": {
                "center_row": 256,
                "center_col": 260,
                "radius": 65,
                "elongation": 2.0,
                "angle": 35,
                "dampening": 0.7,
                "confidence": 0.92,
                "target_area_km2": 15.2,
            },
        },
        "sar_metadata": {
            "satellite": "Sentinel-1A",
            "mode": "IW",
            "polarization": "VV",
            "resolution_m": 10,
            "acquisition_time": detection_time + "Z",
            "orbit": "Descending",
            "slick_centroid_lat": 18.82,
            "slick_centroid_lon": 71.48,
            "slick_orientation": 35,
        },

        # ── Environmental Data ──
        "environmental_data": {
            "wind_speed_ms": 7.5,
            "wind_direction_deg": 225,  # SW wind
            "current_speed_ms": 0.35,
            "current_direction_deg": 190,  # Flowing roughly southward
            "sea_state": "moderate",
            "wave_height_m": 1.2,
            "source_wind": "ERA5 Reanalysis",
            "source_current": "CMEMS Global Ocean Physics",
        },

        # ── AIS Data (8 vessels) ──
        "ais_data": {
            "vessels": [
                # ===== VESSEL 1: THE CULPRIT (Oil Tanker with AIS gap) =====
                {
                    "mmsi": "419001234",
                    "name": "MT OCEAN CARRIER",
                    "vessel_type": "tanker",
                    "flag": "Panama",
                    "imo": "9876543",
                    "callsign": "3FXY9",
                    "length_m": 274,
                    "beam_m": 48,
                    "draft_m": 16.5,
                    "cargo": "Crude Oil",
                    "track": generate_vessel_track(
                        start_lat=19.15, start_lon=71.20,
                        start_time=base_time, num_points=24,
                        speed_knots=11, heading=200, speed_variance=3.0,
                        heading_variance=8, gap_start=8, gap_duration_min=90
                    ),
                },
                # ===== VESSEL 2: Cargo ship passing through =====
                {
                    "mmsi": "477001111",
                    "name": "MV GLOBAL PHOENIX",
                    "vessel_type": "cargo",
                    "flag": "Hong Kong",
                    "imo": "9812345",
                    "callsign": "VRBC7",
                    "length_m": 189,
                    "beam_m": 32,
                    "draft_m": 10.2,
                    "cargo": "General Cargo",
                    "track": generate_vessel_track(
                        start_lat=19.00, start_lon=71.00,
                        start_time=base_time, num_points=24,
                        speed_knots=14, heading=170, speed_variance=0.5,
                        heading_variance=2
                    ),
                },
                # ===== VESSEL 3: Container ship on normal route =====
                {
                    "mmsi": "352001555",
                    "name": "CMA MUMBAI EXPRESS",
                    "vessel_type": "container",
                    "flag": "Panama",
                    "imo": "9654321",
                    "callsign": "3FAB2",
                    "length_m": 336,
                    "beam_m": 45,
                    "draft_m": 14.0,
                    "cargo": "Containers",
                    "track": generate_vessel_track(
                        start_lat=18.60, start_lon=71.80,
                        start_time=base_time, num_points=24,
                        speed_knots=18, heading=250, speed_variance=0.3,
                        heading_variance=1
                    ),
                },
                # ===== VESSEL 4: Fishing vessel (nearby but small) =====
                {
                    "mmsi": "419005678",
                    "name": "FV SAGAR RANI",
                    "vessel_type": "fishing",
                    "flag": "India",
                    "imo": "",
                    "callsign": "VTFR3",
                    "length_m": 22,
                    "beam_m": 6,
                    "draft_m": 2.5,
                    "cargo": "Fish",
                    "track": generate_vessel_track(
                        start_lat=18.90, start_lon=71.55,
                        start_time=base_time, num_points=24,
                        speed_knots=4, heading=90, speed_variance=2.0,
                        heading_variance=30
                    ),
                },
                # ===== VESSEL 5: Bulk carrier far away =====
                {
                    "mmsi": "538003456",
                    "name": "MV IRON FORTUNE",
                    "vessel_type": "bulk",
                    "flag": "Marshall Islands",
                    "imo": "9734567",
                    "callsign": "V7AB3",
                    "length_m": 229,
                    "beam_m": 38,
                    "draft_m": 13.8,
                    "cargo": "Iron Ore",
                    "track": generate_vessel_track(
                        start_lat=18.40, start_lon=70.80,
                        start_time=base_time, num_points=24,
                        speed_knots=13, heading=310, speed_variance=0.3,
                        heading_variance=2
                    ),
                },
                # ===== VESSEL 6: Tanker (passing but no gaps) =====
                {
                    "mmsi": "636012345",
                    "name": "MT PERSIAN STAR",
                    "vessel_type": "tanker",
                    "flag": "Liberia",
                    "imo": "9845678",
                    "callsign": "A8CD1",
                    "length_m": 244,
                    "beam_m": 42,
                    "draft_m": 15.0,
                    "cargo": "Refined Products",
                    "track": generate_vessel_track(
                        start_lat=19.10, start_lon=71.60,
                        start_time=base_time, num_points=24,
                        speed_knots=12, heading=260, speed_variance=0.4,
                        heading_variance=3
                    ),
                },
                # ===== VESSEL 7: Passenger ferry (regular route) =====
                {
                    "mmsi": "419009876",
                    "name": "MV GATEWAY",
                    "vessel_type": "passenger",
                    "flag": "India",
                    "imo": "9567890",
                    "callsign": "VTGW1",
                    "length_m": 95,
                    "beam_m": 16,
                    "draft_m": 4.5,
                    "cargo": "Passengers",
                    "track": generate_vessel_track(
                        start_lat=18.95, start_lon=72.80,
                        start_time=base_time, num_points=24,
                        speed_knots=16, heading=270, speed_variance=0.5,
                        heading_variance=5
                    ),
                },
                # ===== VESSEL 8: Tug/supply vessel =====
                {
                    "mmsi": "419002468",
                    "name": "TUG SAMRAT",
                    "vessel_type": "tug",
                    "flag": "India",
                    "imo": "",
                    "callsign": "VTTS2",
                    "length_m": 38,
                    "beam_m": 11,
                    "draft_m": 4.0,
                    "cargo": "Supply",
                    "track": generate_vessel_track(
                        start_lat=18.75, start_lon=71.35,
                        start_time=base_time, num_points=24,
                        speed_knots=8, heading=150, speed_variance=2.0,
                        heading_variance=15
                    ),
                },
            ],
        },
    }

    return scenario


def scenario_no_spill():
    """
    Scenario 2: No Spill / False Positive
    - Location: Bay of Bengal (13.5°N, 80.8°E)
    - SAR anomaly with low confidence (0.35)
    - Small, irregular shape → biogenic slick / calm water
    - System correctly identifies as "No spill detected"
    """
    base_time = "2026-09-08T08:00:00"
    detection_time = "2026-09-08T16:15:00"

    scenario = {
        "scenario_name": "No Spill — SAR False Positive (Bay of Bengal)",
        "scenario_type": "no_spill",
        "description": (
            "A SAR anomaly detected in the Bay of Bengal near Chennai. "
            "Low confidence detection (0.35) with irregular shape inconsistent "
            "with petroleum-based oil slicks. Likely caused by natural calm water "
            "zones (low wind) or biogenic surface films from algal blooms. "
            "The system correctly classifies this as a false positive."
        ),
        "location": "Bay of Bengal, near Chennai, India",

        "sar_data": {
            "has_slick": True,
            "image_height": 512,
            "image_width": 512,
            "seed": 123,
            "slick_params": {
                "center_row": 300,
                "center_col": 200,
                "radius": 25,
                "elongation": 0.8,  # Nearly circular (unusual for oil)
                "angle": 10,
                "dampening": 0.2,  # Weak signal
                "confidence": 0.35,
                "target_area_km2": 0.5,
            },
        },
        "sar_metadata": {
            "satellite": "Sentinel-1B",
            "mode": "IW",
            "polarization": "VV",
            "resolution_m": 10,
            "acquisition_time": detection_time + "Z",
            "orbit": "Ascending",
            "slick_centroid_lat": 13.52,
            "slick_centroid_lon": 80.78,
            "slick_orientation": 10,
        },

        "environmental_data": {
            "wind_speed_ms": 2.0,  # Very low wind (causes look-alikes)
            "wind_direction_deg": 90,
            "current_speed_ms": 0.15,
            "current_direction_deg": 160,
            "sea_state": "calm",
            "wave_height_m": 0.3,
            "source_wind": "ERA5 Reanalysis",
            "source_current": "CMEMS Global Ocean Physics",
        },

        "ais_data": {
            "vessels": [
                {
                    "mmsi": "419011111",
                    "name": "MV COROMANDEL",
                    "vessel_type": "cargo",
                    "flag": "India",
                    "imo": "9111111",
                    "callsign": "VTCM1",
                    "length_m": 155,
                    "beam_m": 25,
                    "draft_m": 8.5,
                    "cargo": "Steel",
                    "track": generate_vessel_track(
                        start_lat=13.80, start_lon=80.50,
                        start_time=base_time, num_points=20,
                        speed_knots=13, heading=195, speed_variance=0.3,
                        heading_variance=2
                    ),
                },
                {
                    "mmsi": "419022222",
                    "name": "FV MARINA QUEEN",
                    "vessel_type": "fishing",
                    "flag": "India",
                    "imo": "",
                    "callsign": "VTMQ2",
                    "length_m": 18,
                    "beam_m": 5,
                    "draft_m": 2.0,
                    "cargo": "Fish",
                    "track": generate_vessel_track(
                        start_lat=13.55, start_lon=80.90,
                        start_time=base_time, num_points=20,
                        speed_knots=5, heading=45, speed_variance=2.0,
                        heading_variance=40
                    ),
                },
                {
                    "mmsi": "563033333",
                    "name": "MV SINGAPORE PRIDE",
                    "vessel_type": "container",
                    "flag": "Singapore",
                    "imo": "9333333",
                    "callsign": "9VSP3",
                    "length_m": 294,
                    "beam_m": 40,
                    "draft_m": 12.5,
                    "cargo": "Containers",
                    "track": generate_vessel_track(
                        start_lat=13.20, start_lon=81.20,
                        start_time=base_time, num_points=20,
                        speed_knots=19, heading=230, speed_variance=0.2,
                        heading_variance=1
                    ),
                },
            ],
        },
    }

    return scenario


def scenario_minor_spill():
    """
    Scenario 3: Minor / Very Little Spill
    - Location: Near Chennai Port (13.1°N, 80.3°E)
    - Small slick (~0.8 km²), moderate confidence (0.68)
    - 12 vessels in busy shipping lane
    - Top 3 vessels have similar scores — no single clear culprit
    - Demonstrates multi-factor scoring preventing false accusations
    """
    base_time = "2026-09-10T02:00:00"
    detection_time = "2026-09-10T10:45:00"

    scenario = {
        "scenario_name": "Minor Spill — Chennai Port Approaches",
        "scenario_type": "minor_spill",
        "description": (
            "A small oil slick (~0.8 km²) detected near Chennai port approaches. "
            "Moderate confidence (0.68) suggests a possible bilge dump or minor "
            "operational discharge. 12 vessels in busy shipping lane. "
            "Attribution is ambiguous — top 3 candidates have similar scores, "
            "demonstrating the system's refusal to make single-factor accusations."
        ),
        "location": "Chennai Port Approaches, India",

        "sar_data": {
            "has_slick": True,
            "image_height": 512,
            "image_width": 512,
            "seed": 789,
            "slick_params": {
                "center_row": 280,
                "center_col": 230,
                "radius": 18,
                "elongation": 1.8,
                "angle": 60,
                "dampening": 0.45,
                "confidence": 0.68,
                "target_area_km2": 0.8,
            },
        },
        "sar_metadata": {
            "satellite": "Sentinel-1A",
            "mode": "IW",
            "polarization": "VV+VH",
            "resolution_m": 10,
            "acquisition_time": detection_time + "Z",
            "orbit": "Descending",
            "slick_centroid_lat": 13.12,
            "slick_centroid_lon": 80.32,
            "slick_orientation": 60,
        },

        "environmental_data": {
            "wind_speed_ms": 5.0,
            "wind_direction_deg": 180,
            "current_speed_ms": 0.25,
            "current_direction_deg": 210,
            "sea_state": "slight",
            "wave_height_m": 0.8,
            "source_wind": "ERA5 Reanalysis",
            "source_current": "CMEMS Global Ocean Physics",
        },

        "ais_data": {
            "vessels": [
                # Suspect 1: Small tanker with minor gap
                {
                    "mmsi": "419041111",
                    "name": "MT KAVERI",
                    "vessel_type": "tanker",
                    "flag": "India",
                    "imo": "9441111",
                    "callsign": "VTKV1",
                    "length_m": 135,
                    "beam_m": 20,
                    "draft_m": 7.5,
                    "cargo": "Fuel Oil",
                    "track": generate_vessel_track(
                        start_lat=13.25, start_lon=80.20,
                        start_time=base_time, num_points=20,
                        speed_knots=9, heading=175, speed_variance=1.5,
                        heading_variance=8, gap_start=6, gap_duration_min=45
                    ),
                },
                # Suspect 2: Cargo vessel, close proximity
                {
                    "mmsi": "419042222",
                    "name": "MV EASTERN DAWN",
                    "vessel_type": "cargo",
                    "flag": "India",
                    "imo": "9442222",
                    "callsign": "VTED2",
                    "length_m": 115,
                    "beam_m": 18,
                    "draft_m": 6.0,
                    "cargo": "Chemicals",
                    "track": generate_vessel_track(
                        start_lat=13.18, start_lon=80.28,
                        start_time=base_time, num_points=20,
                        speed_knots=8, heading=190, speed_variance=2.0,
                        heading_variance=10
                    ),
                },
                # Suspect 3: Another tanker, some speed variance
                {
                    "mmsi": "477043333",
                    "name": "MT JADE FORTUNE",
                    "vessel_type": "tanker",
                    "flag": "Hong Kong",
                    "imo": "9443333",
                    "callsign": "VRJF3",
                    "length_m": 183,
                    "beam_m": 32,
                    "draft_m": 11.0,
                    "cargo": "Crude Oil",
                    "track": generate_vessel_track(
                        start_lat=13.20, start_lon=80.35,
                        start_time=base_time, num_points=20,
                        speed_knots=10, heading=210, speed_variance=3.5,
                        heading_variance=12
                    ),
                },
                # Regular traffic — container
                {
                    "mmsi": "538044444",
                    "name": "MAERSK CHENNAI",
                    "vessel_type": "container",
                    "flag": "Marshall Islands",
                    "imo": "9444444",
                    "callsign": "V7MC4",
                    "length_m": 300,
                    "beam_m": 42,
                    "draft_m": 13.0,
                    "cargo": "Containers",
                    "track": generate_vessel_track(
                        start_lat=13.05, start_lon=80.50,
                        start_time=base_time, num_points=20,
                        speed_knots=17, heading=260, speed_variance=0.3,
                        heading_variance=1
                    ),
                },
                # Regular traffic — bulk
                {
                    "mmsi": "419045555",
                    "name": "MV BHARATI",
                    "vessel_type": "bulk",
                    "flag": "India",
                    "imo": "9445555",
                    "callsign": "VTBR5",
                    "length_m": 190,
                    "beam_m": 30,
                    "draft_m": 10.5,
                    "cargo": "Coal",
                    "track": generate_vessel_track(
                        start_lat=13.00, start_lon=80.10,
                        start_time=base_time, num_points=20,
                        speed_knots=12, heading=10, speed_variance=0.5,
                        heading_variance=3
                    ),
                },
                # Fishing vessel 1
                {
                    "mmsi": "419046666",
                    "name": "FV LAKSHMI",
                    "vessel_type": "fishing",
                    "flag": "India",
                    "imo": "",
                    "callsign": "VTLK6",
                    "length_m": 15,
                    "beam_m": 4,
                    "draft_m": 1.8,
                    "cargo": "Fish",
                    "track": generate_vessel_track(
                        start_lat=13.15, start_lon=80.35,
                        start_time=base_time, num_points=20,
                        speed_knots=4, heading=120, speed_variance=2.0,
                        heading_variance=35
                    ),
                },
                # Fishing vessel 2
                {
                    "mmsi": "419047777",
                    "name": "FV NILA",
                    "vessel_type": "fishing",
                    "flag": "India",
                    "imo": "",
                    "callsign": "VTNL7",
                    "length_m": 20,
                    "beam_m": 5,
                    "draft_m": 2.2,
                    "cargo": "Fish",
                    "track": generate_vessel_track(
                        start_lat=13.08, start_lon=80.40,
                        start_time=base_time, num_points=20,
                        speed_knots=3, heading=300, speed_variance=1.5,
                        heading_variance=45
                    ),
                },
                # Passenger vessel
                {
                    "mmsi": "419048888",
                    "name": "MV COASTAL QUEEN",
                    "vessel_type": "passenger",
                    "flag": "India",
                    "imo": "9448888",
                    "callsign": "VTCQ8",
                    "length_m": 72,
                    "beam_m": 14,
                    "draft_m": 3.5,
                    "cargo": "Passengers",
                    "track": generate_vessel_track(
                        start_lat=13.30, start_lon=80.30,
                        start_time=base_time, num_points=20,
                        speed_knots=14, heading=185, speed_variance=0.5,
                        heading_variance=3
                    ),
                },
                # Tug
                {
                    "mmsi": "419049999",
                    "name": "TUG VEER",
                    "vessel_type": "tug",
                    "flag": "India",
                    "imo": "",
                    "callsign": "VTVR9",
                    "length_m": 32,
                    "beam_m": 10,
                    "draft_m": 3.8,
                    "cargo": "Supply",
                    "track": generate_vessel_track(
                        start_lat=13.10, start_lon=80.25,
                        start_time=base_time, num_points=20,
                        speed_knots=7, heading=80, speed_variance=2.0,
                        heading_variance=20
                    ),
                },
                # Another cargo
                {
                    "mmsi": "636050000",
                    "name": "MV LIBERTY GRACE",
                    "vessel_type": "cargo",
                    "flag": "Liberia",
                    "imo": "9450000",
                    "callsign": "A8LG0",
                    "length_m": 170,
                    "beam_m": 27,
                    "draft_m": 9.0,
                    "cargo": "Steel Products",
                    "track": generate_vessel_track(
                        start_lat=12.95, start_lon=80.45,
                        start_time=base_time, num_points=20,
                        speed_knots=11, heading=350, speed_variance=0.5,
                        heading_variance=2
                    ),
                },
                # Product tanker
                {
                    "mmsi": "352051111",
                    "name": "MT GULF PEARL",
                    "vessel_type": "tanker",
                    "flag": "Panama",
                    "imo": "9451111",
                    "callsign": "3FGP1",
                    "length_m": 160,
                    "beam_m": 25,
                    "draft_m": 9.5,
                    "cargo": "Clean Products",
                    "track": generate_vessel_track(
                        start_lat=13.35, start_lon=80.15,
                        start_time=base_time, num_points=20,
                        speed_knots=12, heading=165, speed_variance=0.4,
                        heading_variance=3
                    ),
                },
                # LPG carrier
                {
                    "mmsi": "419052222",
                    "name": "MV GAS CARRIER I",
                    "vessel_type": "tanker",
                    "flag": "India",
                    "imo": "9452222",
                    "callsign": "VTGC2",
                    "length_m": 145,
                    "beam_m": 23,
                    "draft_m": 8.0,
                    "cargo": "LPG",
                    "track": generate_vessel_track(
                        start_lat=13.22, start_lon=80.40,
                        start_time=base_time, num_points=20,
                        speed_knots=13, heading=230, speed_variance=0.3,
                        heading_variance=2
                    ),
                },
            ],
        },
    }

    return scenario


def main():
    """Generate all test scenario JSON files."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    scenarios = [
        ("scenario_major_spill.json", scenario_major_spill()),
        ("scenario_no_spill.json", scenario_no_spill()),
        ("scenario_minor_spill.json", scenario_minor_spill()),
    ]

    for filename, data in scenarios:
        filepath = os.path.join(OUTPUT_DIR, filename)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"[OK] Generated: {filepath}")
        print(f"     {data['scenario_name']}")
        print(f"     {data['description'][:80]}...")
        print()

    print(f"[DONE] All {len(scenarios)} test scenarios generated in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
