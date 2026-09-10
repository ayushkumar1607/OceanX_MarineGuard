"""
MarineGuard — Test Sample Generator
Generates 6 test cases as real SAR image files (PNG) + environmental
+ AIS JSON files that mimic real satellite-pipeline input.

Each case folder contains:
    sar.png               — 8-bit grayscale SAR-like image (128x128)
    environmental.json    — wind, currents, location, acquisition time
    ais.json              — historical vessel traffic

Cases:
    01_clear_ocean              — calm ocean, no spill
    02_biogenic_film            — weak irregular patches (SAR look-alike)
    03_large_slick              — big elliptical slick + culprit with AIS gap
    04_large_irregular_slick    — large slick with irregular boundary
    05_small_slick              — small compact slick (moderate confidence)
    06_thin_streak              — thin elongated streak (bilge dump)
"""

import os
import json
import math
import numpy as np
from PIL import Image
from datetime import datetime, timedelta

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_samples")
os.makedirs(OUT_DIR, exist_ok=True)
IMG_SIZE = 128


def make_sar_image(slick=None, seed=42, size=IMG_SIZE,
                   irregular=False, biogenic=False):
    """Create a synthetic SAR image (grayscale) matching training distribution."""
    rng = np.random.RandomState(seed)
    img = rng.rayleigh(scale=0.5, size=(size, size)).astype(np.float32)
    img = np.clip(img / (img.max() + 1e-9), 0.0, 1.0)

    if biogenic:
        # Several faint irregular patches — no clear ellipse
        for _ in range(6):
            pr = rng.randint(8, 20)
            cr = rng.randint(20, size - 20)
            cc = rng.randint(20, size - 20)
            y, x = np.ogrid[:size, :size]
            d = np.sqrt((x - cc) ** 2 + (y - cr) ** 2)
            patch = np.exp(-3 * (d / pr) ** 2) * rng.uniform(0.15, 0.35)
            img = img * (1 - patch)

    elif slick is not None:
        y, x = np.ogrid[:size, :size]
        cr = slick["center_row"]
        cc = slick["center_col"]
        radius = slick["radius"]
        elong = slick.get("elongation", 1.5)
        angle = math.radians(slick.get("angle", 30))
        damp = slick.get("dampening", 0.7)

        x_rot = (x - cc) * math.cos(angle) + (y - cr) * math.sin(angle)
        y_rot = -(x - cc) * math.sin(angle) + (y - cr) * math.cos(angle)
        d = np.sqrt((x_rot / (radius * elong)) ** 2 + (y_rot / radius) ** 2)

        if irregular:
            # Wavy boundary perturbation
            wave = 1.0 + 0.15 * np.sin(x * 0.3) + 0.15 * np.cos(y * 0.25)
            d = d * wave

        mask = np.exp(-3 * d ** 2) * damp
        img = img * (1 - mask)

    # Speckle noise after the slick
    noise = rng.normal(0, 0.04, (size, size)).astype(np.float32)
    img = np.clip(img + noise, 0.0, 1.0)
    return (img * 255).astype(np.uint8)


def vessel_track(start_lat, start_lon, start_iso, n, speed, heading,
                 gap_at=None, gap_len=0,
                 speed_jitter=1.0, heading_jitter=5.0, seed=0):
    """Generate a simple vessel AIS track. gap_at = index where AIS gap starts."""
    rng = np.random.RandomState(seed)
    t = datetime.fromisoformat(start_iso.replace("Z", "+00:00"))
    lat, lon = start_lat, start_lon
    track = []
    for i in range(n):
        spd = max(0.5, speed + rng.normal(0, speed_jitter))
        hdg = (heading + rng.normal(0, heading_jitter)) % 360
        cog = (hdg + rng.normal(0, 2)) % 360

        # Skip emitting during gap
        if gap_at is not None and gap_at <= i < gap_at + gap_len:
            d = spd * 0.5 * 1.852  # km in 30 min
            lat += d * math.cos(math.radians(hdg)) / 111.0
            lon += d * math.sin(math.radians(hdg)) / (111.0 * math.cos(math.radians(lat)))
            t += timedelta(minutes=30)
            continue

        track.append({
            "timestamp": t.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "lat": round(lat, 6),
            "lon": round(lon, 6),
            "speed_knots": round(spd, 1),
            "heading": round(hdg, 1),
            "course": round(cog, 1),
        })
        d = spd * 0.5 * 1.852
        lat += d * math.cos(math.radians(hdg)) / 111.0
        lon += d * math.sin(math.radians(hdg)) / (111.0 * math.cos(math.radians(lat)))
        t += timedelta(minutes=30)
    return track


def write_case(folder, env, ais, slick=None, irregular=False, biogenic=False, seed=42):
    case_dir = os.path.join(OUT_DIR, folder)
    os.makedirs(case_dir, exist_ok=True)
    img = make_sar_image(slick=slick, seed=seed,
                         irregular=irregular, biogenic=biogenic)
    Image.fromarray(img, mode="L").save(os.path.join(case_dir, "sar.png"))
    with open(os.path.join(case_dir, "environmental.json"), "w") as f:
        json.dump(env, f, indent=2)
    with open(os.path.join(case_dir, "ais.json"), "w") as f:
        json.dump({"vessels": ais}, f, indent=2)
    print(f"  [OK] {folder}")


def main():
    print("Generating test samples...\n")

    # ═════════════════════════════════════════════════════════════════
    # 1. Clear ocean — no spill
    # ═════════════════════════════════════════════════════════════════
    env = {
        "location_name": "Bay of Bengal near Chennai",
        "slick_centroid_lat": 13.5, "slick_centroid_lon": 80.8,
        "acquisition_time": "2026-09-08T16:15:00Z",
        "satellite": "Sentinel-1B", "mode": "IW", "polarization": "VV",
        "resolution_m": 10, "region_size_km": 20,
        "wind_speed_ms": 2.0, "wind_direction_deg": 90,
        "current_speed_ms": 0.15, "current_direction_deg": 160,
        "sea_state": "calm", "wave_height_m": 0.3,
        "source_wind": "ERA5 Reanalysis",
        "source_current": "CMEMS Global Ocean Physics",
    }
    ais = [
        {"mmsi": "419011111", "name": "MV COROMANDEL", "vessel_type": "cargo",
         "flag": "India", "imo": "9111111", "length_m": 155, "beam_m": 25,
         "draft_m": 8.5, "cargo": "Steel",
         "track": vessel_track(13.8, 80.5, "2026-09-08T12:00:00", 12, 13, 195, seed=1)},
        {"mmsi": "419022222", "name": "FV MARINA QUEEN", "vessel_type": "fishing",
         "flag": "India", "imo": "", "length_m": 18, "beam_m": 5,
         "draft_m": 2.0, "cargo": "Fish",
         "track": vessel_track(13.55, 80.9, "2026-09-08T13:00:00", 8, 4, 45,
                                speed_jitter=2, heading_jitter=30, seed=2)},
        {"mmsi": "563033333", "name": "MV SINGAPORE PRIDE", "vessel_type": "container",
         "flag": "Singapore", "imo": "9333333", "length_m": 294, "beam_m": 40,
         "draft_m": 12.5, "cargo": "Containers",
         "track": vessel_track(13.2, 81.2, "2026-09-08T12:00:00", 10, 19, 230, seed=3)},
    ]
    write_case("01_clear_ocean", env, ais, slick=None, seed=101)

    # ═════════════════════════════════════════════════════════════════
    # 2. Biogenic film — SAR look-alike (no clear ellipse)
    # ═════════════════════════════════════════════════════════════════
    env = {
        "location_name": "Arabian Sea — open water",
        "slick_centroid_lat": 15.0, "slick_centroid_lon": 72.0,
        "acquisition_time": "2026-09-08T10:00:00Z",
        "satellite": "Sentinel-1A", "mode": "IW", "polarization": "VV",
        "resolution_m": 10, "region_size_km": 20,
        "wind_speed_ms": 3.0, "wind_direction_deg": 180,
        "current_speed_ms": 0.20, "current_direction_deg": 140,
        "sea_state": "slight", "wave_height_m": 0.5,
        "source_wind": "ERA5 Reanalysis",
        "source_current": "CMEMS Global Ocean Physics",
    }
    ais = [
        {"mmsi": "419033333", "name": "MV ARABIAN STAR", "vessel_type": "cargo",
         "flag": "Panama", "imo": "9233333", "length_m": 180, "beam_m": 30,
         "draft_m": 10.0, "cargo": "General",
         "track": vessel_track(15.2, 72.1, "2026-09-08T06:00:00", 10, 12, 190, seed=4)},
        {"mmsi": "419044444", "name": "FV KANYA", "vessel_type": "fishing",
         "flag": "India", "imo": "", "length_m": 20, "beam_m": 5,
         "draft_m": 2.2, "cargo": "Fish",
         "track": vessel_track(14.9, 72.15, "2026-09-08T06:00:00", 8, 3, 90,
                                speed_jitter=1.5, heading_jitter=40, seed=5)},
        {"mmsi": "419055555", "name": "MT GULF SPIRIT", "vessel_type": "tanker",
         "flag": "Liberia", "imo": "9345555", "length_m": 220, "beam_m": 35,
         "draft_m": 12.0, "cargo": "Crude",
         "track": vessel_track(15.1, 71.9, "2026-09-08T06:00:00", 10, 10, 210, seed=6)},
    ]
    write_case("02_biogenic_film", env, ais, biogenic=True, seed=102)

    # ═════════════════════════════════════════════════════════════════
    # 3. Large slick — clear ellipse, culprit has AIS gap
    # ═════════════════════════════════════════════════════════════════
    env = {
        "location_name": "Arabian Sea, west of Mumbai",
        "slick_centroid_lat": 18.82, "slick_centroid_lon": 71.48,
        "acquisition_time": "2026-09-09T14:30:00Z",
        "satellite": "Sentinel-1A", "mode": "IW", "polarization": "VV",
        "resolution_m": 10, "region_size_km": 20,
        "wind_speed_ms": 7.5, "wind_direction_deg": 225,
        "current_speed_ms": 0.35, "current_direction_deg": 190,
        "sea_state": "moderate", "wave_height_m": 1.2,
        "source_wind": "ERA5 Reanalysis",
        "source_current": "CMEMS Global Ocean Physics",
    }
    slick = {"center_row": 64, "center_col": 64, "radius": 32,
             "elongation": 2.0, "angle": 35, "dampening": 0.85}
    ais = [
        {"mmsi": "419001234", "name": "MT OCEAN CARRIER", "vessel_type": "tanker",
         "flag": "Panama", "imo": "9876543", "length_m": 274, "beam_m": 48,
         "draft_m": 16.5, "cargo": "Crude Oil",
         "track": vessel_track(19.15, 71.20, "2026-09-09T06:00:00", 20, 11, 200,
                                gap_at=8, gap_len=3,
                                speed_jitter=3, heading_jitter=8, seed=7)},
        {"mmsi": "477001111", "name": "MV GLOBAL PHOENIX", "vessel_type": "cargo",
         "flag": "Hong Kong", "imo": "9812345", "length_m": 189, "beam_m": 32,
         "draft_m": 10.2, "cargo": "General Cargo",
         "track": vessel_track(19.0, 71.0, "2026-09-09T06:00:00", 20, 14, 170, seed=8)},
        {"mmsi": "352001555", "name": "CMA MUMBAI EXPRESS", "vessel_type": "container",
         "flag": "Panama", "imo": "9654321", "length_m": 336, "beam_m": 45,
         "draft_m": 14.0, "cargo": "Containers",
         "track": vessel_track(18.6, 71.8, "2026-09-09T06:00:00", 20, 18, 250, seed=9)},
        {"mmsi": "419005678", "name": "FV SAGAR RANI", "vessel_type": "fishing",
         "flag": "India", "imo": "", "length_m": 22, "beam_m": 6,
         "draft_m": 2.5, "cargo": "Fish",
         "track": vessel_track(18.9, 71.55, "2026-09-09T06:00:00", 20, 4, 90,
                                speed_jitter=2, heading_jitter=30, seed=10)},
    ]
    write_case("03_large_slick", env, ais, slick=slick, seed=103)

    # ═════════════════════════════════════════════════════════════════
    # 4. Large irregular slick
    # ═════════════════════════════════════════════════════════════════
    env = {
        "location_name": "Arabian Sea near Kochi",
        "slick_centroid_lat": 9.9, "slick_centroid_lon": 75.6,
        "acquisition_time": "2026-09-11T09:00:00Z",
        "satellite": "Sentinel-1A", "mode": "IW", "polarization": "VV",
        "resolution_m": 10, "region_size_km": 20,
        "wind_speed_ms": 6.5, "wind_direction_deg": 200,
        "current_speed_ms": 0.30, "current_direction_deg": 175,
        "sea_state": "moderate", "wave_height_m": 1.0,
        "source_wind": "ERA5 Reanalysis",
        "source_current": "CMEMS Global Ocean Physics",
    }
    slick = {"center_row": 60, "center_col": 70, "radius": 28,
             "elongation": 1.6, "angle": 60, "dampening": 0.80}
    ais = [
        {"mmsi": "419009001", "name": "MT KOCHI TRADER", "vessel_type": "tanker",
         "flag": "India", "imo": "9456789", "length_m": 200, "beam_m": 32,
         "draft_m": 12.5, "cargo": "Refined Products",
         "track": vessel_track(10.2, 75.3, "2026-09-11T04:00:00", 18, 10, 195,
                                gap_at=6, gap_len=2, speed_jitter=2.5, seed=11)},
        {"mmsi": "636020000", "name": "MT PERSIAN STAR", "vessel_type": "tanker",
         "flag": "Liberia", "imo": "9845678", "length_m": 244, "beam_m": 42,
         "draft_m": 15.0, "cargo": "Refined Products",
         "track": vessel_track(10.0, 75.8, "2026-09-11T04:00:00", 18, 12, 260, seed=12)},
        {"mmsi": "419077777", "name": "MV MALABAR", "vessel_type": "cargo",
         "flag": "India", "imo": "9567890", "length_m": 150, "beam_m": 24,
         "draft_m": 8.0, "cargo": "General",
         "track": vessel_track(9.7, 75.9, "2026-09-11T04:00:00", 18, 11, 145, seed=13)},
    ]
    write_case("04_large_irregular_slick", env, ais,
               slick=slick, irregular=True, seed=104)

    # ═════════════════════════════════════════════════════════════════
    # 5. Small slick
    # ═════════════════════════════════════════════════════════════════
    env = {
        "location_name": "Chennai port approaches",
        "slick_centroid_lat": 13.12, "slick_centroid_lon": 80.32,
        "acquisition_time": "2026-09-10T10:45:00Z",
        "satellite": "Sentinel-1A", "mode": "IW", "polarization": "VV+VH",
        "resolution_m": 10, "region_size_km": 15,
        "wind_speed_ms": 5.0, "wind_direction_deg": 180,
        "current_speed_ms": 0.25, "current_direction_deg": 210,
        "sea_state": "slight", "wave_height_m": 0.8,
        "source_wind": "ERA5 Reanalysis",
        "source_current": "CMEMS Global Ocean Physics",
    }
    slick = {"center_row": 70, "center_col": 60, "radius": 12,
             "elongation": 1.8, "angle": 60, "dampening": 0.65}
    ais = [
        {"mmsi": "419041111", "name": "MT KAVERI", "vessel_type": "tanker",
         "flag": "India", "imo": "9441111", "length_m": 135, "beam_m": 20,
         "draft_m": 7.5, "cargo": "Fuel Oil",
         "track": vessel_track(13.25, 80.20, "2026-09-10T06:00:00", 16, 9, 175,
                                gap_at=5, gap_len=2, seed=14)},
        {"mmsi": "419042222", "name": "MV EASTERN DAWN", "vessel_type": "cargo",
         "flag": "India", "imo": "9442222", "length_m": 115, "beam_m": 18,
         "draft_m": 6.0, "cargo": "Chemicals",
         "track": vessel_track(13.18, 80.28, "2026-09-10T06:00:00", 16, 8, 190,
                                speed_jitter=2, heading_jitter=10, seed=15)},
        {"mmsi": "477043333", "name": "MT JADE FORTUNE", "vessel_type": "tanker",
         "flag": "Hong Kong", "imo": "9443333", "length_m": 183, "beam_m": 32,
         "draft_m": 11.0, "cargo": "Crude Oil",
         "track": vessel_track(13.20, 80.35, "2026-09-10T06:00:00", 16, 10, 210,
                                speed_jitter=3.5, heading_jitter=12, seed=16)},
        {"mmsi": "538044444", "name": "MAERSK CHENNAI", "vessel_type": "container",
         "flag": "Marshall Islands", "imo": "9444444", "length_m": 300, "beam_m": 42,
         "draft_m": 13.0, "cargo": "Containers",
         "track": vessel_track(13.05, 80.50, "2026-09-10T06:00:00", 16, 17, 260, seed=17)},
    ]
    write_case("05_small_slick", env, ais, slick=slick, seed=105)

    # ═════════════════════════════════════════════════════════════════
    # 6. Thin streak — bilge dump look
    # ═════════════════════════════════════════════════════════════════
    env = {
        "location_name": "Bay of Bengal near Paradip",
        "slick_centroid_lat": 20.3, "slick_centroid_lon": 86.7,
        "acquisition_time": "2026-09-12T07:30:00Z",
        "satellite": "Sentinel-1B", "mode": "IW", "polarization": "VV",
        "resolution_m": 10, "region_size_km": 15,
        "wind_speed_ms": 4.5, "wind_direction_deg": 220,
        "current_speed_ms": 0.20, "current_direction_deg": 200,
        "sea_state": "slight", "wave_height_m": 0.6,
        "source_wind": "ERA5 Reanalysis",
        "source_current": "CMEMS Global Ocean Physics",
    }
    slick = {"center_row": 65, "center_col": 65, "radius": 6,
             "elongation": 5.0, "angle": 45, "dampening": 0.55}
    ais = [
        {"mmsi": "419055001", "name": "MT ODISHA TRADER", "vessel_type": "tanker",
         "flag": "India", "imo": "9650001", "length_m": 170, "beam_m": 28,
         "draft_m": 10.0, "cargo": "Fuel Oil",
         "track": vessel_track(20.6, 86.5, "2026-09-12T03:00:00", 16, 9, 200,
                                gap_at=5, gap_len=2, seed=18)},
        {"mmsi": "419066002", "name": "MV PARADIP EXPRESS", "vessel_type": "cargo",
         "flag": "India", "imo": "9650002", "length_m": 155, "beam_m": 25,
         "draft_m": 9.0, "cargo": "Iron Ore",
         "track": vessel_track(20.5, 86.9, "2026-09-12T03:00:00", 16, 12, 240, seed=19)},
        {"mmsi": "419077003", "name": "FV SEA PEARL", "vessel_type": "fishing",
         "flag": "India", "imo": "", "length_m": 22, "beam_m": 6,
         "draft_m": 2.5, "cargo": "Fish",
         "track": vessel_track(20.2, 86.8, "2026-09-12T03:00:00", 14, 4, 60,
                                speed_jitter=2, heading_jitter=35, seed=20)},
    ]
    write_case("06_thin_streak", env, ais, slick=slick, seed=106)

    print(f"\nAll 6 test samples generated in: {OUT_DIR}")
    print("Each folder contains: sar.png, environmental.json, ais.json")


if __name__ == "__main__":
    main()