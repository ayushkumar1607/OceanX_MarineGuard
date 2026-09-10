"""
MarineGuard — Quick Pipeline Verification Script
Runs the full pipeline on all 3 test scenarios and prints results.
"""

import sys
import os
import io

# Fix Windows console encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')



import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from engines.detection_engine import DetectionEngine
from engines.drift_engine import DriftEngine
from engines.ais_engine import AISEngine
from engines.attribution_engine import AttributionEngine
from utils.data_loader import load_scenario
from config import TEST_SCENARIOS_DIR


def run_scenario(scenario_path, scenario_name):
    print(f"\n{'='*70}")
    print(f"  SCENARIO: {scenario_name}")
    print(f"{'='*70}")

    start = time.time()
    data = load_scenario(scenario_path)

    # 1. Detection
    det_engine = DetectionEngine()
    detection = det_engine.detect(data)
    print(f"\n[1] DETECTION ENGINE")
    print(f"    Detected: {detection['detected']}")
    print(f"    Confidence: {detection['confidence']:.2%}")
    print(f"    Area: {detection['area_km2']:.3f} km2")
    print(f"    Type: {detection['slick_type']}")
    print(f"    Location: {detection['centroid_lat']:.4f}N, {detection['centroid_lon']:.4f}E")
    print(f"    Age: {detection['estimated_age_hours']:.1f} hours")

    if not detection['detected']:
        print(f"\n[RESULT] No spill detected. Pipeline stops here.")
        print(f"    Explanation: {detection['explanation']}")
        print(f"    Processing time: {time.time()-start:.2f}s")
        return

    # 2. Drift
    drift_engine = DriftEngine()
    env_data = data.get("environmental_data", {})
    origin = drift_engine.trace(detection, env_data)
    print(f"\n[2] DRIFT ENGINE")
    print(f"    Origin: {origin['origin_lat']:.4f}N, {origin['origin_lon']:.4f}E")
    print(f"    Drift distance: {origin['drift_distance_km']:.1f} km")
    print(f"    Drift duration: {origin['drift_duration_hours']:.1f} hours")
    print(f"    Uncertainty: {origin['uncertainty_radius_km']:.1f} km")
    print(f"    Release window: {origin['release_time_earliest']} to {origin['release_time_latest']}")

    # 3. AIS
    ais_engine = AISEngine()
    ais_data = data.get("ais_data", {})
    ais_results = ais_engine.correlate(origin, ais_data)
    print(f"\n[3] AIS ENGINE")
    print(f"    Total vessels: {ais_results['total_vessels_in_data']}")
    print(f"    After spatial filter: {ais_results['spatially_filtered']}")
    print(f"    After temporal filter: {ais_results['temporally_filtered']}")

    # 4. Attribution
    attr_engine = AttributionEngine()
    attribution = attr_engine.attribute(origin, ais_results)
    print(f"\n[4] ATTRIBUTION ENGINE")
    print(f"    Vessels scored: {attribution['total_scored']}")
    print(f"    Conclusion: {attribution['conclusion']}")

    ranked = attribution.get("ranked_vessels", [])
    print(f"\n    RANKED VESSELS:")
    print(f"    {'Rank':<6}{'Name':<22}{'Type':<12}{'Score':<10}{'Risk':<10}")
    print(f"    {'-'*60}")
    for v in ranked:
        print(f"    #{v['rank']:<5}{v['name']:<22}{v['vessel_type']:<12}{v['attribution_score']:.2%}{'':>4}{v['risk_level']}")

    if ranked:
        top = ranked[0]
        print(f"\n    TOP VESSEL EVIDENCE ({top['name']}):")
        for ev in top.get('evidence', []):
            print(f"      {ev}")
        if top.get('anomalies'):
            print(f"    ANOMALIES:")
            for a in top['anomalies']:
                print(f"      - {a}")
        gaps = top.get('ais_gaps', [])
        if gaps:
            print(f"    AIS GAPS ({len(gaps)}):")
            for g in gaps:
                print(f"      - {g['duration_min']:.0f} min gap, {g['distance_km']:.1f} km traveled")

    print(f"\n    Processing time: {time.time()-start:.2f}s")


def main():
    scenarios_dir = TEST_SCENARIOS_DIR
    scenario_files = [
        ("scenario_major_spill.json", "Major Oil Spill - Arabian Sea"),
        ("scenario_no_spill.json", "No Spill - SAR False Positive"),
        ("scenario_minor_spill.json", "Minor Spill - Chennai Port"),
    ]

    print("="*70)
    print("  MARINEGUARD - Pipeline Verification")
    print("  Running all 3 test scenarios...")
    print("="*70)

    for filename, name in scenario_files:
        filepath = os.path.join(scenarios_dir, filename)
        if os.path.exists(filepath):
            run_scenario(filepath, name)
        else:
            print(f"\n[ERROR] Scenario not found: {filepath}")

    print(f"\n{'='*70}")
    print("  ALL SCENARIOS COMPLETE")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
