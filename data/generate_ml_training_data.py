"""
MarineGuard — ML Training Data Generator
Creates labeled datasets for:
  1. AIS anomaly detection (is this vessel behaving suspiciously?)
  2. Attribution (is this vessel the true spill source?)

Outputs:
    data/ml_data/ais_features.csv          — 2000 samples
    data/ml_data/attribution_features.csv  — 3000 samples
"""

import os
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE_DIR, "ml_data")
os.makedirs(OUT_DIR, exist_ok=True)


def generate_ais_features(n_samples=2000, seed=42):
    """
    AIS anomaly dataset.
    Each row = one vessel's trajectory summary + label (0=normal, 1=suspicious).
    Suspicious vessels: have AIS gaps, high speed variance, loitering, many course changes.
    """
    rng = np.random.RandomState(seed)
    rows = []

    for _ in range(n_samples):
        suspicious = rng.rand() < 0.5

        if suspicious:
            avg_speed = rng.uniform(2, 15)
            speed_var = rng.uniform(5, 25)
            max_speed = avg_speed + rng.uniform(2, 10)
            min_speed = max(0.5, avg_speed - rng.uniform(2, 8))
            course_changes = int(rng.uniform(3, 10))
            loitering = int(rng.rand() < 0.5)
            num_gaps = int(rng.randint(1, 4))
            max_gap_dur = rng.uniform(30, 180)
            total_gap_min = num_gaps * max_gap_dur * rng.uniform(0.7, 1.0)
            heading_alignment = rng.uniform(0.0, 0.4)
        else:
            avg_speed = rng.uniform(8, 20)
            speed_var = rng.uniform(0, 5)
            max_speed = avg_speed + rng.uniform(0, 3)
            min_speed = max(0.5, avg_speed - rng.uniform(0, 3))
            course_changes = int(rng.uniform(0, 3))
            loitering = int(rng.rand() < 0.05)
            num_gaps = int(rng.choice([0, 0, 0, 1]))  # mostly zero
            max_gap_dur = rng.uniform(0, 20) if num_gaps > 0 else 0.0
            total_gap_min = num_gaps * max_gap_dur
            heading_alignment = rng.uniform(0.4, 1.0)

        rows.append({
            "avg_speed_knots": avg_speed,
            "speed_variance": speed_var,
            "max_speed_knots": max_speed,
            "min_speed_knots": min_speed,
            "course_changes": course_changes,
            "loitering": loitering,
            "num_ais_gaps": num_gaps,
            "max_gap_duration_min": max_gap_dur,
            "total_gap_minutes": total_gap_min,
            "heading_alignment": heading_alignment,
            "is_suspicious": int(suspicious),
        })
    return pd.DataFrame(rows)


def generate_attribution_features(n_samples=3000, seed=123):
    """
    Attribution dataset.
    Simulates scenarios with 1 culprit + N innocents.
    Culprit: high scores on multiple factors.
    Innocent: lower / random scores.
    """
    rng = np.random.RandomState(seed)
    rows = []

    for _ in range(n_samples):
        n_vessels = rng.randint(3, 10)
        culprit_idx = rng.randint(0, n_vessels)

        for v in range(n_vessels):
            is_culprit = (v == culprit_idx)

            if is_culprit:
                proximity = rng.uniform(0.7, 1.0)
                temporal = rng.uniform(0.6, 1.0)
                trajectory = rng.uniform(0.5, 1.0)
                ais_gap = rng.uniform(0.5, 1.0)
                behavioral = rng.uniform(0.4, 1.0)
                label = 1
            else:
                proximity = rng.uniform(0.0, 0.8)
                temporal = rng.uniform(0.0, 0.7)
                trajectory = rng.uniform(0.0, 0.6)
                ais_gap = rng.uniform(0.0, 0.4)
                behavioral = rng.uniform(0.0, 0.5)
                label = 0

            rows.append({
                "proximity": proximity,
                "temporal": temporal,
                "trajectory": trajectory,
                "ais_gap": ais_gap,
                "behavioral": behavioral,
                "is_culprit": label,
            })
    return pd.DataFrame(rows)


def main():
    ais_df = generate_ais_features(2000)
    ais_path = os.path.join(OUT_DIR, "ais_features.csv")
    ais_df.to_csv(ais_path, index=False)
    print(f"[OK] AIS features -> {ais_path} ({len(ais_df)} rows)")
    print(f"     Class balance: {ais_df['is_suspicious'].value_counts().to_dict()}")

    attr_df = generate_attribution_features(3000)
    attr_path = os.path.join(OUT_DIR, "attribution_features.csv")
    attr_df.to_csv(attr_path, index=False)
    print(f"[OK] Attribution features -> {attr_path} ({len(attr_df)} rows)")
    print(f"     Class balance: {attr_df['is_culprit'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()