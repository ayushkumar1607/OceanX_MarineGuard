"""
MarineGuard — Attribution Engine
Multi-factor evidence scoring + ranking + explainable "Why Ranked?" panel.

Uses a trained Gradient Boosting classifier to learn which factor
combinations indicate a true culprit — replacing hand-tuned weights.
The individual factor scores are still rule-based and shown for
explainability.
"""

import os
import math
import joblib
import numpy as np
from typing import Dict, Any, List
from datetime import datetime

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.geo_utils import haversine_km
from config import (
    SCORING_WEIGHTS,
    ATTRIBUTION_HIGH_CONFIDENCE,
    ATTRIBUTION_MEDIUM_CONFIDENCE,
    ATTRIBUTION_LOW_CONFIDENCE,
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ATTR_MODEL_PATH = os.path.join(BASE_DIR, "models", "attribution_model.pkl")


class AttributionEngine:
    def __init__(self, weights: Dict[str, float] = None):
        self.weights = weights or SCORING_WEIGHTS

        self.ml_model = None
        self.ml_features = None
        if os.path.exists(ATTR_MODEL_PATH):
            try:
                bundle = joblib.load(ATTR_MODEL_PATH)
                self.ml_model = bundle["model"]
                self.ml_features = bundle["features"]
                print(f"[AttributionEngine] Loaded ML attribution model from {ATTR_MODEL_PATH}")
            except Exception as e:
                print(f"[AttributionEngine] Failed to load attribution model: {e}")
        else:
            print(f"[AttributionEngine] No ML attribution model found. "
                  f"Run: python train_attribution.py")

    # ── Factor scores (rule-based, kept for explainability) ──
    def score_proximity(self, vessel, origin_lat, origin_lon, uncertainty_km):
        min_dist = vessel.get("min_distance_to_origin_km", float("inf"))
        if min_dist <= uncertainty_km:
            score = 1.0 - (min_dist / uncertainty_km) * 0.2
        elif min_dist <= uncertainty_km * 2:
            n = (min_dist - uncertainty_km) / uncertainty_km
            score = 0.8 * math.exp(-2 * n)
        else:
            score = max(0.05, 0.3 * math.exp(-0.05 * (min_dist - uncertainty_km * 2)))
        return {
            "score": round(min(1.0, max(0.0, score)), 4),
            "min_distance_km": round(min_dist, 2),
            "in_origin_zone": min_dist <= uncertainty_km,
            "evidence": f"Vessel passed within {min_dist:.1f} km of the estimated "
                        f"origin zone (uncertainty radius: {uncertainty_km:.1f} km).",
        }

    def score_temporal(self, vessel, release_earliest, release_latest):
        track = vessel.get("track", [])
        try:
            t_early = datetime.fromisoformat(release_earliest.replace("Z", "+00:00"))
            t_late = datetime.fromisoformat(release_latest.replace("Z", "+00:00"))
        except (ValueError, AttributeError):
            return {"score": 0.5, "evidence": "Unable to parse release window."}

        positions_in_window = 0
        total_positions = len(track)
        closest_diff = float("inf")

        for pos in track:
            try:
                pt = datetime.fromisoformat(pos.get("timestamp", "").replace("Z", "+00:00"))
                if t_early <= pt <= t_late:
                    positions_in_window += 1
                    closest_diff = 0
                else:
                    d = min(abs((pt - t_early).total_seconds()),
                            abs((pt - t_late).total_seconds())) / 3600.0
                    closest_diff = min(closest_diff, d)
            except (ValueError, AttributeError):
                continue

        if positions_in_window > 0:
            score = 0.8 + 0.2 * min(1.0, positions_in_window / max(1, total_positions * 0.3))
        elif closest_diff < 2:
            score = 0.6 * math.exp(-0.5 * closest_diff)
        else:
            score = max(0.05, 0.2 * math.exp(-0.1 * closest_diff))

        return {
            "score": round(min(1.0, max(0.0, score)), 4),
            "positions_in_window": positions_in_window,
            "closest_time_diff_hours": round(closest_diff, 2),
            "evidence": f"{positions_in_window}/{total_positions} AIS positions "
                        f"fall within the release window. Closest approach: "
                        f"{closest_diff:.1f} hours.",
        }

    def score_trajectory(self, vessel, drift_path):
        traj = vessel.get("trajectory_analysis", {})
        alignment = traj.get("heading_alignment", 0)
        course_changes = traj.get("course_changes", 0)
        score = alignment
        if course_changes <= 2:
            score = min(1.0, score * 1.1)
        elif course_changes >= 5:
            score = score * 0.8
        return {
            "score": round(min(1.0, max(0.0, score)), 4),
            "heading_alignment": round(alignment, 4),
            "course_changes": course_changes,
            "evidence": f"Trajectory alignment with drift path: {alignment:.0%}. "
                        f"Course changes: {course_changes}.",
        }

    def score_ais_gap(self, vessel, release_earliest, release_latest,
                      origin_lat, origin_lon):
        gaps = vessel.get("ais_gaps", [])
        if not gaps:
            return {"score": 0.1, "num_gaps": 0,
                    "evidence": "No AIS transponder gaps detected — vessel "
                                "maintained continuous reporting."}

        try:
            t_early = datetime.fromisoformat(release_earliest.replace("Z", "+00:00"))
            t_late = datetime.fromisoformat(release_latest.replace("Z", "+00:00"))
        except (ValueError, AttributeError):
            t_early = t_late = None

        max_score = 0.0
        suspicious = 0
        total_minutes = 0.0

        for gap in gaps:
            dur = gap.get("duration_min", 0)
            total_minutes += dur
            duration_score = min(1.0, dur / 120)
            gs = gap.get("start_pos", {})
            gd = haversine_km(origin_lat, origin_lon,
                              gs.get("lat", 0), gs.get("lon", 0))
            proximity_score = max(0, 1.0 - gd / 50)
            time_score = 0.5
            if t_early and t_late:
                try:
                    gt = datetime.fromisoformat(gap.get("start_time", "").replace("Z", "+00:00"))
                    if t_early <= gt <= t_late:
                        time_score = 1.0
                    else:
                        dh = min(abs((gt - t_early).total_seconds()),
                                 abs((gt - t_late).total_seconds())) / 3600.0
                        time_score = max(0.1, 1.0 - dh / 12)
                except (ValueError, AttributeError):
                    pass
            g_score = duration_score * 0.4 + proximity_score * 0.3 + time_score * 0.3
            max_score = max(max_score, g_score)
            if g_score > 0.5:
                suspicious += 1

        return {
            "score": round(min(1.0, max(0.0, max_score)), 4),
            "num_gaps": len(gaps),
            "suspicious_gaps": suspicious,
            "total_gap_minutes": round(total_minutes, 1),
            "evidence": f"{len(gaps)} AIS gap(s) detected "
                        f"(total: {total_minutes:.0f} min). "
                        f"{suspicious} flagged as suspicious.",
        }

    def score_behavioral(self, vessel):
        traj = vessel.get("trajectory_analysis", {})
        anomalies = []
        speed_var = traj.get("speed_variance", 0)
        loitering = traj.get("loitering_detected", False)
        course_changes = traj.get("course_changes", 0)
        avg_speed = traj.get("avg_speed_knots", 0)
        score = 0.0

        if loitering:
            score += 0.3
            anomalies.append("Loitering behavior detected")
        if speed_var > 10:
            score += 0.25
            anomalies.append(f"High speed variance ({speed_var:.1f})")
        if course_changes >= 4:
            score += 0.2
            anomalies.append(f"Frequent course changes ({course_changes})")
        vtype = vessel.get("vessel_type", "").lower()
        if vtype in ("tanker", "cargo", "bulk", "container") and avg_speed < 5:
            score += 0.15
            anomalies.append(f"Unusually low speed ({avg_speed:.1f} kn)")
        max_speed = traj.get("max_speed_knots", 0)
        if max_speed > 18 and avg_speed < 12:
            score += 0.1
            anomalies.append(f"Speed spike to {max_speed:.1f} kn")

        return {
            "score": round(min(1.0, max(0.0, score)), 4),
            "anomalies": anomalies,
            "evidence": f"{len(anomalies)} anomaly(ies): "
                        + (" | ".join(anomalies) if anomalies else "none"),
        }

    def compute_attribution(self, vessel, origin):
        origin_lat = origin["origin_lat"]
        origin_lon = origin["origin_lon"]
        unc = origin.get("uncertainty_radius_km", 15)
        re_ = origin.get("release_time_earliest", "")
        rl_ = origin.get("release_time_latest", "")
        drift = origin.get("drift_path", [])

        prox = self.score_proximity(vessel, origin_lat, origin_lon, unc)
        temp = self.score_temporal(vessel, re_, rl_)
        traj = self.score_trajectory(vessel, drift)
        gap = self.score_ais_gap(vessel, re_, rl_, origin_lat, origin_lon)
        behav = self.score_behavioral(vessel)

        factor_scores = {
            "proximity": prox["score"],
            "temporal": temp["score"],
            "trajectory": traj["score"],
            "ais_gap": gap["score"],
            "behavioral": behav["score"],
        }
        weighted = {k: round(v * self.weights[k], 4)
                    for k, v in factor_scores.items()}

        # ── ML attribution score ──
        ml_score = None
        if self.ml_model is not None:
            x = np.array([[factor_scores[f] for f in self.ml_features]])
            ml_score = float(self.ml_model.predict_proba(x)[0, 1])

        # Use ML score if available, else fall back to weighted sum
        rule_score = sum(weighted.values())
        final_score = ml_score if ml_score is not None else rule_score

        # Risk level
        if final_score >= ATTRIBUTION_HIGH_CONFIDENCE:
            risk = "CRITICAL"
        elif final_score >= ATTRIBUTION_MEDIUM_CONFIDENCE:
            risk = "HIGH"
        elif final_score >= ATTRIBUTION_LOW_CONFIDENCE:
            risk = "MEDIUM"
        elif final_score >= 0.10:
            risk = "LOW"
        else:
            risk = "NONE"

        evidence = [
            f"[P] Proximity: {prox['evidence']}",
            f"[T] Temporal: {temp['evidence']}",
            f"[J] Trajectory: {traj['evidence']}",
            f"[G] AIS Gaps: {gap['evidence']}",
            f"[B] Behavior: {behav['evidence']}",
        ]

        ml_anomaly = vessel.get("ml_anomaly", {})

        return {
            "mmsi": vessel.get("mmsi", ""),
            "name": vessel.get("name", ""),
            "vessel_type": vessel.get("vessel_type", ""),
            "flag": vessel.get("flag", ""),
            "attribution_score": round(final_score, 4),
            "rule_based_score": round(rule_score, 4),
            "ml_score": round(ml_score, 4) if ml_score is not None else None,
            "risk_level": risk,
            "factor_scores": factor_scores,
            "weighted_scores": weighted,
            "evidence": evidence,
            "anomalies": behav.get("anomalies", []),
            "track": vessel.get("track", []),
            "ais_gaps": vessel.get("ais_gaps", []),
            "ml_anomaly": ml_anomaly,
            "factor_details": {
                "proximity": prox, "temporal": temp, "trajectory": traj,
                "ais_gap": gap, "behavioral": behav,
            },
        }

    def attribute(self, origin, ais_results):
        candidates = ais_results.get("candidates", [])
        scored = [self.compute_attribution(v, origin) for v in candidates]
        scored.sort(key=lambda v: v["attribution_score"], reverse=True)
        for i, v in enumerate(scored):
            v["rank"] = i + 1

        if scored:
            top = scored[0]
            if top["attribution_score"] >= ATTRIBUTION_HIGH_CONFIDENCE:
                conclusion = (f"HIGH CONFIDENCE: {top['name']} "
                              f"(MMSI: {top['mmsi']}) is the most likely source "
                              f"with {top['attribution_score']:.0%} score.")
            elif top["attribution_score"] >= ATTRIBUTION_MEDIUM_CONFIDENCE:
                conclusion = (f"MODERATE CONFIDENCE: {top['name']} "
                              f"(MMSI: {top['mmsi']}) is the primary suspect "
                              f"with {top['attribution_score']:.0%} score.")
            else:
                conclusion = (f"LOW CONFIDENCE: No vessel shows strong "
                              f"attribution. Top candidate scores only "
                              f"{top['attribution_score']:.0%}.")
        else:
            conclusion = ("NO CANDIDATES: No vessels found in the origin zone "
                          "during the release window.")

        return {
            "ranked_vessels": scored,
            "total_scored": len(scored),
            "conclusion": conclusion,
            "scoring_weights": self.weights,
            "ml_model_loaded": self.ml_model is not None,
        }