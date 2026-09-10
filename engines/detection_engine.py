"""
MarineGuard — Detection Engine
Accepts either a raw SAR image array (from upload) OR synthetic parameters.

- Real image path (via sar_data["image_array"]): runs the trained U-Net
  directly on the image, no synthetic generation.
- Synthetic path (existing): generates SAR image from slick_params.
"""

import os
import numpy as np
from typing import Dict, Any, Optional

import torch
import torch.nn.functional as F

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.unet import UNet
from utils.geo_utils import create_ellipse_polygon
from config import DETECTION_CONFIDENCE_THRESHOLD, MIN_SLICK_AREA_KM2

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEIGHTS_PATH = os.path.join(BASE_DIR, "models", "unet_weights.pth")
MODEL_INPUT_SIZE = 128


class DetectionEngine:
    def __init__(self):
        self.device = torch.device("cpu")
        self.model = UNet(in_channels=1, out_channels=1).to(self.device)

        self.has_trained_weights = False
        if os.path.exists(WEIGHTS_PATH):
            try:
                state = torch.load(WEIGHTS_PATH, map_location=self.device)
                self.model.load_state_dict(state)
                self.has_trained_weights = True
                print(f"[DetectionEngine] Loaded trained weights from {WEIGHTS_PATH}")
            except Exception as e:
                print(f"[DetectionEngine] Failed to load weights: {e}")
        else:
            print(f"[DetectionEngine] No trained weights found. "
                  f"Run: python train_unet.py")

        self.model.eval()
        self.confidence_threshold = DETECTION_CONFIDENCE_THRESHOLD
        self.min_area_km2 = MIN_SLICK_AREA_KM2

    def preprocess_sar(self, sar_data: Dict[str, Any]) -> np.ndarray:
        """Use provided image array if present; else generate synthetic."""
        # ── Real image case ──
        if "image_array" in sar_data:
            img = sar_data["image_array"]
            if img.ndim == 3:
                img = img.mean(axis=2)
            img = img.astype(np.float32)
            if img.max() > 1.5:
                img = img / 255.0
            return img

        # ── Synthetic case (existing behaviour) ──
        height = sar_data.get("image_height", 512)
        width = sar_data.get("image_width", 512)

        np.random.seed(sar_data.get("seed", 42))
        ocean = np.random.rayleigh(scale=0.5, size=(int(height), int(width))).astype(np.float32)
        ocean = np.clip(ocean / (ocean.max() + 1e-9), 0, 1)

        if sar_data.get("has_slick", False):
            slick = sar_data.get("slick_params", {})
            cr = int(slick.get("center_row", height // 2))
            cc = int(slick.get("center_col", width // 2))
            radius = int(slick.get("radius", 50))
            elong = float(slick.get("elongation", 1.5))
            angle = float(slick.get("angle", 30))
            damp = float(slick.get("dampening", 0.3))

            y, x = np.ogrid[:height, :width]
            cos_a = np.cos(np.radians(angle))
            sin_a = np.sin(np.radians(angle))
            x_rot = (x - cc) * cos_a + (y - cr) * sin_a
            y_rot = -(x - cc) * sin_a + (y - cr) * cos_a
            dist = np.sqrt((x_rot / (radius * elong)) ** 2 + (y_rot / radius) ** 2)
            damp_mask = np.exp(-3 * dist ** 2) * damp
            ocean = ocean * (1 - damp_mask)

        return ocean.astype(np.float32)

    def run_segmentation(self, preprocessed: np.ndarray,
                         slick_params: Optional[Dict[str, Any]] = None) -> np.ndarray:
        """U-Net inference. Falls back to synthetic mask if no weights."""
        if not self.has_trained_weights:
            return self.model.simulate_inference(
                sar_shape=preprocessed.shape,
                slick_params=slick_params,
            )

        H, W = preprocessed.shape
        img = torch.from_numpy(preprocessed).float().unsqueeze(0).unsqueeze(0)
        img_resized = F.interpolate(
            img, size=(MODEL_INPUT_SIZE, MODEL_INPUT_SIZE),
            mode="bilinear", align_corners=False
        ).to(self.device)

        with torch.no_grad():
            prob_resized = self.model(img_resized)

        prob = F.interpolate(
            prob_resized, size=(H, W),
            mode="bilinear", align_corners=False
        ).squeeze().cpu().numpy()

        return prob.astype(np.float32)

    def extract_slick_properties(self, prob_map: np.ndarray,
                                  sar_metadata: Dict[str, Any]) -> Dict[str, Any]:
        binary = prob_map > self.confidence_threshold

        if binary.any():
            confidence = float(np.mean(prob_map[binary]))
            slick_pixels = int(binary.sum())
        else:
            confidence = float(np.max(prob_map))
            slick_pixels = 0

        resolution_m = sar_metadata.get("resolution_m", 10)
        pixel_area_km2 = (resolution_m ** 2) / 1e6
        area_km2 = slick_pixels * pixel_area_km2

        centroid_lat = sar_metadata.get("slick_centroid_lat", 0)
        centroid_lon = sar_metadata.get("slick_centroid_lon", 0)

        if area_km2 > 0:
            semi_major = np.sqrt(area_km2 / np.pi) * 1.5
            semi_minor = np.sqrt(area_km2 / np.pi) * 0.8
            rotation = sar_metadata.get("slick_orientation", 45)
            polygon = create_ellipse_polygon(
                centroid_lat, centroid_lon,
                semi_major, semi_minor,
                rotation_deg=rotation, num_points=48,
            )
        else:
            polygon = []

        return {
            "confidence": round(confidence, 4),
            "area_km2": round(area_km2, 3),
            "centroid_lat": centroid_lat,
            "centroid_lon": centroid_lon,
            "polygon": polygon,
            "slick_pixels": slick_pixels,
        }

    def classify_detection(self, confidence: float, area_km2: float) -> Dict[str, Any]:
        if confidence < 0.40:
            return {
                "detected": False, "slick_type": "look-alike",
                "explanation": (
                    f"Low confidence ({confidence:.0%}). This SAR anomaly is "
                    "likely a look-alike (calm water / biogenic film), not a "
                    "petroleum slick."
                ),
            }
        elif confidence < self.confidence_threshold:
            return {
                "detected": True, "slick_type": "possible_oil",
                "explanation": (
                    f"Moderate confidence ({confidence:.0%}). Possible oil "
                    "slick. Proceed with drift analysis."
                ),
            }
        elif area_km2 < self.min_area_km2:
            return {
                "detected": True, "slick_type": "minor_oil",
                "explanation": (
                    f"High confidence ({confidence:.0%}) but very small area "
                    f"({area_km2:.3f} km²) — likely a minor discharge or bilge dump."
                ),
            }
        else:
            return {
                "detected": True, "slick_type": "oil",
                "explanation": (
                    f"High confidence oil slick ({confidence:.0%}). "
                    f"Area: {area_km2:.2f} km²."
                ),
            }

    def estimate_age(self, area_km2: float, wind_speed_ms: float = 5.0) -> float:
        base_age = np.sqrt(area_km2) * 4
        wind_factor = 1 + (wind_speed_ms - 5) * 0.1
        return round(max(1, base_age * wind_factor), 1)

    def detect(self, scenario_data: Dict[str, Any]) -> Dict[str, Any]:
        sar_data = scenario_data.get("sar_data", {})
        sar_metadata = scenario_data.get("sar_metadata", {})
        environmental = scenario_data.get("environmental_data", {})

        has_real_image = "image_array" in sar_data
        preprocessed = self.preprocess_sar(sar_data)
        slick_params = sar_data.get("slick_params", None)
        prob_map = self.run_segmentation(preprocessed, slick_params)
        properties = self.extract_slick_properties(prob_map, sar_metadata)

        # Synthetic-only fallback: honour scenario-declared values
        if (not self.has_trained_weights and not has_real_image and slick_params):
            if "confidence" in slick_params:
                properties["confidence"] = float(slick_params["confidence"])
            if "target_area_km2" in slick_params:
                properties["area_km2"] = float(slick_params["target_area_km2"])

        classification = self.classify_detection(
            properties["confidence"], properties["area_km2"])
        wind_speed = environmental.get("wind_speed_ms", 5.0)
        age_hours = self.estimate_age(properties["area_km2"], wind_speed)
        det_time = sar_metadata.get("acquisition_time", "2026-01-01T00:00:00Z")

        return {
            "detected": classification["detected"],
            "confidence": properties["confidence"],
            "centroid_lat": properties["centroid_lat"],
            "centroid_lon": properties["centroid_lon"],
            "area_km2": properties["area_km2"],
            "polygon": properties["polygon"],
            "estimated_age_hours": age_hours,
            "slick_type": classification["slick_type"],
            "detection_time": det_time,
            "explanation": classification["explanation"],
            "model_info": self.model.get_architecture_summary(),
            "weights_loaded": self.has_trained_weights,
            "sar_metadata": {
                "satellite": sar_metadata.get("satellite", "Sentinel-1A"),
                "mode": sar_metadata.get("mode", "IW"),
                "polarization": sar_metadata.get("polarization", "VV"),
                "resolution_m": sar_metadata.get("resolution_m", 10),
                "acquisition_time": det_time,
            },
        }