"""
MarineGuard — Fast Synthetic SAR Dataset Generator
Generates a small dataset suitable for CPU training in ~10 seconds.

Outputs:
    data/sar_dataset/images/XXXXX.png       — 128x128 SAR images
    data/sar_dataset/masks/XXXXX.png        — binary masks
    data/sar_dataset/dataset_metadata.csv   — metadata
    data/sar_dataset/samples/XXXXX.png      — side-by-side samples for slides
"""

import os
import csv
import json
import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm

# ── Paths ──
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "sar_dataset")
IMG_DIR = os.path.join(OUTPUT_DIR, "images")
MASK_DIR = os.path.join(OUTPUT_DIR, "masks")
SAMPLE_DIR = os.path.join(OUTPUT_DIR, "samples")
CSV_PATH = os.path.join(OUTPUT_DIR, "dataset_metadata.csv")

os.makedirs(IMG_DIR, exist_ok=True)
os.makedirs(MASK_DIR, exist_ok=True)
os.makedirs(SAMPLE_DIR, exist_ok=True)

# ── Config (smaller & faster) ──
IMG_SIZE = 128
NUM_POSITIVE_PER_TYPE = 100    # 2 base sets × 100 = 200 positives
NUM_NEGATIVE = 150             # 150 negatives  →  350 total
SAMPLE_VIS_COUNT = 12

PIXEL_AREA_KM2 = 0.0001        # 10 m per pixel


# ── Slick drawing ──
def _apply_slick(img: np.ndarray, params: dict):
    H, W = img.shape
    H, W = int(H), int(W)
    cr = int(params["center_row"]); cc = int(params["center_col"])
    radius = int(params["radius"]); elongation = float(params["elongation"])
    angle = float(params["angle"]); dampening = float(params["dampening"])

    y, x = np.ogrid[:H, :W]
    cos_a = np.cos(np.radians(angle)); sin_a = np.sin(np.radians(angle))
    x_rot = (x - cc) * cos_a + (y - cr) * sin_a
    y_rot = -(x - cc) * sin_a + (y - cr) * cos_a

    dist = np.sqrt((x_rot / (radius * elongation)) ** 2 + (y_rot / radius) ** 2)
    gaussian = np.exp(-3.0 * dist ** 2)
    dampening_mask = gaussian * dampening

    img = img * (1.0 - dampening_mask)
    mask = (dist < 1.0).astype(np.float32)

    meta = {
        "slick_center_row": int(cr), "slick_center_col": int(cc),
        "slick_radius": int(radius), "slick_elongation": float(elongation),
        "slick_angle": float(angle), "slick_dampening": float(dampening),
        "slick_area_pixels": int(mask.sum()),
        "slick_area_km2": round(float(mask.sum()) * PIXEL_AREA_KM2, 4),
    }
    return img, mask, meta


def generate_sample(height=IMG_SIZE, width=IMG_SIZE, has_slick=True,
                    slick_params=None, seed=0):
    rng = np.random.RandomState(seed)

    # ── FIX: keyword args + int cast (avoids "integer is required" TypeError) ──
    image = rng.rayleigh(scale=0.5, size=(int(height), int(width))).astype(np.float32)
    image = np.clip(image / (image.max() + 1e-9), 0.0, 1.0)

    mask = np.zeros((int(height), int(width)), dtype=np.float32)
    meta = {"has_slick": int(has_slick)}

    if has_slick and slick_params is not None:
        params = dict(slick_params)
        params["center_row"] = int(rng.uniform(0.25, 0.75) * height)
        params["center_col"] = int(rng.uniform(0.25, 0.75) * width)
        params["radius"] = int(rng.uniform(0.5, 1.5) * params.get("radius", 25))
        params["angle"] = float(rng.uniform(0, 180))
        params["elongation"] = float(rng.uniform(1.0, 2.5))
        params["dampening"] = float(rng.uniform(0.4, 0.9))
        image, mask, m = _apply_slick(image, params)
        meta.update(m)
    else:
        meta.update({
            "slick_center_row": -1, "slick_center_col": -1,
            "slick_radius": -1, "slick_elongation": -1,
            "slick_angle": -1, "slick_dampening": -1,
            "slick_area_pixels": 0, "slick_area_km2": 0.0,
        })

    # ── FIX: keyword args + int cast ──
    noise = rng.normal(loc=0.0, scale=0.04, size=(int(height), int(width))).astype(np.float32)
    image = np.clip(image + noise, 0.0, 1.0)

    return (image * 255).astype(np.uint8), (mask * 255).astype(np.uint8), meta


def load_scenario_slick_params():
    scenarios_dir = os.path.join(SCRIPT_DIR, "test_scenarios")
    base_params = []
    for fname in ("scenario_major_spill.json", "scenario_minor_spill.json"):
        fpath = os.path.join(scenarios_dir, fname)
        if not os.path.exists(fpath):
            continue
        with open(fpath) as f:
            sc = json.load(f)
        sp = sc.get("sar_data", {}).get("slick_params", {})
        if sp:
            base_params.append(sp)
    if not base_params:
        base_params = [{"radius": 25, "elongation": 1.8, "angle": 45, "dampening": 0.6}]
    return base_params


def save_side_by_side(img_arr, mask_arr, path):
    h, w = img_arr.shape
    canvas = Image.new("RGB", (w * 2 + 10, h), color=(30, 30, 30))
    canvas.paste(Image.fromarray(img_arr).convert("RGB"), (0, 0))
    canvas.paste(Image.fromarray(mask_arr).convert("RGB"), (w + 10, 0))
    draw = ImageDraw.Draw(canvas)
    draw.text((5, 5), "SAR Image", fill=(0, 255, 200))
    draw.text((w + 15, 5), "Ground Truth Mask", fill=(0, 255, 200))
    canvas.save(path)


def main():
    base_params = load_scenario_slick_params()
    print(f"Loaded {len(base_params)} base slick parameter sets.")

    rows = []
    idx = 0
    sample_counter = 0
    total = (NUM_POSITIVE_PER_TYPE * len(base_params)) + NUM_NEGATIVE

    with tqdm(total=total, desc="Generating dataset") as pbar:
        # ── Positives ──
        for base in base_params:
            for _ in range(NUM_POSITIVE_PER_TYPE):
                # FIX: use keyword args (was passing base as width)
                img, mask, meta = generate_sample(
                    has_slick=True, slick_params=base, seed=idx
                )
                Image.fromarray(img).save(os.path.join(IMG_DIR, f"{idx:05d}.png"))
                Image.fromarray(mask).save(os.path.join(MASK_DIR, f"{idx:05d}.png"))
                rows.append({"image_file": f"{idx:05d}.png",
                             "mask_file": f"{idx:05d}.png", **meta})
                if sample_counter < SAMPLE_VIS_COUNT:
                    save_side_by_side(
                        img, mask,
                        os.path.join(SAMPLE_DIR, f"sample_{sample_counter:02d}.png")
                    )
                    sample_counter += 1
                idx += 1
                pbar.update(1)

        # ── Negatives ──
        for _ in range(NUM_NEGATIVE):
            # FIX: use keyword args
            img, mask, meta = generate_sample(has_slick=False, seed=idx)
            Image.fromarray(img).save(os.path.join(IMG_DIR, f"{idx:05d}.png"))
            Image.fromarray(mask).save(os.path.join(MASK_DIR, f"{idx:05d}.png"))
            rows.append({"image_file": f"{idx:05d}.png",
                         "mask_file": f"{idx:05d}.png", **meta})
            if sample_counter < SAMPLE_VIS_COUNT:
                save_side_by_side(
                    img, mask,
                    os.path.join(SAMPLE_DIR, f"sample_{sample_counter:02d}.png")
                )
                sample_counter += 1
            idx += 1
            pbar.update(1)

    # ── CSV ──
    fieldnames = ["image_file", "mask_file", "has_slick",
                  "slick_area_pixels", "slick_area_km2",
                  "slick_center_row", "slick_center_col",
                  "slick_radius", "slick_elongation",
                  "slick_angle", "slick_dampening"]
    with open(CSV_PATH, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    print(f"\nDone. {idx} samples")
    print(f"  Images  -> {IMG_DIR}")
    print(f"  Masks   -> {MASK_DIR}")
    print(f"  CSV     -> {CSV_PATH}")
    print(f"  Samples -> {SAMPLE_DIR}")


if __name__ == "__main__":
    main()