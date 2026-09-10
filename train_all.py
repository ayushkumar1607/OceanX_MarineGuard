"""
MarineGuard — Train ALL models end-to-end
Generates data + trains U-Net, AIS anomaly, and attribution models.

Usage: python train_all.py
"""

import os
import sys
import time
import subprocess

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

STEPS = [
    ("Generate SAR dataset",
     [sys.executable, os.path.join(BASE_DIR, "data", "generate_sar_dataset.py")]),
    ("Generate ML tabular data",
     [sys.executable, os.path.join(BASE_DIR, "data", "generate_ml_training_data.py")]),
    ("Train U-Net (detection)",
     [sys.executable, os.path.join(BASE_DIR, "train_unet.py"), "--no-live-plot"]),
    ("Train AIS anomaly model",
     [sys.executable, os.path.join(BASE_DIR, "train_ais_anomaly.py")]),
    ("Train attribution model",
     [sys.executable, os.path.join(BASE_DIR, "train_attribution.py")]),
]


def main():
    print("=" * 70)
    print("  MARINEGUARD — TRAIN ALL MODELS")
    print("=" * 70)
    t0 = time.time()
    for name, cmd in STEPS:
        print(f"\n>>> {name}")
        print(f"    Command: {' '.join(cmd)}")
        result = subprocess.run(cmd, cwd=BASE_DIR)
        if result.returncode != 0:
            print(f"\n[FAILED] Step: {name}")
            sys.exit(1)
    print("\n" + "=" * 70)
    print(f"  ALL MODELS TRAINED SUCCESSFULLY in {time.time()-t0:.1f}s")
    print("=" * 70)


if __name__ == "__main__":
    main()