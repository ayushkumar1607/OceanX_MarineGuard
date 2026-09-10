"""
MarineGuard — Fast CPU U-Net Training
Optimized for laptops: 128x128 images, lightweight model, ~1-2 min total.

Usage:
    python train_unet.py
    python train_unet.py --epochs 10
    python train_unet.py --no-live-plot     # headless
    python train_unet.py --resume
"""

import os
# ── Kill torch's lazy compile/dynamo imports (they pull in sympy = slow on Windows) ──
os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ["TORCHINDUCTOR_DISABLE"] = "1"
os.environ.setdefault("OMP_NUM_THREADS", str(os.cpu_count() or 4))

import csv
import json
import time
import argparse
import numpy as np
import torch

# ── Pre-warm torch internals before training (avoids stall mid-epoch) ──
print("Warming up torch internals (first time only, ~10-40s)...")
try:
    import torch.optim  # noqa
    _ = torch.optim.Adam([torch.zeros(1, requires_grad=True)], lr=1e-3)
    print("Warmup done.\n")
except Exception as e:
    print(f"Warmup warning (ignored): {e}\n")

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
from tqdm import tqdm

import matplotlib
import matplotlib.pyplot as plt

from models.unet import UNet


# ── Paths ──
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMG_DIR = os.path.join(BASE_DIR, "data", "sar_dataset", "images")
MASK_DIR = os.path.join(BASE_DIR, "data", "sar_dataset", "masks")
WEIGHTS_DIR = os.path.join(BASE_DIR, "models")
WEIGHTS_PATH = os.path.join(WEIGHTS_DIR, "unet_weights.pth")
CHECKPOINT_PATH = os.path.join(WEIGHTS_DIR, "unet_checkpoint.pth")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
LIVE_DIR = os.path.join(REPORTS_DIR, "live")
os.makedirs(WEIGHTS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)
os.makedirs(LIVE_DIR, exist_ok=True)

HISTORY_CSV = os.path.join(REPORTS_DIR, "training_history.csv")
CURVES_PNG = os.path.join(REPORTS_DIR, "training_curves.png")
SUMMARY_JSON = os.path.join(REPORTS_DIR, "training_summary.json")

# ── Fast CPU hyperparameters ──
BATCH_SIZE = 16
EPOCHS = 8
LR = 1e-3
VAL_SPLIT = 0.15
SEED = 42
THRESHOLD = 0.5
CHECKPOINT_EVERY = 4
NUM_WORKERS = 0   # keep 0 on Windows

torch.set_num_threads(os.cpu_count() or 4)


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────
class SARDataset(Dataset):
    def __init__(self, img_dir, mask_dir, augment=False):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.files = sorted(f for f in os.listdir(img_dir) if f.endswith(".png"))
        self.augment = augment

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        name = self.files[idx]
        img = Image.open(os.path.join(self.img_dir, name)).convert("L")
        mask = Image.open(os.path.join(self.mask_dir, name)).convert("L")
        img = np.array(img, dtype=np.float32) / 255.0
        mask = np.array(mask, dtype=np.float32) / 255.0
        if self.augment and np.random.rand() < 0.5:
            img = img[:, ::-1].copy()
            mask = mask[:, ::-1].copy()
        return (torch.from_numpy(img).unsqueeze(0),
                torch.from_numpy(mask).unsqueeze(0))


# ─────────────────────────────────────────────────────────────────────────────
# Loss
# ─────────────────────────────────────────────────────────────────────────────
class BCEDiceLoss(nn.Module):
    def __init__(self, bce_weight=0.5):
        super().__init__()
        self.bce = nn.BCELoss()
        self.bce_weight = bce_weight

    def forward(self, pred, target):
        bce_loss = self.bce(pred, target)
        smooth = 1e-6
        pf = pred.view(-1); tf = target.view(-1)
        inter = (pf * tf).sum()
        dice_loss = 1 - (2.0 * inter + smooth) / (pf.sum() + tf.sum() + smooth)
        return self.bce_weight * bce_loss + (1 - self.bce_weight) * dice_loss


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────
def accumulate_confusion(pred, target, threshold=THRESHOLD):
    pb = (pred > threshold).float()
    tb = (target > 0.5).float()
    tp = ((pb == 1) & (tb == 1)).sum().item()
    tn = ((pb == 0) & (tb == 0)).sum().item()
    fp = ((pb == 1) & (tb == 0)).sum().item()
    fn = ((pb == 0) & (tb == 1)).sum().item()
    return tp, tn, fp, fn


def metrics_from_confusion(tp, tn, fp, fn, eps=1e-6):
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    iou = tp / (tp + fp + fn + eps)
    dice = 2 * tp / (2 * tp + fp + fn + eps)
    accuracy = (tp + tn) / (tp + tn + fp + fn + eps)
    specificity = tn / (tn + fp + eps)
    return {"precision": precision, "recall": recall, "f1": f1,
            "iou": iou, "dice": dice, "accuracy": accuracy,
            "specificity": specificity}


# ─────────────────────────────────────────────────────────────────────────────
# Train / eval one epoch
# ─────────────────────────────────────────────────────────────────────────────
def train_one_epoch(model, loader, optimizer, criterion, device, epoch, epochs):
    model.train()
    total_loss = 0.0
    TP = TN = FP = FN = 0
    pbar = tqdm(loader, desc=f"Epoch {epoch:02d}/{epochs} [train]",
                unit="batch", ncols=110, leave=False)
    for img, mask in pbar:
        img, mask = img.to(device), mask.to(device)
        optimizer.zero_grad()
        pred = model(img)
        loss = criterion(pred, mask)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        tp, tn, fp, fn = accumulate_confusion(pred.detach(), mask)
        TP += tp; TN += tn; FP += fp; FN += fn
        running_loss = total_loss / (pbar.n + 1)
        running_iou = TP / (TP + FP + FN + 1e-6)
        pbar.set_postfix({"loss": f"{running_loss:.4f}", "IoU": f"{running_iou:.4f}"})

    m = metrics_from_confusion(TP, TN, FP, FN)
    m["loss"] = total_loss / len(loader)
    m["tp"], m["tn"], m["fp"], m["fn"] = TP, TN, FP, FN
    return m


@torch.no_grad()
def evaluate(model, loader, criterion, device, epoch, epochs):
    model.eval()
    total_loss = 0.0
    TP = TN = FP = FN = 0
    pbar = tqdm(loader, desc=f"Epoch {epoch:02d}/{epochs} [val]  ",
                unit="batch", ncols=110, leave=False)
    for img, mask in pbar:
        img, mask = img.to(device), mask.to(device)
        pred = model(img)
        loss = criterion(pred, mask)
        total_loss += loss.item()
        tp, tn, fp, fn = accumulate_confusion(pred, mask)
        TP += tp; TN += tn; FP += fp; FN += fn
        running_loss = total_loss / (pbar.n + 1)
        running_iou = TP / (TP + FP + FN + 1e-6)
        pbar.set_postfix({"loss": f"{running_loss:.4f}", "IoU": f"{running_iou:.4f}"})

    m = metrics_from_confusion(TP, TN, FP, FN)
    m["loss"] = total_loss / len(loader)
    m["tp"], m["tn"], m["fp"], m["fn"] = TP, TN, FP, FN
    return m


# ─────────────────────────────────────────────────────────────────────────────
# Live plot
# ─────────────────────────────────────────────────────────────────────────────
class LivePlot:
    def __init__(self, enabled=True):
        self.enabled = enabled
        if not enabled:
            return
        plt.ion()
        self.fig, self.axes = plt.subplots(1, 3, figsize=(15, 4))
        self.fig.suptitle("MarineGuard U-Net — Live Training Progress",
                          fontsize=13, fontweight="bold")
        try:
            self.fig.canvas.manager.set_window_title("MarineGuard Training")
        except Exception:
            pass

        self.axes[0].set_title("Loss (BCE + Dice)")
        self.axes[0].set_xlabel("Epoch"); self.axes[0].set_ylabel("Loss")
        self.axes[0].grid(alpha=0.3)

        self.axes[1].set_title("IoU")
        self.axes[1].set_xlabel("Epoch"); self.axes[1].set_ylabel("IoU")
        self.axes[1].set_ylim(0, 1); self.axes[1].grid(alpha=0.3)

        self.axes[2].set_title("Dice / F1")
        self.axes[2].set_xlabel("Epoch"); self.axes[2].set_ylabel("Dice")
        self.axes[2].set_ylim(0, 1); self.axes[2].grid(alpha=0.3)

        plt.tight_layout()
        plt.show(block=False)
        plt.pause(0.1)

    def update(self, history):
        if not self.enabled or not history:
            return
        epochs = [h["epoch"] for h in history]
        for ax in self.axes:
            for line in ax.lines:
                line.remove()
            if ax.get_legend() is not None:
                ax.get_legend().remove()

        self.axes[0].plot(epochs, [h["train_loss"] for h in history], "o-", label="Train")
        self.axes[0].plot(epochs, [h["val_loss"] for h in history], "s-", label="Val")
        self.axes[0].legend(loc="upper right")

        self.axes[1].plot(epochs, [h["train_iou"] for h in history], "o-", label="Train")
        self.axes[1].plot(epochs, [h["val_iou"] for h in history], "s-", label="Val")
        self.axes[1].legend(loc="lower right")

        self.axes[2].plot(epochs, [h["train_dice"] for h in history], "o-", label="Train")
        self.axes[2].plot(epochs, [h["val_dice"] for h in history], "s-", label="Val")
        self.axes[2].legend(loc="lower right")

        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
        plt.pause(0.05)

    def snapshot(self, path):
        if self.enabled:
            self.fig.savefig(path, dpi=110, bbox_inches="tight")

    def close(self):
        if self.enabled:
            plt.ioff()


def plot_final_curves(history, path):
    epochs = [h["epoch"] for h in history]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle("MarineGuard U-Net — Final Training History",
                 fontsize=15, fontweight="bold")

    def _p(ax, kt, kv, title, ylabel="Score", ylim=None):
        ax.plot(epochs, [h[kt] for h in history], "o-", label="Train")
        ax.plot(epochs, [h[kv] for h in history], "s-", label="Val")
        ax.set_title(title); ax.set_xlabel("Epoch"); ax.set_ylabel(ylabel)
        if ylim: ax.set_ylim(*ylim)
        ax.grid(alpha=0.3); ax.legend()

    _p(axes[0, 0], "train_loss", "val_loss", "Loss", "Loss")
    _p(axes[0, 1], "train_iou", "val_iou", "IoU", "IoU", (0, 1))
    _p(axes[0, 2], "train_dice", "val_dice", "Dice", "Dice", (0, 1))
    _p(axes[1, 0], "train_f1", "val_f1", "F1", "F1", (0, 1))
    _p(axes[1, 1], "train_precision", "val_precision", "Precision", "Precision", (0, 1))
    _p(axes[1, 2], "train_recall", "val_recall", "Recall", "Recall", (0, 1))

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"[Plot] Final curves -> {path}")


def print_epoch_table(history):
    print("\n" + "=" * 100)
    print(f"  {'Epoch':>5} | {'TrainLoss':>10} | {'ValLoss':>9} | "
          f"{'TrainIoU':>9} | {'ValIoU':>8} | {'ValDice':>8} | "
          f"{'ValF1':>7} | {'ValPrec':>8} | {'ValRec':>7} | {'LR':>9}")
    print("-" * 100)
    for h in history:
        print(f"  {h['epoch']:>5} | {h['train_loss']:>10.4f} | {h['val_loss']:>9.4f} | "
              f"{h['train_iou']:>9.4f} | {h['val_iou']:>8.4f} | {h['val_dice']:>8.4f} | "
              f"{h['val_f1']:>7.4f} | {h['val_precision']:>8.4f} | {h['val_recall']:>7.4f} | "
              f"{h['lr']:>9.2e}")
    print("=" * 100 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no-live-plot", action="store_true")
    args = parser.parse_args()

    if not os.path.isdir(IMG_DIR) or not os.listdir(IMG_DIR):
        print(f"[ERROR] Dataset not found at {IMG_DIR}")
        print("Run: python data/generate_sar_dataset.py")
        return

    torch.manual_seed(SEED); np.random.seed(SEED)
    device = torch.device("cpu")

    print("=" * 70)
    print("  MARINEGUARD U-NET — FAST CPU TRAINING")
    print("=" * 70)
    print(f"  Device          : {device}")
    print(f"  CPU threads     : {torch.get_num_threads()}")
    print(f"  Epochs          : {args.epochs}")
    print(f"  Batch size      : {args.batch_size}")
    print(f"  Learning rate   : {args.lr}")
    print(f"  Image size      : 128 x 128")
    print(f"  Live plot       : {'OFF' if args.no_live_plot else 'ON'}")
    print("=" * 70 + "\n")

    full_ds = SARDataset(IMG_DIR, MASK_DIR, augment=False)
    n_total = len(full_ds)
    n_val = int(n_total * VAL_SPLIT)
    n_train = n_total - n_val
    train_ds, val_ds = random_split(
        full_ds, [n_train, n_val],
        generator=torch.Generator().manual_seed(SEED))
    train_ds.dataset.augment = True

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                            shuffle=False, num_workers=NUM_WORKERS)

    print(f"  Total samples      : {n_total}")
    print(f"  Training samples   : {n_train}")
    print(f"  Validation samples : {n_val}")
    print(f"  Batches per epoch  : {len(train_loader)}\n")

    model = UNet(in_channels=1, out_channels=1).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters   : {n_params:,} ({n_params/1e6:.3f}M)")
    print(f"  Model size (FP32)  : {n_params*4/(1024**2):.2f} MB\n")

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=2, factor=0.5)
    criterion = BCEDiceLoss()

    history = []
    start_epoch = 1
    best_val_loss = float("inf")

    if args.resume and os.path.exists(CHECKPOINT_PATH):
        ckpt = torch.load(CHECKPOINT_PATH, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        scheduler.load_state_dict(ckpt["scheduler_state"])
        history = ckpt.get("history", [])
        start_epoch = ckpt["epoch"] + 1
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        print(f"  Resumed from epoch {start_epoch}\n")

    live = LivePlot(enabled=not args.no_live_plot)
    if not args.no_live_plot:
        print("  >> Live training window opened. Watch it update each epoch.\n")

    t_start = time.time()
    try:
        for epoch in range(start_epoch, args.epochs + 1):
            ep_start = time.time()
            train_m = train_one_epoch(model, train_loader, optimizer,
                                      criterion, device, epoch, args.epochs)
            val_m = evaluate(model, val_loader, criterion, device,
                             epoch, args.epochs)
            lr_now = optimizer.param_groups[0]["lr"]
            scheduler.step(val_m["loss"])
            ep_time = time.time() - ep_start

            row = {
                "epoch": epoch, "lr": lr_now,
                "train_loss": train_m["loss"], "val_loss": val_m["loss"],
                "train_iou": train_m["iou"], "val_iou": val_m["iou"],
                "train_dice": train_m["dice"], "val_dice": val_m["dice"],
                "train_precision": train_m["precision"], "val_precision": val_m["precision"],
                "train_recall": train_m["recall"], "val_recall": val_m["recall"],
                "train_f1": train_m["f1"], "val_f1": val_m["f1"],
                "train_accuracy": train_m["accuracy"], "val_accuracy": val_m["accuracy"],
                "train_specificity": train_m["specificity"], "val_specificity": val_m["specificity"],
                "val_tp": val_m["tp"], "val_tn": val_m["tn"],
                "val_fp": val_m["fp"], "val_fn": val_m["fn"],
                "time_sec": round(ep_time, 2),
            }
            history.append(row)

            marker = ""
            if val_m["loss"] < best_val_loss:
                best_val_loss = val_m["loss"]
                torch.save(model.state_dict(), WEIGHTS_PATH)
                marker = "  * best"

            print(f"[Epoch {epoch:02d}/{args.epochs}] "
                  f"train_loss={train_m['loss']:.4f} "
                  f"val_loss={val_m['loss']:.4f} "
                  f"val_IoU={val_m['iou']:.4f} "
                  f"val_Dice={val_m['dice']:.4f} "
                  f"val_F1={val_m['f1']:.4f} "
                  f"lr={lr_now:.2e} ({ep_time:.1f}s){marker}")

            live.update(history)
            live.snapshot(os.path.join(LIVE_DIR, "live_plot.png"))

            if epoch % CHECKPOINT_EVERY == 0:
                torch.save({
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "scheduler_state": scheduler.state_dict(),
                    "history": history,
                    "best_val_loss": best_val_loss,
                }, CHECKPOINT_PATH)
                print(f"         └─ checkpoint saved")

    except KeyboardInterrupt:
        print("\n\n  [Interrupted] Saving checkpoint...")
        torch.save({
            "epoch": history[-1]["epoch"] if history else 0,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "history": history,
            "best_val_loss": best_val_loss,
        }, CHECKPOINT_PATH)

    elapsed = time.time() - t_start
    live.close()

    if not history:
        print("\nNo epochs completed.")
        return

    print("\n" + "=" * 70)
    print(f"  TRAINING COMPLETE — Total time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print("=" * 70)
    print_epoch_table(history)

    with open(HISTORY_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        w.writeheader(); w.writerows(history)
    print(f"[CSV]  Training history -> {HISTORY_CSV}")

    plot_final_curves(history, CURVES_PNG)

    best = min(history, key=lambda h: h["val_loss"])
    summary = {
        "model_name": "U-Net (Lightweight)",
        "total_parameters": n_params,
        "model_size_mb": round(n_params * 4 / (1024 ** 2), 2),
        "training_samples": n_train,
        "validation_samples": n_val,
        "epochs_completed": len(history),
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "optimizer": "Adam",
        "loss_function": "BCE + Dice",
        "device": "CPU",
        "training_time_sec": round(elapsed, 2),
        "best_epoch": best["epoch"],
        "best_val_loss": round(best["val_loss"], 4),
        "best_val_iou": round(best["val_iou"], 4),
        "best_val_dice": round(best["val_dice"], 4),
        "best_val_f1": round(best["val_f1"], 4),
        "final_val_iou": round(history[-1]["val_iou"], 4),
        "final_val_dice": round(history[-1]["val_dice"], 4),
        "final_val_f1": round(history[-1]["val_f1"], 4),
    }
    with open(SUMMARY_JSON, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[JSON] Training summary -> {SUMMARY_JSON}")

    print(f"\nBest epoch    : {best['epoch']}  "
          f"(val_loss={best['val_loss']:.4f}, IoU={best['val_iou']:.4f})")
    print(f"Total time    : {elapsed:.1f}s")
    print(f"Best weights  : {WEIGHTS_PATH}")


if __name__ == "__main__":
    main()