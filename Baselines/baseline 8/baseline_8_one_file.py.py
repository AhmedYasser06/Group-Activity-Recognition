#  ── HOW TO RESUME AFTER TIMEOUT ─────────────────────────────
#  Case 1: Timed out during Phase A
#    RESUME_PHASE_A = True
#    PHASE_A_RESUME_CKPT = "/kaggle/working/phase_a_epoch_latest.pkl"
#
#  Case 2: Phase A done, timed out during Phase B
#    SKIP_PHASE_A = True
#    PHASE_A_BEST_FOR_B = "/kaggle/working/phase_a_best_b8.pkl"
#    RESUME_PHASE_B = True
#    PHASE_B_RESUME_CKPT = "/kaggle/working/phase_b_epoch_latest.pkl"
#
#  Case 3: Phase A done, starting Phase B fresh
#    SKIP_PHASE_A = True
#    PHASE_A_BEST_FOR_B = "/kaggle/working/phase_a_best_b8.pkl"
#    RESUME_PHASE_B = False
# ============================================================

# ── 0. Installs ──────────────────────────────────────────────
import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "torchinfo", "albumentations", "scikit-learn", "seaborn"],
               check=True)

# ── 1. Imports ───────────────────────────────────────────────
import os, pickle, random, cv2, warnings
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import Dataset, DataLoader
from torchvision import models
from sklearn.metrics import (f1_score, classification_report,
                             confusion_matrix, accuracy_score)
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from typing import List, Tuple
from tqdm import tqdm
warnings.filterwarnings("ignore")

# ── 2. Paths ─────────────────────────────────────────────────
VIDEOS_PATH      = "/kaggle/input/datasets/ahmedmohamed365/volleyball/volleyball_/videos"
ANNOT_PATH       = "/kaggle/working/annot_all.pkl"
PHASE_A_CKPT_OUT = "/kaggle/working/phase_a_best_b8.pkl"
PHASE_B_CKPT_OUT = "/kaggle/working/phase_b_best_b8.pkl"
OUTPUT_DIR       = "/kaggle/working"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── 3. Resume flags ───────────────────────────────────────────
SKIP_PHASE_A        = True
RESUME_PHASE_A      = False
PHASE_A_RESUME_CKPT = "/kaggle/working/phase_a_epoch_latest.pkl"

RESUME_PHASE_B      = False
PHASE_B_RESUME_CKPT = "/kaggle/working/phase_b_epoch_latest.pkl"

PHASE_A_BEST_FOR_B  = "/kaggle/input/notebooks/ahmedyasser06/b7-22-2/phase_a_best_b7.pkl"

# ── 4. Seed ──────────────────────────────────────────────────
SEED = 31

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False

set_seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device : {DEVICE}")
print(f"Videos : {VIDEOS_PATH}")
print(f"Annots : {ANNOT_PATH}")

# ── 5. Labels ─────────────────────────────────────────────────
PERSON_CLASSES = ["Waiting","Setting","Digging","Falling",
                  "Spiking","Blocking","Jumping","Moving","Standing"]
PERSON_LABELS  = {c.lower(): i for i, c in enumerate(PERSON_CLASSES)}

GROUP_CLASSES  = ["r_set","r_spike","r-pass","r_winpoint",
                  "l_winpoint","l-pass","l-spike","l_set"]
GROUP_LABELS   = {c: i for i, c in enumerate(GROUP_CLASSES)}

TRAIN_SPLIT = [1,3,6,7,10,13,15,16,18,22,23,31,32,36,38,39,40,41,42,48,50,52,53,54]
VAL_SPLIT   = [0,2,8,12,17,19,24,26,27,28,30,33,46,49,51]
TEST_SPLIT  = [4,5,9,11,14,20,21,25,29,34,35,37,43,44,45,47]

# ── 6. Hyper-parameters ───────────────────────────────────────

# Phase A — aligned with reference B7 Phase A (Doc 25/28)
# Key changes vs before: WD=0.01, smoothing=0.05, aug p=0.55, patience=2
PA_HIDDEN      = 512
PA_LAYERS      = 1
PA_NUM_CLASSES = 9
PA_BS          = 2
PA_LR          = 2e-4
PA_WD          = 0.1           # B8.yml
PA_EPOCHS      = 15
PA_SMOOTHING   = 0.1           # B8.yml
PA_PATIENCE    = 3             # B8.yml

# Phase B — from Baseline B8.yml + reference train_b.py observations
GA_HIDDEN      = 512
GA_LAYERS      = 2
GA_NUM_CLASSES = 8
GA_BS          = 8
GA_LR          = 6e-6          # B8.yml value — used when Phase A was properly trained
GA_LR_FRESH    = 3e-5          # reference train_b.py LR — use when loading B7 Phase A ckpt
GA_WD          = 1.0
GA_EPOCHS      = 50
GA_SMOOTHING   = 0.0

# ── 7. Datasets ───────────────────────────────────────────────

class PersonActivityDataset(Dataset):
    """
    Returns per-clip player sequences for person-action classification.
    Returns:
        clip   : (P, 9, C, H, W)
        labels : (P, 9, 9)
    person_collate_fn pads to (B, 12, 9, C, H, W), labels → (B*12, 9)
    """
    def __init__(self, videos_path, annot_path, split, labels,
                 sort=True, transform=None):
        self.videos_path = videos_path
        self.labels      = labels
        self.sort        = sort
        self.transform   = transform
        self.items       = []
        with open(annot_path, "rb") as f:
            annots = pickle.load(f)
        for clip_id in split:
            clip_data = annots[str(clip_id)]
            for clip_dir, clip_info in clip_data.items():
                frames_list = list(clip_info["frame_boxes_dct"].items())
                self.items.append({
                    "clip_id"  : clip_id,
                    "clip_dir" : clip_dir,
                    "frames"   : frames_list,
                })

    def __len__(self): return len(self.items)

    def _load_frame(self, clip_id, clip_dir, frame_id):
        p = os.path.join(self.videos_path,
                         str(clip_id), str(clip_dir), f"{frame_id}.jpg")
        img = cv2.imread(p)
        if img is None:
            raise FileNotFoundError(f"Frame not found: {p}")
        return img

    def _crop_transform(self, frame, box):
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = box.box
        x1, x2 = max(0, x1), min(w, x2)
        y1, y2 = max(0, y1), min(h, y2)
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            crop = np.zeros((64, 64, 3), dtype=np.uint8)
        if self.transform:
            crop = self.transform(image=crop)["image"]
        return crop

    def __getitem__(self, idx):
        item = self.items[idx]
        frame_crops, frame_labels = [], []

        for frame_id, boxes in item["frames"]:
            frame = self._load_frame(item["clip_id"],
                                     item["clip_dir"], frame_id)
            valid = [b for b in boxes
                     if not (hasattr(b, "lost") and b.lost == 1)
                     and b.category in self.labels]
            if self.sort:
                valid = sorted(valid,
                               key=lambda b: (b.box[0] + b.box[2]) / 2.0)
            crops, lbls = [], []
            for box in valid:
                crop = self._crop_transform(frame, box)
                lbl  = torch.zeros(len(self.labels))
                lbl[self.labels[box.category]] = 1.0
                crops.append(crop); lbls.append(lbl)
            frame_crops.append(crops)
            frame_labels.append(lbls)

        # pad players within each frame to max_p across this clip
        max_p      = max((len(fc) for fc in frame_crops), default=1)
        max_p      = max(max_p, 1)
        dummy_crop = torch.zeros(3, 224, 224)
        dummy_lbl  = torch.zeros(len(self.labels))

        seq_crops, seq_labels = [], []
        for fc, fl in zip(frame_crops, frame_labels):
            while len(fc) < max_p:
                fc.append(dummy_crop); fl.append(dummy_lbl)
            seq_crops.append(torch.stack(fc[:max_p]))
            seq_labels.append(torch.stack(fl[:max_p]))

        clips  = torch.stack(seq_crops).permute(1, 0, 2, 3, 4)   # (P,9,C,H,W)
        labels = torch.stack(seq_labels).permute(1, 0, 2)          # (P,9,9)
        return clips, labels


class GroupActivityDataset(Dataset):
    """
    Returns per-clip player sequences for group-activity classification.
    Returns:
        clip   : (P, 9, C, H, W)   — P consistent across all frames in clip
        labels : (9, 8)              group label repeated per frame

    FIX: frames in the same clip can have different numbers of valid
    players after lost-filtering. We now pad every frame to the SAME
    player count (max across all frames in the clip) BEFORE stacking,
    so torch.stack never sees mismatched sizes.
    """
    def __init__(self, videos_path, annot_path, split, labels,
                 sort=True, transform=None):
        self.videos_path = videos_path
        self.labels      = labels
        self.sort        = sort
        self.transform   = transform
        self.items       = []
        with open(annot_path, "rb") as f:
            annots = pickle.load(f)
        for clip_id in split:
            clip_data = annots[str(clip_id)]
            for clip_dir, clip_info in clip_data.items():
                category = clip_info["category"]
                if category not in labels: continue
                frames_list = list(clip_info["frame_boxes_dct"].items())
                self.items.append({
                    "clip_id"  : clip_id,
                    "clip_dir" : clip_dir,
                    "frames"   : frames_list,
                    "category" : category,
                })

    def __len__(self): return len(self.items)

    def _load_frame(self, clip_id, clip_dir, frame_id):
        p = os.path.join(self.videos_path,
                         str(clip_id), str(clip_dir), f"{frame_id}.jpg")
        img = cv2.imread(p)
        if img is None:
            raise FileNotFoundError(f"Frame not found: {p}")
        return img

    def _extract_crops_list(self, frame, boxes):
        """
        Returns a list of tensors (one per player).

        CRITICAL FIX — matches the reference exactly:
        The reference (Group_Activity_DataSet in data_utils.py) never
        filters by lost==1. It keeps ALL players from the pkl, sorts
        them by x-centre, and the collate_fn pads to 12.
        The team split (indices :6 and 6:) only works correctly when
        all players are present in their original sorted order.
        Filtering out lost players shifts indices and breaks the split,
        causing r_winpoint to be predicted as l_winpoint (9% recall).
        """
        h, w  = frame.shape[:2]
        # NO lost filtering — keep all boxes, sort by x-centre
        if self.sort:
            boxes = sorted(boxes,
                           key=lambda b: (b.box[0] + b.box[2]) / 2.0)
        crops = []
        for box in boxes:
            x1, y1, x2, y2 = box.box
            x1, x2 = max(0, x1), min(w, x2)
            y1, y2 = max(0, y1), min(h, y2)
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                crop = np.zeros((64, 64, 3), dtype=np.uint8)
            if self.transform:
                crop = self.transform(image=crop)["image"]
            crops.append(crop)
        return crops

    def __getitem__(self, idx):
        item    = self.items[idx]
        one_hot = torch.zeros(len(self.labels))
        one_hot[self.labels[item["category"]]] = 1.0

        # Collect per-frame crop lists — all frames have the same
        # player count (all 12 kept) so torch.stack is always safe.
        frame_crop_lists = []
        for frame_id, boxes in item["frames"]:
            frame = self._load_frame(item["clip_id"],
                                     item["clip_dir"], frame_id)
            crops = self._extract_crops_list(frame, boxes)
            frame_crop_lists.append(crops)

        # All frames should have the same count, but pad to be safe
        max_p      = max((len(fc) for fc in frame_crop_lists), default=1)
        max_p      = max(max_p, 1)
        dummy_crop = torch.zeros(3, 224, 224)

        padded_frames = []
        for fc in frame_crop_lists:
            while len(fc) < max_p:
                fc.append(dummy_crop)
            padded_frames.append(torch.stack(fc[:max_p]))

        clip   = torch.stack(padded_frames).permute(1, 0, 2, 3, 4)
        labels = one_hot.unsqueeze(0).repeat(len(item["frames"]), 1)
        return clip, labels


# ── 8. Collate functions (exact reference) ────────────────────

def person_collate_fn(batch):
    """Pads player dim to 12. Returns (B,12,9,C,H,W), (B*12,9)."""
    clips, labels = zip(*batch)
    max_bboxes = 12
    padded_clips, padded_labels = [], []
    for clip, label in zip(clips, labels):
        num_bboxes = clip.size(0)
        if num_bboxes < max_bboxes:
            clip_pad  = torch.zeros(max_bboxes - num_bboxes,
                                    clip.size(1), clip.size(2),
                                    clip.size(3), clip.size(4))
            label_pad = torch.zeros(max_bboxes - num_bboxes,
                                    label.size(1), label.size(2))
            clip  = torch.cat((clip,  clip_pad),  dim=0)
            label = torch.cat((label, label_pad), dim=0)
        padded_clips.append(clip)
        padded_labels.append(label)
    padded_clips  = torch.stack(padded_clips)
    padded_labels = torch.stack(padded_labels)
    padded_labels = padded_labels[:, :, -1, :]
    b, bb, num_class = padded_labels.shape
    padded_labels = padded_labels.view(b * bb, num_class)
    return padded_clips, padded_labels


def group_collate_fn(batch):
    """Pads player dim to 12. Returns (B,12,9,C,H,W), (B,8)."""
    clips, labels = zip(*batch)
    max_bboxes = 12
    padded_clips = []
    for clip in clips:
        num_bboxes = clip.size(0)
        if num_bboxes < max_bboxes:
            clip_pad = torch.zeros(max_bboxes - num_bboxes,
                                   clip.size(1), clip.size(2),
                                   clip.size(3), clip.size(4))
            clip = torch.cat((clip, clip_pad), dim=0)
        padded_clips.append(clip)
    padded_clips = torch.stack(padded_clips)
    labels       = torch.stack(labels)
    labels       = labels[:, -1, :]
    return padded_clips, labels


# ── 9. Models (exact reference) ───────────────────────────────

class PersonActivityClassifier(nn.Module):
    def __init__(self, num_classes=9, hidden_size=512, num_layers=1):
        super().__init__()
        self.resnet50 = nn.Sequential(
            *list(models.resnet50(
                weights=models.ResNet50_Weights.DEFAULT).children())[:-1]
        )
        self.layer_norm = nn.LayerNorm(2048)
        self.lstm_1 = nn.LSTM(
            input_size=2048, hidden_size=hidden_size,
            num_layers=num_layers, batch_first=True,
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.LayerNorm(512), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.LayerNorm(256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        b, bb, seq, c, h, w = x.shape
        x = x.view(b * bb * seq, c, h, w)
        x = self.resnet50(x)
        x = x.view(b * bb, seq, -1)
        x = self.layer_norm(x)
        x, (h, c) = self.lstm_1(x)
        x = x[:, -1, :]
        x = self.fc(x)
        return x


class GroupActivityClassifier(nn.Module):
    def __init__(self, person_model, hidden_size=512,
                 num_layers=2, num_classes=8):
        super().__init__()
        self.resnet50 = person_model.resnet50
        self.lstm_1   = person_model.lstm_1
        for module in [self.resnet50, self.lstm_1]:
            for param in module.parameters():
                param.requires_grad = False
        self.layer_norm = nn.LayerNorm(2048)
        self.pool = nn.AdaptiveMaxPool2d((1, 1024))
        self.lstm_2 = nn.LSTM(
            input_size=2048, hidden_size=hidden_size,
            num_layers=num_layers, batch_first=True, dropout=0.3,
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        b, bb, seq, c, h, w = x.shape
        x  = x.view(b * bb * seq, c, h, w)
        x1 = self.resnet50(x)
        x1 = x1.view(b * bb, seq, -1)
        x1 = self.layer_norm(x1)
        x2, (h_1, c_1) = self.lstm_1(x1)
        x  = torch.cat([x1, x2], dim=2).contiguous()
        x  = x.view(b * seq, bb, -1)
        team_1 = self.pool(x[:, :6, :])
        team_2 = self.pool(x[:, 6:, :])
        x  = torch.cat([team_1, team_2], dim=1)
        x  = x.view(b, seq, -1)
        x  = self.layer_norm(x)
        x, (h_2, c_2) = self.lstm_2(x)
        x  = x[:, -1, :]
        x  = self.fc(x)
        return x


# ── 10. Transforms ────────────────────────────────────────────

# Phase A — B8.yml hyperparameters, augmentation p=0.55 from B5 reference
train_transform_pa = A.Compose([
    A.Resize(224, 224),
    A.OneOf([
        A.GaussianBlur(blur_limit=(3, 7)),
        A.ColorJitter(brightness=0.2),
        A.RandomBrightnessContrast(),
        A.GaussNoise(),
    ], p=0.55),                    # B5 reference aug probability
    A.OneOf([A.HorizontalFlip(), A.VerticalFlip()], p=0.05),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])

# Phase B augmentation (unchanged from yml)
train_transform_ga = A.Compose([
    A.Resize(224, 224),
    A.OneOf([
        A.GaussianBlur(blur_limit=(3, 7)),
        A.ColorJitter(brightness=0.2),
        A.RandomBrightnessContrast(),
        A.GaussNoise(),
        A.MotionBlur(blur_limit=5),
        A.MedianBlur(blur_limit=5),
    ], p=0.75),
    A.OneOf([A.HorizontalFlip(), A.VerticalFlip(), A.RandomRotate90()], p=0.05),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])

val_transform = A.Compose([
    A.Resize(224, 224),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])


# ── 11. Checkpoint utilities ──────────────────────────────────

def save_checkpoint(path, model, optimizer, epoch, val_acc,
                    config, history=None, best_val_acc=0.0):
    torch.save({
        "epoch"               : epoch,
        "model_state_dict"    : model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_acc"             : val_acc,
        "best_val_acc"        : best_val_acc,
        "config"              : config,
        "history"             : history,
    }, path)


def load_checkpoint(path, model, optimizer=None, device=DEVICE):
    ckpt        = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    epoch        = ckpt.get("epoch", 0)
    val_acc      = ckpt.get("val_acc", 0.0)
    best_val_acc = ckpt.get("best_val_acc", val_acc)
    history      = ckpt.get("history", None)
    print(f"  ✓ Loaded  epoch={epoch}  val_acc={val_acc*100:.2f}%  "
          f"best={best_val_acc*100:.2f}%")
    return model, optimizer, epoch, best_val_acc, history


def load_weights_only(path, model, device=DEVICE):
    ckpt    = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    val_acc = ckpt.get("val_acc", 0.0)
    print(f"  ✓ Weights loaded  val_acc={val_acc*100:.2f}%")
    return model


# ── 12. Visualization ─────────────────────────────────────────

def plot_training_curves_phase_a(history, save_path):
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Phase A — Person Activity Training Curves",
                 fontsize=14, fontweight="bold")
    axes[0].plot(epochs, history["train_loss"], "b-o", markersize=3,
                 label="Train")
    axes[0].plot(epochs, history["val_loss"],   "r-o", markersize=3,
                 label="Val")
    best_ep = int(np.argmin(history["val_loss"])) + 1
    axes[0].axvline(best_ep, color="green", linestyle="--", alpha=0.6,
                    label=f"Best ep {best_ep}")
    axes[0].set_title("Loss"); axes[0].legend(); axes[0].grid(True, alpha=0.3)

    train_pct = [a * 100 for a in history["train_acc"]]
    val_pct   = [a * 100 for a in history["val_acc"]]
    axes[1].plot(epochs, train_pct, "b-o", markersize=3, label="Train")
    axes[1].plot(epochs, val_pct,   "r-o", markersize=3, label="Val")
    best_acc = max(val_pct)
    axes[1].axhline(best_acc, color="green", linestyle="--", alpha=0.6,
                    label=f"Best {best_acc:.1f}%")
    axes[1].yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f%%"))
    axes[1].set_title("Accuracy"); axes[1].legend(); axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150); plt.show()
    print(f"  ✓ Phase A curves → {save_path}")


def plot_training_curves_phase_b(history, save_path):
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle("Phase B — Group Activity Training Curves",
                 fontsize=14, fontweight="bold")

    ax = axes[0, 0]
    ax.plot(epochs, history["train_loss"], "b-o", markersize=3, label="Train")
    ax.plot(epochs, history["val_loss"],   "r-o", markersize=3, label="Val")
    best_ep = int(np.argmin(history["val_loss"])) + 1
    ax.axvline(best_ep, color="green", linestyle="--", alpha=0.6,
               label=f"Best ep {best_ep}")
    ax.set_title("Loss"); ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    train_pct = [a * 100 for a in history["train_acc"]]
    val_pct   = [a * 100 for a in history["val_acc"]]
    ax.plot(epochs, train_pct, "b-o", markersize=3, label="Train")
    ax.plot(epochs, val_pct,   "r-o", markersize=3, label="Val")
    best_acc = max(val_pct)
    ax.axvline(val_pct.index(best_acc)+1, color="green", linestyle="--",
               alpha=0.6, label=f"Best {best_acc:.1f}%")
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f%%"))
    ax.set_title("Accuracy"); ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.plot(epochs, history["val_f1"], "m-o", markersize=3)
    ax.axhline(max(history["val_f1"]), color="green", linestyle="--",
               alpha=0.6, label=f"Best F1={max(history['val_f1']):.4f}")
    ax.set_title("Val F1"); ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.plot(epochs, history["lr"], "g-o", markersize=3)
    ax.set_yscale("log"); ax.set_title("LR (log)"); ax.grid(True, alpha=0.3)
    lr_arr = history["lr"]
    for i in range(1, len(lr_arr)):
        if lr_arr[i] < lr_arr[i-1] * 0.5:
            ax.annotate(f"↓ ep {i+1}", xy=(i+1, lr_arr[i]),
                        xytext=(i+1, lr_arr[i]*3), fontsize=8, color="red",
                        arrowprops=dict(arrowstyle="->", color="red", lw=0.8))

    plt.tight_layout()
    plt.savefig(save_path, dpi=150); plt.show()
    print(f"  ✓ Phase B curves → {save_path}")


def plot_confusion_matrices(targets, preds, class_names, title, save_path):
    cm      = confusion_matrix(targets, preds)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    fig.suptitle(title, fontsize=14, fontweight="bold")
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names,
                ax=axes[0], cbar_kws={"label": "Count"}, linewidths=0.5)
    axes[0].set_xlabel("Predicted"); axes[0].set_ylabel("True")
    axes[0].set_title("Count")
    axes[0].set_xticklabels(axes[0].get_xticklabels(),
                             rotation=30, ha="right", fontsize=9)
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names,
                ax=axes[1], cbar_kws={"label": "Recall rate"},
                vmin=0, vmax=1, linewidths=0.5)
    axes[1].set_xlabel("Predicted"); axes[1].set_ylabel("True")
    axes[1].set_title("Normalized (recall per class)")
    axes[1].set_xticklabels(axes[1].get_xticklabels(),
                             rotation=30, ha="right", fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150); plt.show()
    print(f"  ✓ Confusion matrix → {save_path}")


def plot_per_class_accuracy(targets, preds, class_names, title, save_path):
    cm      = confusion_matrix(targets, preds)
    per_cls = cm.diagonal() / cm.sum(axis=1)
    counts  = cm.sum(axis=1)
    colors  = ["#2ecc71" if a >= 0.90 else
               "#f39c12" if a >= 0.75 else
               "#e74c3c" for a in per_cls]
    fig, ax = plt.subplots(figsize=(10, 6))
    y_pos   = np.arange(len(class_names))
    bars    = ax.barh(y_pos, per_cls * 100, color=colors,
                      edgecolor="white", height=0.6)
    for bar, acc, n in zip(bars, per_cls, counts):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                f"{acc*100:.1f}%  (n={n})", va="center",
                ha="left", fontsize=9)
    ax.set_yticks(y_pos); ax.set_yticklabels(class_names, fontsize=10)
    ax.set_xlabel("Accuracy (%)"); ax.set_xlim(0, 115)
    ax.axvline(90, color="green",  linestyle="--", alpha=0.5, label="90%")
    ax.axvline(75, color="orange", linestyle="--", alpha=0.5, label="75%")
    ax.legend(fontsize=8); ax.set_title(title, fontsize=12, fontweight="bold")
    ax.invert_yaxis(); ax.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150); plt.show()
    print(f"  ✓ Per-class accuracy → {save_path}")


def print_and_save_report(targets, preds, class_names,
                           split_name, acc, f1, loss, save_path):
    report = classification_report(targets, preds,
                                    target_names=class_names, digits=4)
    cm  = confusion_matrix(targets, preds)
    sep = "=" * 60
    lines = [sep, f"  {split_name} — Baseline B8", sep,
             f"  Accuracy : {acc*100:.2f}%",
             f"  F1 Score : {f1:.4f}  (weighted)",
             f"  Loss     : {loss:.4f}", "",
             "  Classification Report:", report,
             "  Confusion Matrix (raw counts):",
             "  Rows = True label, Cols = Predicted", ""]
    header = "  " + " ".join(f"{c:>10}" for c in class_names)
    lines.append(header)
    for i, row in enumerate(cm):
        lines.append("  " + f"{class_names[i]:>10}" +
                     " ".join(f"{v:>10}" for v in row))
    lines.append(sep)
    text = "\n".join(lines)
    print(text)
    with open(save_path, "w") as f: f.write(text)
    print(f"  ✓ Report → {save_path}")


def per_class_recall_table(targets, preds, class_names):
    """
    Print a compact per-class recall row every epoch.
    Highlights r_winpoint / l_winpoint with a verdict icon.
    Returns True if team-split looks healthy (both winpoints > 40%).
    """
    cm = confusion_matrix(targets, preds, labels=list(range(len(class_names))))
    print(f"\n  {'Class':>12}  {'Recall':>7}  {'Correct':>7}  {'Total':>5}")
    print(f"  {'─'*12}  {'─'*7}  {'─'*7}  {'─'*5}")
    all_ok = True
    for i, cls in enumerate(class_names):
        row_sum = cm[i].sum()
        correct = cm[i, i]
        recall  = correct / row_sum if row_sum > 0 else 0.0
        key     = cls in ("r_winpoint", "l_winpoint")
        if key:
            if recall < 0.40:
                icon = "🔴"; all_ok = False
            elif recall < 0.70:
                icon = "🟡"
            else:
                icon = "🟢"
        else:
            icon = "  "
        star = "★" if key else " "
        print(f"  {star}{cls:>11}  {recall*100:>6.1f}%  "
              f"{correct:>7}  {row_sum:>5}  {icon}")
    return all_ok


def early_diagnostic(model, val_loader, criterion, device,
                     epoch, class_names, output_dir):
    """
    Full diagnostic snapshot: confusion heatmap + per-class bar + verdict.
    Called at DIAG_EPOCH (epoch 5 by default) and at the very end.
    """
    print("\n" + "█"*65)
    print(f"  EARLY DIAGNOSTIC — after epoch {epoch}")
    print("█"*65)

    val_loss, val_acc, val_f1, gt, pred = evaluate(
        model, val_loader, criterion, device)
    print(f"  Val acc={val_acc*100:.2f}%   F1={val_f1:.4f}   "
          f"loss={val_loss:.4f}")

    per_class_recall_table(gt, pred, class_names)

    cm_path  = os.path.join(output_dir, f"diag_ep{epoch:02d}_confusion.png")
    acc_path = os.path.join(output_dir, f"diag_ep{epoch:02d}_per_class.png")
    plot_confusion_matrices(gt, pred, class_names,
        f"Diagnostic Confusion Matrix @ epoch {epoch}", cm_path)
    plot_per_class_accuracy(gt, pred, class_names,
        f"Diagnostic Per-Class Accuracy @ epoch {epoch}", acc_path)

    cm = confusion_matrix(gt, pred, labels=list(range(len(class_names))))
    r_win = cm[3, 3] / cm[3].sum() if cm[3].sum() > 0 else 0.0
    l_win = cm[4, 4] / cm[4].sum() if cm[4].sum() > 0 else 0.0

    print("\n  ── TEAM-SPLIT VERDICT ──────────────────────────────────")
    print(f"  r_winpoint recall : {r_win*100:.1f}%")
    print(f"  l_winpoint recall : {l_win*100:.1f}%")
    if r_win > 0.40 and l_win > 0.40:
        print("   FIX CONFIRMED — both winpoints learning.")
        print("     → Safe to continue to 50 epochs.")
    elif r_win < 0.20:
        print("   FIX NOT WORKING — r_winpoint still collapsing.")
        print("     → Check that class weights ARE enabled in criterion.")
        print("     → Check that sort=True AND no lost filtering in GroupActivityDataset.")
    else:
        print("    PARTIAL — r_winpoint recovering but still low.")
        print("     → Continue; should improve with more epochs.")
    print("█"*65 + "\n")


# How many epochs to run before the first diagnostic pause
DIAG_EPOCH = 5


def epoch_report(phase, epoch, total_epochs,
                 train_loss, train_acc, val_loss, val_acc,
                 val_f1, lr, best_val_acc, no_improve,
                 val_targets=None, val_preds=None):
    bar = "█" * int(val_acc * 30)
    gap = train_acc - val_acc
    print(f"\n{'─'*65}")
    print(f"[{phase}]  Epoch {epoch:3d}/{total_epochs}   LR={lr:.2e}")
    print(f"  Train  →  loss={train_loss:.4f}   acc={train_acc*100:.2f}%")
    print(f"  Val    →  loss={val_loss:.4f}   acc={val_acc*100:.2f}%   "
          f"f1={val_f1:.4f}")
    print(f"  Val acc  [{bar:<30}]  {val_acc*100:.1f}%")
    if val_acc > best_val_acc:
        print(f"   New best!  val acc → {val_acc*100:.2f}%")
    else:
        print(f"   No improvement for {no_improve} epoch(s)  "
              f"(best={best_val_acc*100:.2f}%)")
    if gap > 0.15:
        print(f"    Over-fitting: train-val gap = {gap*100:.1f}%")
    if lr < 1e-7:
        print(f"    LR very small ({lr:.2e}) — may have stalled")
    # compact per-class recall table every epoch (Phase B only)
    if val_targets is not None and val_preds is not None:
        per_class_recall_table(val_targets, val_preds, GROUP_CLASSES)
    print(f"{'─'*65}")


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_targets = [], []
    for x, y in loader:
        x, y   = x.to(device), y.to(device)
        out    = model(x)
        loss   = criterion(out, y)
        total_loss += loss.item()
        pred   = out.argmax(1); gt = y.argmax(1)
        correct += (pred == gt).sum().item()
        total   += y.size(0)
        all_preds.extend(pred.cpu().tolist())
        all_targets.extend(gt.cpu().tolist())
    return (total_loss / len(loader), correct / total,
            f1_score(all_targets, all_preds,
                     average="weighted", zero_division=0),
            all_targets, all_preds)


@torch.no_grad()
def evaluate_person(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_targets = [], []
    for clips, labels in loader:
        x = clips.to(device); y = labels.to(device)
        out  = model(x); loss = criterion(out, y)
        total_loss += loss.item()
        pred = out.argmax(1); gt = y.argmax(1)
        correct += (pred == gt).sum().item()
        total   += y.size(0)
        all_preds.extend(pred.cpu().tolist())
        all_targets.extend(gt.cpu().tolist())
    return (total_loss / len(loader), correct / total,
            f1_score(all_targets, all_preds,
                     average="weighted", zero_division=0))


def compute_class_weights(loader, num_classes, device):
    counts = torch.zeros(num_classes)
    for _, labels in loader:
        for c in labels.argmax(1): counts[c] += 1
    total   = counts.sum()
    weights = total / (num_classes * counts.clamp(min=1))
    weights = weights / weights.sum()
    for i, (cls, w) in enumerate(zip(GROUP_CLASSES, weights)):
        print(f"    {cls:12s}  count={int(counts[i]):4d}  weight={w:.4f}")
    return weights.to(device)


# ── 13. Phase A Training ──────────────────────────────────────

def train_phase_a():
    print("\n" + "="*65)
    print("  PHASE A — Person Activity")
    print("  (B8.yml: WD=0.1, smoothing=0.1, patience=3 | aug p=0.55)")
    print("="*65)

    train_ds = PersonActivityDataset(
        VIDEOS_PATH, ANNOT_PATH, TRAIN_SPLIT,
        PERSON_LABELS, sort=True, transform=train_transform_pa)
    val_ds   = PersonActivityDataset(
        VIDEOS_PATH, ANNOT_PATH, VAL_SPLIT,
        PERSON_LABELS, sort=True, transform=val_transform)

    train_loader = DataLoader(train_ds, batch_size=PA_BS, shuffle=True,
                              num_workers=4, pin_memory=True,
                              collate_fn=person_collate_fn)
    val_loader   = DataLoader(val_ds, batch_size=PA_BS * 2, shuffle=False,
                              num_workers=4, pin_memory=True,
                              collate_fn=person_collate_fn)

    model     = PersonActivityClassifier(
        PA_NUM_CLASSES, PA_HIDDEN, PA_LAYERS).to(DEVICE)
    criterion = nn.CrossEntropyLoss(label_smoothing=PA_SMOOTHING)
    optimizer = optim.AdamW(model.parameters(), lr=PA_LR, weight_decay=PA_WD)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.1, patience=PA_PATIENCE)
    scaler    = GradScaler()

    start_epoch  = 1
    best_val_acc = 0.0
    history      = {k: [] for k in
                    ["train_loss","train_acc","val_loss","val_acc","val_f1"]}

    if RESUME_PHASE_A and os.path.isfile(PHASE_A_RESUME_CKPT):
        print(f"  Resuming from: {PHASE_A_RESUME_CKPT}")
        model, optimizer, last_epoch, best_val_acc, saved_history = \
            load_checkpoint(PHASE_A_RESUME_CKPT, model, optimizer, DEVICE)
        start_epoch = last_epoch + 1
        if saved_history: history = saved_history
        for v in history.get("val_loss", []): scheduler.step(v)
        print(f"  Continuing from epoch {start_epoch}/{PA_EPOCHS}")
    else:
        print(f"  Starting fresh ({PA_EPOCHS} epochs)")

    print(f"  Train clips : {len(train_ds):,}  |  Val clips : {len(val_ds):,}")

    no_improve = 0
    for epoch in range(start_epoch, PA_EPOCHS + 1):
        model.train()
        run_loss, correct, total = 0.0, 0, 0

        pbar = tqdm(train_loader,
                    desc=f"[Phase-A] Ep {epoch:2d}/{PA_EPOCHS}",
                    unit="batch", leave=False,
                    bar_format="{l_bar}{bar:25}{r_bar}")
        for clips, labels in pbar:
            x = clips.to(DEVICE); y = labels.to(DEVICE)
            optimizer.zero_grad()
            with autocast():
                out  = model(x); loss = criterion(out, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer); scaler.update()
            run_loss += loss.item()
            correct  += (out.argmax(1) == y.argmax(1)).sum().item()
            total    += y.size(0)
            pbar.set_postfix(loss=f"{loss.item():.4f}",
                             acc=f"{100*correct/total:.1f}%")
        pbar.close()

        train_loss = run_loss / len(train_loader)
        train_acc  = correct / total
        val_loss, val_acc, val_f1 = evaluate_person(
            model, val_loader, criterion, DEVICE)
        scheduler.step(val_loss)
        lr = optimizer.param_groups[0]["lr"]

        for k, v in zip(history,
                        [train_loss, train_acc, val_loss, val_acc, val_f1]):
            history[k].append(v)

        if val_acc > best_val_acc:
            no_improve   = 0
            best_val_acc = val_acc
            save_checkpoint(PHASE_A_CKPT_OUT, model, optimizer,
                            epoch, val_acc, {"phase": "A"},
                            history, best_val_acc)
        else:
            no_improve += 1

        latest = os.path.join(OUTPUT_DIR, "phase_a_epoch_latest.pkl")
        save_checkpoint(latest, model, optimizer,
                        epoch, val_acc, {"phase": "A"},
                        history, best_val_acc)

        epoch_report("Phase-A", epoch, PA_EPOCHS,
                     train_loss, train_acc, val_loss, val_acc,
                     val_f1, lr, best_val_acc, no_improve)
        torch.cuda.empty_cache()

    plot_training_curves_phase_a(
        history, os.path.join(OUTPUT_DIR, "phase_a_curves.png"))
    print(f"\n  Phase A done.  Best val acc = {best_val_acc*100:.2f}%")
    return model


# ── 14. Phase B Training ──────────────────────────────────────

def train_phase_b(person_model):
    print("\n" + "="*65)
    print("  PHASE B — Group Activity")
    print("="*65)

    train_ds = GroupActivityDataset(
        VIDEOS_PATH, ANNOT_PATH, TRAIN_SPLIT,
        GROUP_LABELS, sort=True, transform=train_transform_ga)
    val_ds   = GroupActivityDataset(
        VIDEOS_PATH, ANNOT_PATH, VAL_SPLIT,
        GROUP_LABELS, sort=True, transform=val_transform)
    test_ds  = GroupActivityDataset(
        VIDEOS_PATH, ANNOT_PATH, TEST_SPLIT,
        GROUP_LABELS, sort=True, transform=val_transform)

    print(f"  Train:{len(train_ds):,}  Val:{len(val_ds):,}  Test:{len(test_ds):,}")

    train_loader = DataLoader(train_ds, batch_size=GA_BS, shuffle=True,
                              num_workers=4, pin_memory=True, drop_last=True,
                              collate_fn=group_collate_fn)
    val_loader   = DataLoader(val_ds,   batch_size=GA_BS, shuffle=False,
                              num_workers=4, pin_memory=True,
                              collate_fn=group_collate_fn)
    test_loader  = DataLoader(test_ds,  batch_size=GA_BS, shuffle=False,
                              num_workers=4, pin_memory=True,
                              collate_fn=group_collate_fn)

    model = GroupActivityClassifier(
        person_model, GA_HIDDEN, GA_LAYERS, GA_NUM_CLASSES).to(DEVICE)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_p   = sum(p.numel() for p in model.parameters())
    print(f"  Params  total={total_p:,}  trainable={trainable:,}  "
          f"frozen={total_p-trainable:,}")

    # always compute class weights (consistent fresh & resume)
    print("  Computing class weights …")
    class_weights = compute_class_weights(train_loader, GA_NUM_CLASSES, DEVICE)
    criterion = nn.CrossEntropyLoss(
        label_smoothing=GA_SMOOTHING, weight=class_weights)

    optimizer = optim.AdamW(model.parameters(), lr=GA_LR_FRESH, weight_decay=GA_WD)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.1, patience=3)
    scaler    = GradScaler()

    start_epoch  = 1
    best_val_acc = 0.0
    history      = {k: [] for k in
                    ["train_loss","train_acc","val_loss","val_acc","val_f1","lr"]}

    if RESUME_PHASE_B and os.path.isfile(PHASE_B_RESUME_CKPT):
        print(f"  Resuming from: {PHASE_B_RESUME_CKPT}")
        model, optimizer, last_epoch, best_val_acc, saved_history = \
            load_checkpoint(PHASE_B_RESUME_CKPT, model, optimizer, DEVICE)
        start_epoch = last_epoch + 1
        if saved_history: history = saved_history
        for v in history.get("val_loss", []): scheduler.step(v)
        print(f"  Continuing from epoch {start_epoch}/{GA_EPOCHS}")
    else:
        print(f"  Starting fresh  ({GA_EPOCHS} epochs)")

    no_improve = 0
    for epoch in range(start_epoch, GA_EPOCHS + 1):
        model.train()
        run_loss, correct, total = 0.0, 0, 0

        pbar = tqdm(train_loader,
                    desc=f"[Phase-B] Ep {epoch:2d}/{GA_EPOCHS}",
                    unit="batch", leave=False,
                    bar_format="{l_bar}{bar:25}{r_bar}")
        for x, y in pbar:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            with autocast():
                out  = model(x); loss = criterion(out, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer); scaler.update()
            run_loss += loss.item()
            correct  += (out.argmax(1) == y.argmax(1)).sum().item()
            total    += y.size(0)
            pbar.set_postfix(loss=f"{loss.item():.4f}",
                             acc=f"{100*correct/total:.1f}%")
        pbar.close()

        train_loss = run_loss / len(train_loader)
        train_acc  = correct / total
        val_loss, val_acc, val_f1, val_gt, val_pred = evaluate(
            model, val_loader, criterion, DEVICE)
        scheduler.step(val_loss)
        lr = optimizer.param_groups[0]["lr"]

        for k, v in zip(history,
                        [train_loss, train_acc, val_loss,
                         val_acc, val_f1, lr]):
            history[k].append(v)

        if val_acc > best_val_acc:
            no_improve   = 0
            best_val_acc = val_acc
            save_checkpoint(PHASE_B_CKPT_OUT, model, optimizer,
                            epoch, val_acc, {"phase": "B"},
                            history, best_val_acc)
        else:
            no_improve += 1

        latest = os.path.join(OUTPUT_DIR, "phase_b_epoch_latest.pkl")
        save_checkpoint(latest, model, optimizer,
                        epoch, val_acc, {"phase": "B"},
                        history, best_val_acc)

        epoch_report("Phase-B", epoch, GA_EPOCHS,
                     train_loss, train_acc, val_loss, val_acc,
                     val_f1, lr, best_val_acc, no_improve,
                     val_targets=val_gt, val_preds=val_pred)

        # ── Early diagnostic at DIAG_EPOCH ───────────────────
        if epoch == DIAG_EPOCH:
            early_diagnostic(model, val_loader, criterion, DEVICE,
                             epoch, GROUP_CLASSES, OUTPUT_DIR)
            print(f"  ℹ️  Set GA_EPOCHS = 50 and re-run to continue training.")

        torch.cuda.empty_cache()

    plot_training_curves_phase_b(
        history, os.path.join(OUTPUT_DIR, "phase_b_curves.png"))

    # final evaluation
    model, _, _, _, _ = load_checkpoint(
        PHASE_B_CKPT_OUT, model, device=DEVICE)

    print("\n" + "="*65)
    print("  FINAL EVALUATION — VALIDATION SET")
    print("="*65)
    val_loss_f, val_acc_f, val_f1_f, val_gt, val_pred = evaluate(
        model, val_loader, criterion, DEVICE)
    plot_confusion_matrices(val_gt, val_pred, GROUP_CLASSES,
        "Baseline B8 — Validation Confusion Matrix",
        os.path.join(OUTPUT_DIR, "b8_val_confusion_matrix.png"))
    plot_per_class_accuracy(val_gt, val_pred, GROUP_CLASSES,
        "Baseline B8 — Validation Per-Class Accuracy",
        os.path.join(OUTPUT_DIR, "b8_val_per_class_acc.png"))
    print_and_save_report(val_gt, val_pred, GROUP_CLASSES,
        "Validation set", val_acc_f, val_f1_f, val_loss_f,
        os.path.join(OUTPUT_DIR, "b8_val_report.txt"))

    print("\n" + "="*65)
    print("  FINAL EVALUATION — TEST SET")
    print("="*65)
    test_loss, test_acc, test_f1, test_gt, test_pred = evaluate(
        model, test_loader, criterion, DEVICE)
    print(f"  Test Accuracy : {test_acc*100:.2f}%")
    print(f"  Test F1       : {test_f1:.4f}")
    plot_confusion_matrices(test_gt, test_pred, GROUP_CLASSES,
        "Baseline B8 — Test Confusion Matrix",
        os.path.join(OUTPUT_DIR, "b8_test_confusion_matrix.png"))
    plot_per_class_accuracy(test_gt, test_pred, GROUP_CLASSES,
        "Baseline B8 — Test Per-Class Accuracy",
        os.path.join(OUTPUT_DIR, "b8_test_per_class_acc.png"))
    print_and_save_report(test_gt, test_pred, GROUP_CLASSES,
        "Test set", test_acc, test_f1, test_loss,
        os.path.join(OUTPUT_DIR, "b8_test_report.txt"))

    return model


# ── 15. Main ──────────────────────────────────────────────────

if __name__ == "__main__":
    set_seed(SEED)

    if SKIP_PHASE_A:
        print("  Skipping Phase A — loading weights …")
        person_model = PersonActivityClassifier(
            PA_NUM_CLASSES, PA_HIDDEN, PA_LAYERS).to(DEVICE)
        person_model = load_weights_only(PHASE_A_BEST_FOR_B, person_model, DEVICE)
    else:
        person_model = train_phase_a()
        person_model = load_weights_only(PHASE_A_CKPT_OUT, person_model, DEVICE)

    group_model = train_phase_b(person_model)

    print("\n  ✅ Baseline B8 complete.")
    print(f"  phase_a_best_b8.pkl          — best Phase A")
    print(f"  phase_a_epoch_latest.pkl     — latest Phase A (resume)")
    print(f"  phase_b_best_b8.pkl          — best Phase B")
    print(f"  phase_b_epoch_latest.pkl     — latest Phase B (resume)")
    print(f"  All outputs → {OUTPUT_DIR}")
