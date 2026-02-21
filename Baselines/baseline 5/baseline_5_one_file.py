"""
VOLLEYBALL GROUP ACTIVITY RECOGNITION - BASELINE B5
====================================================

Baseline B5: Temporal on Crops (LSTM on Player Level)
From the paper: "Similar 2 implementation paths to build representation per person"

Implementation Strategy:
- Phase A: Train person-level temporal LSTM (ResNet50 → LSTM → 9 action classes)
- Phase B: Use frozen Phase A features to train group classifier (8 classes)

Key Architecture (from paper):
  Phase A: ResNet50 → LSTM → Person Action Classifier (9 classes)
  Phase B: For each frame:
    1. Extract ResNet50 + LSTM features for each player (frozen from Phase A)
    2. Concatenate ResNet + LSTM features per player
    3. Use LAST hidden state as player representation
    4. Max pool across all 12 players
    5. Feed to group classifier (NN network like B3)

Difference from B3: B3 uses single-frame ResNet features, B5 uses temporal LSTM features
Difference from B4: B4 uses LSTM on frame sequences, B5 uses LSTM on person sequences
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
from tqdm import tqdm
from collections import Counter
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from torch.cuda.amp import autocast, GradScaler


import random
torch.manual_seed(31)
torch.cuda.manual_seed_all(31)
np.random.seed(31)
random.seed(31)
# ============================================================================
# GLOBAL CONFIGURATIONS
# ============================================================================

dataset_root = "/kaggle/input/volleyball"
videos_root = f"{dataset_root}/volleyball_/videos"
output_dir = "/kaggle/working"

# Phase A settings - Person Activity Temporal Classifier
PHASE_A_BATCH_SIZE = 2  # Lower batch size due to sequences
PHASE_A_EPOCHS = 10
PHASE_A_LR = 1e-5
LSTM_HIDDEN_SIZE = 128
LSTM_NUM_LAYERS = 1

# Phase B settings - Group Activity Classifier  
PHASE_B_BATCH_SIZE = 8  # Lower batch size due to 12 players × 9 frames
PHASE_B_EPOCHS = 30
PHASE_B_LR = 1e-4

NUM_PERSON_CLASSES = 9
NUM_GROUP_CLASSES = 8
SEQUENCE_LENGTH = 9  # 4 before + middle + 4 after

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================================
# CLASS CATEGORIES & DATA SPLITS
# ============================================================================

Categories = {
    'l-pass': 0,
    'r-pass': 1,
    'l-spike': 2,
    'r_spike': 3,
    'l_set': 4,
    'r_set': 5,
    'l_winpoint': 6,
    'r_winpoint': 7
}

Players_Categories = {
    'standing': 0,
    'moving': 1,
    'digging': 2,
    'setting': 3,
    'falling': 4,
    'blocking': 5,
    'spiking': 6,
    'jumping': 7,
    'waiting': 8
}

train_data = ["1","3","6","7","10","13","15","16","18","22","23","31",
              "32","36","38","39","40","41","42","48","50","52","53","54"]
val_data = ["0","2","8","12","17","19","24","26","27","28","30","33","46","49","51"]
test_data = ["4","5","9","11","14","20","21","25","29","34","35","37","43","44","45","47"]

# ============================================================================
# PHASE A DATASET - PERSON TEMPORAL SEQUENCES
# ============================================================================

class PersonTemporalDataset(Dataset):
    """
    Phase A: Individual player temporal sequences with action labels
    Each sample: one player's 9-frame sequence + their action label
    """
    
    def __init__(self, video_ids, transform):
        self.samples = []
        self.transform = transform
        self.skipped = 0
        
        for vid in video_ids:
            video_dir = os.path.join(videos_root, vid)
            annot_file = os.path.join(video_dir, "annotations.txt")
            
            if not os.path.isfile(annot_file):
                continue
            
            with open(annot_file, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 2:
                        continue
                    
                    frame_name = parts[0]
                    group_label_str = parts[1]
                    
                    if group_label_str not in Categories:
                        continue
                    
                    clip_id = frame_name.replace(".jpg", "")
                    
                    try:
                        frame_num = int(clip_id)
                    except:
                        self.skipped += 1
                        continue
                    
                    # Get 9 frames: 4 before + middle + 4 after
                    frame_numbers = [frame_num - 4 + i for i in range(9)]
                    clip_dir = os.path.join(video_dir, clip_id)
                    
                    # Check all 9 frames exist
                    all_exist = True
                    for fn in frame_numbers:
                        if not os.path.isfile(os.path.join(clip_dir, f"{fn}.jpg")):
                            all_exist = False
                            break
                    
                    if not all_exist:
                        self.skipped += 1
                        continue
                    
                    # Process all 12 players
                    num_players = 12
                    start = 2
                    
                    for idx in range(num_players):
                        base = start + idx * 5
                        try:
                            x = int(parts[base])
                            y = int(parts[base + 1])
                            w = int(parts[base + 2])
                            h = int(parts[base + 3])
                            action_str = parts[base + 4]
                            
                            if action_str not in Players_Categories:
                                continue
                            
                            action_label = Players_Categories[action_str]
                            
                            # Store: (video_dir, clip_id, frame_numbers, bbox, action_label)
                            self.samples.append((video_dir, clip_id, frame_numbers, (x, y, w, h), action_label))
                            
                        except:
                            continue
        
        print(f"Loaded {len(self.samples)} person temporal samples | Skipped {self.skipped}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        video_dir, clip_id, frame_numbers, (x, y, w, h), action_label = self.samples[idx]
        
        frames = []
        for fn in frame_numbers:
            img_path = os.path.join(video_dir, clip_id, f"{fn}.jpg")
            try:
                image = Image.open(img_path).convert("RGB")
                crop = image.crop((x, y, x + w, y + h))
                if self.transform:
                    crop = self.transform(crop)
                frames.append(crop)
            except Exception as e:
                # Use black frame as padding if crop fails
                frames.append(torch.zeros(3, 224, 224))
        
        frames = torch.stack(frames)  # [9, 3, 224, 224]
        return frames, action_label


# ============================================================================
# PHASE B DATASET - GROUP TEMPORAL SEQUENCES
# ============================================================================

class GroupTemporalDataset(Dataset):
    """
    Phase B: All player temporal sequences per frame with group activity label
    Each sample: 12 players × 9 frames + group activity label
    """
    
    def __init__(self, video_ids, transform):
        self.samples = []
        self.transform = transform
        self.skipped = 0
        
        for vid in video_ids:
            video_dir = os.path.join(videos_root, vid)
            annot_file = os.path.join(video_dir, "annotations.txt")
            
            if not os.path.isfile(annot_file):
                continue
            
            with open(annot_file, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 2:
                        continue
                    
                    frame_name = parts[0]
                    group_label_str = parts[1]
                    
                    if group_label_str not in Categories:
                        continue
                    
                    group_label = Categories[group_label_str]
                    clip_id = frame_name.replace(".jpg", "")
                    
                    try:
                        frame_num = int(clip_id)
                    except:
                        self.skipped += 1
                        continue
                    
                    # Get 9 frames: 4 before + middle + 4 after
                    frame_numbers = [frame_num - 4 + i for i in range(9)]
                    clip_dir = os.path.join(video_dir, clip_id)
                    
                    # Check all 9 frames exist
                    all_exist = True
                    for fn in frame_numbers:
                        if not os.path.isfile(os.path.join(clip_dir, f"{fn}.jpg")):
                            all_exist = False
                            break
                    
                    if not all_exist:
                        self.skipped += 1
                        continue
                    
                    # Collect ALL 12 player bboxes
                    player_bboxes = []
                    num_players = 12
                    start = 2
                    
                    for idx in range(num_players):
                        base = start + idx * 5
                        try:
                            x = int(parts[base])
                            y = int(parts[base + 1])
                            w = int(parts[base + 2])
                            h = int(parts[base + 3])
                            player_bboxes.append((x, y, w, h))
                        except:
                            # Use dummy bbox for missing players
                            player_bboxes.append((0, 0, 50, 50))
                    
                    self.samples.append((video_dir, clip_id, frame_numbers, player_bboxes, group_label))
        
        print(f"Loaded {len(self.samples)} group temporal samples | Skipped {self.skipped}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        video_dir, clip_id, frame_numbers, player_bboxes, group_label = self.samples[idx]
        
        # Load all 12 players × 9 frames
        all_player_sequences = []
        
        for (x, y, w, h) in player_bboxes:
            player_frames = []
            
            for fn in frame_numbers:
                img_path = os.path.join(video_dir, clip_id, f"{fn}.jpg")
                
                try:
                    image = Image.open(img_path).convert("RGB")
                    img_w, img_h = image.size
                    
                    # Handle invalid/padding boxes
                    if (x, y, w, h) == (0, 0, 50, 50):
                        crop = Image.new('RGB', (224, 224), (0, 0, 0))
                    else:
                        # Clip to image boundaries
                        x_clip = max(0, min(x, img_w-1))
                        y_clip = max(0, min(y, img_h-1))
                        w_clip = max(1, min(w, img_w-x_clip))
                        h_clip = max(1, min(h, img_h-y_clip))
                        crop = image.crop((x_clip, y_clip, x_clip + w_clip, y_clip + h_clip))
                    
                    if self.transform:
                        crop = self.transform(crop)
                    player_frames.append(crop)
                    
                except Exception as e:
                    player_frames.append(torch.zeros(3, 224, 224))
            
            player_frames = torch.stack(player_frames)  # [9, 3, 224, 224]
            all_player_sequences.append(player_frames)
        
        all_player_sequences = torch.stack(all_player_sequences)  # [12, 9, 3, 224, 224]
        return all_player_sequences, group_label


# ============================================================================
# TRANSFORMS
# ============================================================================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ============================================================================
# PHASE A MODEL - PERSON TEMPORAL CLASSIFIER
# ============================================================================

class Person_Activity_Temporal_Classifier(nn.Module):
    """
    Phase A: Person-level temporal action classifier
    Input: [batch, 9, 3, 224, 224] - one person's 9-frame sequence
    Output: [batch, 9] - person action logits
    """
    
    def __init__(self, num_classes=9, hidden_size=512, num_layers=1):
        super(Person_Activity_Temporal_Classifier, self).__init__()
        
        # ResNet50 feature extractor (remove FC layer)
        self.resnet50 = nn.Sequential(
            *list(models.resnet50(weights=models.ResNet50_Weights.DEFAULT).children())[:-1]
        )
        
        # LSTM for temporal modeling
        self.lstm = nn.LSTM(
            input_size=2048,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        
        # Classification head
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        """
        x: [batch, seq, c, h, w] or [batch*bbox, seq, c, h, w]
        """
        if x.dim() == 5:
            b, seq, c, h, w = x.shape
        else:
            # Handle case where input is [batch, bbox, seq, c, h, w]
            b_bb, seq, c, h, w = x.shape
            b = b_bb
        
        # Extract features from all frames
        x = x.view(b * seq, c, h, w)  # [b*seq, c, h, w]
        x = self.resnet50(x)  # [b*seq, 2048, 1, 1]
        x = x.view(b, seq, -1)  # [b, seq, 2048]
        
        # LSTM processing
        x, (h, c) = self.lstm(x)  # [b, seq, hidden_size]
        
        # Use last hidden state
        x = x[:, -1, :]  # [b, hidden_size]
        
        # Classification
        x = self.fc(x)  # [b, num_classes]
        
        return x


# ============================================================================
# PHASE B MODEL - GROUP ACTIVITY CLASSIFIER
# ============================================================================

class Group_Activity_Classifier(nn.Module):
    def __init__(self, person_feature_extraction, num_classes):
        super(Group_Activity_Classifier, self).__init__()

        self.resnet50 = person_feature_extraction.resnet50
        self.lstm = person_feature_extraction.lstm

        for module in [self.resnet50,  self.lstm]:
            for param in module.parameters():
                param.requires_grad = False
                
        self.pool = nn.AdaptiveMaxPool2d((1, 2048))  # [Batch, 12, hidden_size] -> [Batch, 1, 2048]
        
        self.fc = nn.Sequential(
            nn.Linear(2048, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes), 
        )
    
    def forward(self, x):
        # x.shape => batch, bbox, frames, channals , hight, width
        b, bb, seq, c, h, w = x.shape # seq => frames
        x = x.view(b*bb*seq, c, h, w) # (b * bb * seq, c, h, w)
        x1 = self.resnet50(x) # (batch * bbox * seq, 2048, 1 , 1)

        x1 = x1.view(b*bb, seq, -1) # (batch * bbox, seq, 2048)
        x2, (h , c) = self.lstm(x1) # (batch * bbox, seq, hidden_size)

        x = torch.cat([x1, x2], dim=2) # Concat the Resnet50 representation and LSTM layer for every  
        x = x.contiguous()             # person and pool over all people in a scene.
        x = x[:, -1, :]                # (batch * bbox, hidden_size)
        
        x = x.view(b, bb, -1) # (batch , bbox, hidden_size)
        x = self.pool(x) # (batch, 1, 2048)
        x = x.squeeze(dim=1) # (batch, 2048)

        x = self.fc(x) # (batch, num_class)
        return x

# ============================================================================
# TRAINING UTILITIES
# ============================================================================

def evaluate_model(model, loader, criterion):
    """Evaluate model on given loader"""
    model.eval()
    preds, labels_list = [], []
    total_loss = 0
    
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out, y)
            total_loss += loss.item()
            
            p = out.argmax(1)
            preds.extend(p.cpu().numpy())
            labels_list.extend(y.cpu().numpy())
    
    avg_loss = total_loss / len(loader)
    acc = accuracy_score(labels_list, preds)
    f1 = f1_score(labels_list, preds, average="weighted")
    return acc, f1, avg_loss, labels_list, preds


def save_checkpoint(path, model, optimizer, epoch, val_acc):
    """Save model checkpoint"""
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_acc": val_acc
    }, path)


def plot_training_curves(history, title_prefix, save_path):
    """Plot training history"""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss
    axes[0].plot(history['train_loss'], 'b-o', label='Train', linewidth=2, markersize=4)
    axes[0].plot(history['val_loss'], 'r-s', label='Val', linewidth=2, markersize=4)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title(f'{title_prefix} - Loss Curves', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy
    axes[1].plot(history['train_acc'], 'b-o', label='Train', linewidth=2, markersize=4)
    axes[1].plot(history['val_acc'], 'r-s', label='Val', linewidth=2, markersize=4)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Accuracy', fontsize=12)
    axes[1].set_title(f'{title_prefix} - Accuracy Curves', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_confusion_matrix(labels, preds, title, save_path, class_names):
    """Plot confusion matrix"""
    cm = confusion_matrix(labels, preds)
    cm_norm = cm.astype("float") / cm.sum(axis=1, keepdims=True)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Normalized Frequency'}
    )
    plt.xlabel("Predicted Label", fontsize=12)
    plt.ylabel("True Label", fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


# ============================================================================
# PHASE A: TRAIN PERSON TEMPORAL CLASSIFIER
# ============================================================================

def train_phase_a():
    print("\n" + "="*70)
    print("PHASE A: PERSON TEMPORAL ACTION CLASSIFIER (9 CLASSES)")
    print("="*70)
    print(f"Architecture: ResNet50 → LSTM({LSTM_HIDDEN_SIZE}) → FC(512) → 9 classes")
    print(f"Sequence length: {SEQUENCE_LENGTH} frames per person")
    print("="*70)
    
    # Load datasets
    print("\nLoading Phase A datasets...")
    train_ds = PersonTemporalDataset(train_data, transform)
    val_ds = PersonTemporalDataset(val_data, transform)
    
    train_loader = DataLoader(train_ds, batch_size=PHASE_A_BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=PHASE_A_BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
    
    print(f"Train: {len(train_ds)} samples | Val: {len(val_ds)} samples")
    
    # Class distribution
    print("\nPerson action class distribution in training set:")
    labels = [label for _, _, _, _, label in train_ds.samples]
    counts = Counter(labels)
    for name, idx in sorted(Players_Categories.items(), key=lambda x: x[1]):
        print(f"  {name:12s}: {counts[idx]} samples")
    
    # Build model
    print("\nBuilding Phase A model...")
    model = Person_Activity_Temporal_Classifier(
        num_classes=NUM_PERSON_CLASSES,
        hidden_size=LSTM_HIDDEN_SIZE,
        num_layers=LSTM_NUM_LAYERS
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=PHASE_A_LR, weight_decay=0.1)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)
    scaler = GradScaler()
    
    # Training loop
    print("\n" + "="*70)
    print("STARTING PHASE A TRAINING")
    print("="*70)
    
    best_val_acc = 0.0
    best_model_state = None
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [], 'val_f1': []
    }
    
    for epoch in range(PHASE_A_EPOCHS):
        model.train()
        running_loss = 0
        correct = 0
        total = 0
        
        for x, y in tqdm(train_loader, desc=f"Phase A Epoch {epoch+1}/{PHASE_A_EPOCHS}"):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            
            with autocast():
                out = model(x)
                loss = criterion(out, y)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            running_loss += loss.item()
            pred = out.argmax(1)
            correct += (pred == y).sum().item()
            total += y.size(0)
        
        train_loss = running_loss / len(train_loader)
        train_acc = correct / total
        val_acc, val_f1, val_loss, _, _ = evaluate_model(model, val_loader, criterion)
        
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_f1'].append(val_f1)
        
        print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f} | Train Acc={train_acc:.4f} | "
              f"Val Loss={val_loss:.4f} | Val Acc={val_acc:.4f} | Val F1={val_f1:.4f} | LR={current_lr:.6f}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict()
            print(f"  ✓ New best: {best_val_acc:.4f}")
            save_checkpoint(f"{output_dir}/phase_a_best.pkl", model, optimizer, epoch+1, best_val_acc)
    
    # Load best model
    model.load_state_dict(best_model_state)
    
    print("\n" + "="*70)
    print("PHASE A COMPLETED")
    print("="*70)
    print(f"Best Val Acc: {best_val_acc:.4f}")
    
    # Visualizations
    print("\nGenerating Phase A visualizations...")
    plot_training_curves(history, "Phase A - Person Temporal", f"{output_dir}/phase_a_training_history.png")
    
    # Save log
    with open(f"{output_dir}/phase_a_log.txt", "w") as f:
        f.write("BASELINE B5 - PHASE A: Person Temporal Action Classifier\n")
        f.write("="*70 + "\n\n")
        f.write(f"Architecture: ResNet50 → LSTM({LSTM_HIDDEN_SIZE}) → FC → 9 classes\n")
        f.write(f"Sequence length: {SEQUENCE_LENGTH} frames\n\n")
        f.write(f"Best Val Acc: {best_val_acc:.4f}\n")
    
    print("\n✓ Phase A Training completed!")
    print(f"✓ Checkpoint saved: phase_a_best.pkl")
    
    return model, best_val_acc


# ============================================================================
# PHASE B: TRAIN GROUP ACTIVITY CLASSIFIER
# ============================================================================

def train_phase_b(person_classifier):
    print("\n" + "="*70)
    print("PHASE B: GROUP ACTIVITY CLASSIFIER (8 CLASSES)")
    print("="*70)
    print("Architecture: Frozen(ResNet50 + LSTM) → Concat → MaxPool → FC → 8 classes")
    print("="*70)
    
    # Load datasets
    print("\nLoading Phase B datasets...")
    train_ds = GroupTemporalDataset(train_data, transform)
    val_ds = GroupTemporalDataset(val_data, transform)
    test_ds = GroupTemporalDataset(test_data, transform)
    
    train_loader = DataLoader(train_ds, batch_size=PHASE_B_BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=PHASE_B_BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=PHASE_B_BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
    
    print(f"Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")
    
    # Class distribution
    print("\nGroup activity class distribution in training set:")
    labels = [label for _, _, _, _, label in train_ds.samples]
    counts = Counter(labels)
    for name, idx in sorted(Categories.items(), key=lambda x: x[1]):
        print(f"  {name:12s}: {counts[idx]} samples")
    
    # Build model
    print("\nBuilding Phase B model...")
    model = Group_Activity_Classifier(person_classifier, num_classes=NUM_GROUP_CLASSES).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters:        {total_params:,}")
    print(f"Trainable (classifier):  {trainable_params:,}")
    print(f"Frozen (ResNet+LSTM):    {total_params - trainable_params:,}")
    
    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=PHASE_B_LR, weight_decay=1)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)
    scaler = GradScaler()
    
    # Training loop
    print("\n" + "="*70)
    print("STARTING PHASE B TRAINING")
    print("="*70)
    
    best_val_acc = 0.0
    best_model_state = None
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [], 'val_f1': []
    }
    
    for epoch in range(PHASE_B_EPOCHS):
        model.train()
        running_loss = 0
        correct = 0
        total = 0
        
        for x, y in tqdm(train_loader, desc=f"Phase B Epoch {epoch+1}/{PHASE_B_EPOCHS}"):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            
            with autocast():
                out = model(x)
                loss = criterion(out, y)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            running_loss += loss.item()
            pred = out.argmax(1)
            correct += (pred == y).sum().item()
            total += y.size(0)
        
        train_loss = running_loss / len(train_loader)
        train_acc = correct / total
        val_acc, val_f1, val_loss, _, _ = evaluate_model(model, val_loader, criterion)
        
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_f1'].append(val_f1)
        
        print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f} | Train Acc={train_acc:.4f} | "
              f"Val Loss={val_loss:.4f} | Val Acc={val_acc:.4f} | Val F1={val_f1:.4f} | LR={current_lr:.6f}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict()
            print(f"  ✓ New best: {best_val_acc:.4f}")
            save_checkpoint(f"{output_dir}/phase_b_best.pkl", model, optimizer, epoch+1, best_val_acc)
    
    # Test evaluation
    print("\n" + "="*70)
    print("PHASE B TEST EVALUATION")
    print("="*70)
    
    model.load_state_dict(best_model_state)
    test_acc, test_f1, test_loss, test_labels, test_preds = evaluate_model(model, test_loader, criterion)
    
    print(f"\nB5 Test Results:")
    print(f"  Accuracy: {test_acc*100:.2f}%")
    print(f"  F1 Score: {test_f1:.4f}")
    print(f"  Loss:     {test_loss:.4f}")
    
    # Classification report
    print("\nClassification Report:")
    report = classification_report(test_labels, test_preds, target_names=list(Categories.keys()), digits=4)
    print(report)
    
    # Per-class accuracy
    print("\nPer-Class Accuracy:")
    for name, idx in sorted(Categories.items(), key=lambda x: x[1]):
        mask = [l == idx for l in test_labels]
        if sum(mask) > 0:
            class_acc = accuracy_score(
                [test_labels[i] for i, m in enumerate(mask) if m],
                [test_preds[i] for i, m in enumerate(mask) if m]
            )
            print(f"  {name:12s}: {class_acc:.4f} ({sum(mask)} samples)")
    
    # Visualizations
    print("\nGenerating Phase B visualizations...")
    plot_training_curves(history, "Phase B - Group Activity", f"{output_dir}/phase_b_training_history.png")
    plot_confusion_matrix(test_labels, test_preds, "B5 - Group Activity Confusion Matrix", 
                         f"{output_dir}/b5_confusion_matrix.png", list(Categories.keys()))
    
    # Save log
    with open(f"{output_dir}/phase_b_log.txt", "w") as f:
        f.write("BASELINE B5 - PHASE B: Group Activity Classifier\n")
        f.write("="*70 + "\n\n")
        f.write(f"Architecture: Frozen(ResNet50 + LSTM) → Concat → MaxPool → FC\n")
        f.write(f"12 players × {SEQUENCE_LENGTH} frames per sample\n\n")
        f.write(f"Best Val Acc: {best_val_acc:.4f}\n")
        f.write(f"Test Acc:     {test_acc:.4f}\n")
        f.write(f"Test F1:      {test_f1:.4f}\n\n")
        f.write(report)
    
    print("\n✓ Phase B Training completed!")
    print(f"✓ Checkpoint saved: phase_b_best.pkl")
    
    return best_val_acc, test_acc, test_f1


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("STARTING BASELINE B5 TRAINING")
    print("="*70)
    print("\nB5 Key Features:")
    print("  ✓ Temporal modeling at person level (LSTM)")
    print("  ✓ 9-frame sequences per person")
    print("  ✓ Concatenate ResNet + LSTM features")
    print("  ✓ Max pooling across 12 players")
    print("  ✓ Two-phase training (person → group)")
    print("="*70)
    
    # Phase A: Train person temporal classifier
    person_classifier, phase_a_val_acc = train_phase_a()
    
    # Phase B: Train group activity classifier
    phase_b_val_acc, test_acc, test_f1 = train_phase_b(person_classifier)
    
    # Final Summary
    print("\n" + "="*70)
    print("FINAL SUMMARY - BASELINE B5")
    print("="*70)
    print("\nPhase A (Person Temporal):")
    print(f"  Best Val Acc: {phase_a_val_acc:.4f}")
    
    print("\nPhase B (Group Activity):")
    print(f"  Best Val Acc: {phase_b_val_acc:.4f}")
    print(f"  Test Acc:     {test_acc:.4f} ({test_acc*100:.2f}%)")
    print(f"  Test F1:      {test_f1:.4f}")
    
    print("\n" + "-"*70)
    print("COMPARISON WITH OTHER BASELINES:")
    print("-"*70)
    print(f"  B1 (ResNet50 single-frame):     79.88%")
    print(f"  B3 (Two-phase spatial):         ~73-75%")
    print(f"  B4 (Frame-level LSTM):          ~75-77%")
    print(f"  B5 (Person-level LSTM):         {test_acc*100:.2f}%")
    print("\nB5 Architecture Summary:")
    print("  - Phase A: ResNet50 → LSTM on each person's 9-frame sequence")
    print("  - Phase B: Concat(ResNet+LSTM) → MaxPool(12 players) → Classifier")
    print("  - Key: Uses TEMPORAL features at PERSON level (not frame level)")
    print("  - Expected: Similar or better than B4 (~77% from paper)")
    
    if test_acc >= 0.77:
        improvement = (test_acc - 0.77) * 100
        print(f"\n✓ SUCCESS: Matches/exceeds paper's B5 performance!")
        if improvement > 0:
            print(f"  Improvement: +{improvement:.2f}% above 77% baseline")
    else:
        gap = (0.77 - test_acc) * 100
        print(f"\n⚠ Performance: {gap:.2f}% below paper's 77% baseline")
        print("  Possible reasons: fewer epochs, different hyperparameters, or data variations")
    
    print("\n" + "="*70)
    print("FILES SAVED:")
    print("="*70)
    print("Phase A Files:")
    print("  - phase_a_best.pkl")
    print("  - phase_a_training_history.png")
    print("  - phase_a_log.txt")
    print("\nPhase B Files:")
    print("  - phase_b_best.pkl")
    print("  - phase_b_training_history.png")
    print("  - b5_confusion_matrix.png")
    print("  - phase_b_log.txt")
    print("="*70)
    print("\n✓ BASELINE B5 TRAINING COMPLETED!")
    print("Note: Download files from 'Output' tab on the right →")