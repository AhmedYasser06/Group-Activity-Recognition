"""
Baseline B6 - Two-stage Model WITHOUT LSTM 1 (FINAL CORRECTED VERSION)
========================================================================
This baseline is a variant of the full model, omitting the person-level temporal 
model (LSTM 1). Instead, the person-level classification is done only with the 
fine-tuned person CNN.

Architecture:
- Phase A: Train person action classifier (9 classes) using ResNet50
- Phase B: Extract features per person → Max pool across players → LSTM on image level → Group activity (8 classes)

Key difference from B3:
- B3: No temporal modeling at all
- B6: Temporal modeling at IMAGE level only (LSTM on pooled features)
- B7: Full two-stage model (LSTM on person level + LSTM on image level)

Dataset: Volleyball Dataset
Paper: Ibrahim et al., CVPR 2016

CORRECTIONS:
- Fixed GaussNoise to match reference (no var_limit parameter)
- Removed verbose from scheduler (not supported in older PyTorch)
- Changed scheduler to monitor val_loss (mode='min') like reference
- Fixed LR tracking to capture BEFORE scheduler step
- Matched augmentation strategy to reference
"""

import os
import torch
import numpy as np
import torch.nn as nn
from PIL import Image
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torchvision import models
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

# ============================================================================
# CONFIGURATIONS
# ============================================================================

dataset_root = "/kaggle/input/volleyball"
videos_root = f"{dataset_root}/volleyball_/videos"
output_dir = "/kaggle/working"

# Load Phase A checkpoint
PHASE_A_CHECKPOINT = "/kaggle/input/notebooks/ahmedyasser06/b6-volleyball/phase_a_best.pkl"

# Phase A settings - Already trained, we'll load it
PHASE_A_EPOCHS = 5
PHASE_A_BATCH_SIZE = 64
PHASE_A_LR = 1e-3

# Phase B settings - LSTM on image level
PHASE_B_EPOCHS = 15  
PHASE_B_BATCH_SIZE = 8   # Adjusted for memory efficiency
PHASE_B_LR = 1e-4

# LSTM settings
LSTM_HIDDEN_SIZE = 512
SEQUENCE_LENGTH = 9  # Middle frame + 4 before + 4 after

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Output directory: {output_dir}")
print(f"Device: {device}")
print("Note: Download files from 'Output' tab on the right after training →")

assert os.path.isdir(videos_root), "Videos path is WRONG"
print("Videos path OK:", videos_root)

# ============================================================================
# CLASS CATEGORIES
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
# DATASET CLASSES (FIXED TO MATCH REFERENCE)
# ============================================================================

class PersonActionDataset(Dataset):
    """Phase A: Individual player crops with action labels (single frame)"""
    
    def __init__(self, video_ids, transform, use_onehot=True):
        self.samples = []
        self.transform = transform
        self.use_onehot = use_onehot
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
                    img_path = os.path.join(video_dir, clip_id, frame_name)
                    
                    if not os.path.isfile(img_path):
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
                                self.skipped += 1
                                continue
                            
                            action_label = Players_Categories[action_str]
                            self.samples.append((img_path, (x, y, w, h), action_label))
                            
                        except:
                            self.skipped += 1
                            continue
        
        print(f"Loaded {len(self.samples)} player samples | Skipped {self.skipped}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, box, label = self.samples[idx]
        
        image = Image.open(img_path).convert("RGB")
        x, y, w, h = box
        image = image.crop((x, y, x + w, y + h))
        
        # Convert PIL to numpy for albumentations
        image = np.array(image)
        
        if self.transform:
            transformed = self.transform(image=image)
            image = transformed['image']
        
        if self.use_onehot:
            label_onehot = torch.zeros(9, dtype=torch.float32)
            label_onehot[label] = 1.0
            return image, label_onehot
        
        return image, label


class GroupActivitySequenceDataset(Dataset):
    """Phase B: Sequence of frames with all player crops (TEMPORAL)
    
    Returns format: (12, 9, C, H, W) matching reference implementation
    """
    
    def __init__(self, video_ids, transform, use_onehot=True, seq_length=9):
        self.samples = []
        self.transform = transform
        self.use_onehot = use_onehot
        self.seq_length = seq_length
        self.skipped = 0
        
        # First, collect all annotations per video
        for vid in video_ids:
            video_dir = os.path.join(videos_root, vid)
            annot_file = os.path.join(video_dir, "annotations.txt")
            
            if not os.path.isfile(annot_file):
                continue
            
            # Parse annotations
            annotations = {}
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
                    frame_num = int(clip_id)
                    
                    # Collect player bboxes
                    bboxes = []
                    num_players = 12
                    start = 2
                    
                    valid = True
                    for idx in range(num_players):
                        base = start + idx * 5
                        try:
                            x = int(parts[base])
                            y = int(parts[base + 1])
                            w = int(parts[base + 2])
                            h = int(parts[base + 3])
                            bboxes.append((x, y, w, h))
                        except:
                            valid = False
                            break
                    
                    if valid and len(bboxes) == 12:
                        annotations[frame_num] = {
                            'group_label': Categories[group_label_str],
                            'bboxes': bboxes,
                            'clip_id': clip_id,
                            'video_id': vid
                        }
            
            # Create sequences (middle frame + 4 before + 4 after)
            sorted_frames = sorted(annotations.keys())
            for frame_num in sorted_frames:
                # Get sequence frames
                half_seq = self.seq_length // 2
                seq_frames = []
                
                for offset in range(-half_seq, half_seq + 1):
                    seq_frame = frame_num + offset
                    if seq_frame in annotations:
                        seq_frames.append(seq_frame)
                    else:
                        # If frame doesn't exist, use middle frame (for padding)
                        seq_frames.append(frame_num)
                
                if len(seq_frames) == self.seq_length:
                    self.samples.append({
                        'video_id': vid,
                        'frames': seq_frames,
                        'annotations': annotations,
                        'group_label': annotations[frame_num]['group_label']
                    })
        
        print(f"Loaded {len(self.samples)} temporal sequences | Skipped {self.skipped}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        video_id = sample['video_id']
        frames = sample['frames']
        annotations = sample['annotations']
        group_label = sample['group_label']
        
        video_dir = os.path.join(videos_root, video_id)
        
        # Collect sequence of player crops
        sequence_crops = []  # [seq_length, num_players, C, H, W]
        
        for frame_num in frames:
            annot = annotations[frame_num]
            clip_id = annot['clip_id']
            bboxes = annot['bboxes']
            
            frame_name = f"{clip_id}.jpg"
            img_path = os.path.join(video_dir, clip_id, frame_name)
            
            if not os.path.isfile(img_path):
                # Use black image if frame doesn't exist
                image = np.zeros((720, 1280, 3), dtype=np.uint8)
            else:
                image = Image.open(img_path).convert("RGB")
                image = np.array(image)
            
            img_h, img_w = image.shape[:2]
            
            player_crops = []
            for (x, y, w, h) in bboxes:
                # Clip to image boundaries
                x = max(0, min(x, img_w-1))
                y = max(0, min(y, img_h-1))
                w = max(1, min(w, img_w-x))
                h = max(1, min(h, img_h-y))
                
                crop = image[y:y+h, x:x+w]
                
                if self.transform:
                    transformed = self.transform(image=crop)
                    crop = transformed['image']
                
                player_crops.append(crop)
            
            player_crops = torch.stack(player_crops)  # [12, C, H, W]
            sequence_crops.append(player_crops)
        
        sequence_crops = torch.stack(sequence_crops)  # [seq_length, 12, C, H, W]
        
        # Transpose to match reference format: (12, 9, C, H, W)
        sequence_crops = sequence_crops.permute(1, 0, 2, 3, 4)
        
        if self.use_onehot:
            label_onehot = torch.zeros(8, dtype=torch.float32)
            label_onehot[group_label] = 1.0
            return sequence_crops, label_onehot
        
        return sequence_crops, group_label


# ============================================================================
# MODEL ARCHITECTURES
# ============================================================================

class Person_Activity_Classifier(nn.Module):
    """Phase A: Person-level action classifier (9 classes)"""
    
    def __init__(self, num_classes=9):
        super(Person_Activity_Classifier, self).__init__()
        self.resnet50 = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.in_features = self.resnet50.fc.in_features
        
        self.resnet50.fc = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(in_features=self.in_features, out_features=num_classes)
        )
    
    def forward(self, x):
        return self.resnet50(x)


class Group_Activity_Classifier_Temporal(nn.Module):
    """
    Phase B: Group activity classifier with TEMPORAL modeling
    
    Architecture:
    1. Extract features per person per frame using pre-trained ResNet50 (frozen)
    2. Max pool across players for each frame → [batch, seq_length, 2048]
    3. LSTM on temporal sequence → [batch, seq_length, hidden_size]
    4. Take last timestep → [batch, hidden_size]
    5. FC layers → [batch, 8]
    """
    
    def __init__(self, person_feature_extractor, hidden_size=512, num_classes=8):
        super(Group_Activity_Classifier_Temporal, self).__init__()
        
        # Extract feature extractor (without FC)
        self.feature_extraction = nn.Sequential(*list(person_feature_extractor.resnet50.children())[:-1])
        
        # Freeze feature extractor
        for param in self.feature_extraction.parameters():
            param.requires_grad = False
        
        # Max pooling across players
        self.pool = nn.AdaptiveMaxPool2d((1, 2048))  # [num_players, 2048] -> [1, 2048]
        
        # LSTM for temporal modeling
        self.lstm = nn.LSTM(
            input_size=2048,
            hidden_size=hidden_size,
            batch_first=True,
            num_layers=1
        )
        
        # Classifier head
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes),
        )
    
    def forward(self, x):
        # Input: [batch, bbox=12, seq=9, C, H, W]
        b, bb, seq, c, h, w = x.shape
        
        # Reshape for feature extraction
        x = x.view(b*bb*seq, c, h, w)  # [b*bb*seq, c, h, w]
        
        with torch.no_grad():
            with autocast():
                x = self.feature_extraction(x)  # [b*bb*seq, 2048, 1, 1]
        
        # Reshape for pooling across players
        x = x.view(b*seq, bb, -1)  # [b*seq, bb, 2048]
        x = self.pool(x)  # [b*seq, 1, 2048]
        
        # Reshape for LSTM
        x = x.squeeze(dim=1)  # [b*seq, 2048]
        x = x.view(b, seq, -1)  # [b, seq, 2048]
        
        # LSTM temporal modeling
        x, (h, c) = self.lstm(x)  # [b, seq, hidden]
        x = x[:, -1, :]  # [b, hidden] - take last timestep
        
        # Classification
        x = self.fc(x)  # [b, num_classes]
        return x


def collate_fn(batch):
    """
    Collate function to pad bounding boxes to 12 per frame and handle labels.
    Input: List of (clip, label) where clip is (12, 9, C, H, W)
    Output: (batch, 12, 9, C, H, W), (batch,) for integer labels OR (batch, 8) for one-hot
    """
    clips, labels = zip(*batch)
    
    max_bboxes = 12
    padded_clips = []
    
    for clip in clips:
        num_bboxes = clip.size(0)
        if num_bboxes < max_bboxes:
            # Pad with zeros if less than 12 players
            clip_padding = torch.zeros((max_bboxes - num_bboxes, clip.size(1), clip.size(2), clip.size(3), clip.size(4)))
            clip = torch.cat((clip, clip_padding), dim=0)
        
        padded_clips.append(clip)
    
    padded_clips = torch.stack(padded_clips)
    
    # Handle both integer labels and one-hot encoded labels
    if isinstance(labels[0], torch.Tensor):
        labels = torch.stack(labels)
    else:
        # Convert integer labels to tensor
        labels = torch.tensor(labels, dtype=torch.long)
    
    return padded_clips, labels


# ============================================================================
# TRANSFORMS - MATCHING REFERENCE TRAINER
# ============================================================================

# Phase A: Simple transforms (already trained)
train_transforms_phase_a = A.Compose([
    A.Resize(224, 224),
    A.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
    ToTensorV2()
])

# Phase B: Matching reference trainer augmentations
train_transforms_phase_b = A.Compose([
    A.Resize(224, 224),
    A.OneOf([
        A.GaussianBlur(blur_limit=(3, 7)),
        A.ColorJitter(brightness=0.2),
        A.RandomBrightnessContrast(),
        A.GaussNoise()  # FIXED: No parameters (matches reference)
    ], p=0.5),
    A.OneOf([
        A.HorizontalFlip(),
        A.VerticalFlip(),
    ], p=0.05),
    A.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
    ToTensorV2()
])

# Test/Val: No augmentation
simple_transform = A.Compose([
    A.Resize(224, 224),
    A.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
    ToTensorV2()
])


# ============================================================================
# TRAINING UTILITIES
# ============================================================================

def evaluate_model(model, loader, criterion):
    model.eval()
    preds, labels_list = [], []
    total_loss = 0
    
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            with autocast():
                out = model(x)
                loss = criterion(out, y)
                total_loss += loss.item()
            
            p = out.argmax(1)
            # Handle both integer labels and one-hot labels
            if y.dim() > 1:  # One-hot encoded
                y_class = y.argmax(1)
            else:  # Integer labels
                y_class = y
            
            preds.extend(p.cpu().numpy())
            labels_list.extend(y_class.cpu().numpy())
    
    avg_loss = total_loss / len(loader)
    acc = accuracy_score(labels_list, preds)
    f1 = f1_score(labels_list, preds, average="weighted")
    return acc, f1, avg_loss, labels_list, preds


def save_checkpoint(path, model, optimizer, epoch, val_acc):
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_acc": val_acc
    }, path)


def load_checkpoint(path, model, optimizer=None):
    """Load checkpoint and return model"""
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']} with val_acc={checkpoint['val_acc']:.4f}")
    return model


# ============================================================================
# PHASE B: TRAIN GROUP ACTIVITY CLASSIFIER WITH TEMPORAL LSTM
# ============================================================================

def train_phase_b(person_classifier):
    print("\n" + "="*70)
    print("PHASE B: GROUP ACTIVITY CLASSIFIER WITH TEMPORAL LSTM (8 CLASSES)")
    print("="*70)
    print("Architecture: Features → Max Pool Players → LSTM → Classifier")
    print("="*70)
    
    train_ds = GroupActivitySequenceDataset(train_data, train_transforms_phase_b, use_onehot=True, seq_length=SEQUENCE_LENGTH)
    val_ds = GroupActivitySequenceDataset(val_data, simple_transform, use_onehot=True, seq_length=SEQUENCE_LENGTH)
    test_ds = GroupActivitySequenceDataset(test_data, simple_transform, use_onehot=True, seq_length=SEQUENCE_LENGTH)
    
    train_loader = DataLoader(train_ds, batch_size=PHASE_B_BATCH_SIZE, collate_fn=collate_fn, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=PHASE_B_BATCH_SIZE, collate_fn=collate_fn, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=PHASE_B_BATCH_SIZE, collate_fn=collate_fn, shuffle=False, num_workers=4, pin_memory=True)
    
    print(f"Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")
    
    model = Group_Activity_Classifier_Temporal(
        person_classifier, 
        hidden_size=LSTM_HIDDEN_SIZE, 
        num_classes=8
    ).to(device)
    
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Total: {total:,} | Trainable: {trainable:,} | Frozen: {total-trainable:,}")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=PHASE_B_LR, weight_decay=1e-3)
    
    # FIXED: Match reference - mode='min', no verbose, factor=0.1, patience=2
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',  # Monitor loss (like reference)
        factor=0.1,
        patience=2
    )
    
    scaler = GradScaler()
    
    best_val_acc = 0.0
    best_model_state = None
    history = {
        'train_loss': [], 
        'train_acc': [], 
        'val_loss': [], 
        'val_acc': [], 
        'val_f1': [],
        'lr': []  # Track learning rate
    }
    
    print("\nTraining Phase B...")
    for epoch in range(PHASE_B_EPOCHS):
        model.train()
        running_loss = 0
        correct = 0
        total = 0
        
        for x, y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{PHASE_B_EPOCHS}"):
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
            # Handle both integer labels and one-hot labels
            if y.dim() > 1:  # One-hot encoded
                y_class = y.argmax(1)
            else:  # Integer labels
                y_class = y
            correct += (pred == y_class).sum().item()
            total += y.size(0)
        
        train_loss = running_loss / len(train_loader)
        train_acc = correct / total
        val_acc, val_f1, val_loss, _, _ = evaluate_model(model, val_loader, criterion)
        
        # FIXED: Get current LR BEFORE scheduler step
        current_lr = optimizer.param_groups[0]['lr']
        
        # FIXED: Step scheduler with val_loss (matches reference)
        scheduler.step(val_loss)
        
        # Detect LR change
        new_lr = optimizer.param_groups[0]['lr']
        if new_lr != current_lr:
            print(f"  → LR reduced: {current_lr:.6f} → {new_lr:.6f}")
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_f1'].append(val_f1)
        history['lr'].append(new_lr)  # Save new LR
        
        print(f"Epoch {epoch+1}: Train Acc={train_acc:.4f} | Val Acc={val_acc:.4f} | Val F1={val_f1:.4f} | LR={new_lr:.6f}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict()
            print(f"  ✓ Best: {best_val_acc:.4f}")
            save_checkpoint(f"{output_dir}/phase_b_best.pkl", model, optimizer, epoch+1, best_val_acc)
    
    model.load_state_dict(best_model_state)
    
    print("\n" + "="*70)
    print("FINAL TEST EVALUATION")
    print("="*70)
    
    test_acc, test_f1, test_loss, test_labels, test_preds = evaluate_model(model, test_loader, criterion)
    
    print(f"\nTest Accuracy: {test_acc*100:.2f}%")
    print(f"Test F1 Score: {test_f1:.4f}")
    print(f"Test Loss:     {test_loss:.4f}")
    
    print("\nClassification Report:")
    report = classification_report(test_labels, test_preds, target_names=list(Categories.keys()), digits=4)
    print(report)
    
    print("\nPer-Class Accuracy:")
    for name, idx in sorted(Categories.items(), key=lambda x: x[1]):
        mask = [l == idx for l in test_labels]
        if sum(mask) > 0:
            class_acc = accuracy_score([test_labels[i] for i, m in enumerate(mask) if m],
                                      [test_preds[i] for i, m in enumerate(mask) if m])
            print(f"  {name:12s}: {class_acc:.4f} ({sum(mask)} samples)")
    
    # Confusion Matrix
    cm = confusion_matrix(test_labels, test_preds)
    cm_norm = cm.astype("float") / cm.sum(axis=1, keepdims=True)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=list(Categories.keys()),
                yticklabels=list(Categories.keys()),
                cbar_kws={'label': 'Frequency'})
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Baseline B6 - Confusion Matrix")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/b6_confusion_matrix.png", dpi=150)
    plt.show()
    
    # Training curves with LR visualization
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Loss curves
    axes[0].plot(history['train_loss'], 'b-', label='Train', linewidth=2)
    axes[0].plot(history['val_loss'], 'r-', label='Val', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Loss Curves', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy curves
    axes[1].plot(history['train_acc'], 'b-', label='Train', linewidth=2)
    axes[1].plot(history['val_acc'], 'r-', label='Val', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Accuracy', fontsize=12)
    axes[1].set_title('Accuracy Curves', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    # Learning Rate curve
    axes[2].plot(history['lr'], 'g-', linewidth=2, marker='o', markersize=4)
    axes[2].set_xlabel('Epoch', fontsize=12)
    axes[2].set_ylabel('Learning Rate', fontsize=12)
    axes[2].set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
    axes[2].set_yscale('log')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/b6_training_history.png", dpi=150)
    plt.show()
    
    # Save detailed log
    with open(f"{output_dir}/b6_log.txt", "w") as f:
        f.write(f"Baseline B6 - Two-stage Model WITHOUT LSTM 1\n")
        f.write(f"="*70 + "\n")
        f.write(f"Phase A: Person action classifier (loaded from checkpoint)\n")
        f.write(f"Phase B: LSTM on IMAGE level (no person-level LSTM)\n")
        f.write(f"Sequence: {SEQUENCE_LENGTH} frames\n")
        f.write(f"LSTM Hidden: {LSTM_HIDDEN_SIZE}\n")
        f.write(f"Augmentations: GaussianBlur, ColorJitter, RandomBrightnessContrast, GaussNoise, HFlip, VFlip\n\n")
        f.write(f"Best Val Acc: {best_val_acc:.4f}\n")
        f.write(f"Test Acc:     {test_acc:.4f}\n")
        f.write(f"Test F1:      {test_f1:.4f}\n\n")
        f.write(report)
    
    return test_acc, test_f1, history


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":

    # Load Phase A model
    print("\nLoading Phase A checkpoint...")
    person_classifier = Person_Activity_Classifier(num_classes=9)
    if os.path.exists(PHASE_A_CHECKPOINT):
        person_classifier = load_checkpoint(PHASE_A_CHECKPOINT, person_classifier)
    else:
        print("ERROR: Phase A checkpoint not found! Please run Phase A training first.")
        exit(1)
    
    person_classifier = person_classifier.to(device)
    
    # Train Phase B
    test_acc, test_f1, history = train_phase_b(person_classifier)
    