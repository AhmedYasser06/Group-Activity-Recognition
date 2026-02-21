"""
VOLLEYBALL GROUP ACTIVITY RECOGNITION - B1 + B4 SEQUENTIAL TRAINING
====================================================================

This notebook trains two baselines sequentially:
1. Baseline B1: ResNet50 single-frame classifier
2. Baseline B4: LSTM with 9-frame temporal sequences (uses B1 features)

Execution: B1 → Save Checkpoint → B4 (automatic)
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

# ============================================================================
# GLOBAL CONFIGURATIONS
# ============================================================================

dataset_root = "/kaggle/input/volleyball"
videos_root = f"{dataset_root}/volleyball_/videos"
output_dir = "/kaggle/working"

# Training settings
B1_BATCH_SIZE = 64
B1_EPOCHS = 30
B1_LR = 1e-3

B4_BATCH_SIZE = 32
B4_EPOCHS = 30
B4_LR = 1e-3

NUM_CLASSES = 8
SEQUENCE_LENGTH = 9  # 4 before + middle + 4 after

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("="*70)
print("VOLLEYBALL GROUP ACTIVITY RECOGNITION - SEQUENTIAL TRAINING")
print("="*70)
print(f"Output directory: {output_dir}")
print(f"Device: {device}")
print(f"Phase 1: Train B1 (ResNet50 single-frame)")
print(f"Phase 2: Train B4 (LSTM with B1 features)")
print("="*70)

assert os.path.isdir(videos_root), "Videos path is WRONG"
print("✓ Videos path OK:", videos_root)

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

train_data = ["1","3","6","7","10","13","15","16","18","22","23","31",
              "32","36","38","39","40","41","42","48","50","52","53","54"]
val_data = ["0","2","8","12","17","19","24","26","27","28","30","33","46","49","51"]
test_data = ["4","5","9","11","14","20","21","25","29","34","35","37","43","44","45","47"]

# ============================================================================
# BASELINE 1 (B1) - SINGLE FRAME DATASET
# ============================================================================

class VolleyballImageDataset(Dataset):
    """B1 Dataset: Single frame per sample"""
    
    def __init__(self, video_ids, transform):
        self.samples = []
        self.transform = transform
        self.skipped = 0
        
        for vid in video_ids:
            video_dir = os.path.join(videos_root, vid)
            annot_file = os.path.join(video_dir, "annotations.txt")
            
            if not os.path.isfile(annot_file):
                print(f"Warning: annotations.txt not found in video {vid}")
                continue
            
            with open(annot_file, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 2:
                        continue
                    
                    frame_name = parts[0]
                    label_str = parts[1]
                    
                    if label_str not in Categories:
                        print(f"Warning: unknown label '{label_str}' in video {vid}")
                        continue
                    
                    label = Categories[label_str]
                    clip_id = frame_name.replace(".jpg", "")
                    img_path = os.path.join(video_dir, clip_id, frame_name)
                    
                    if os.path.isfile(img_path):
                        self.samples.append((img_path, label))
                    else:
                        self.skipped += 1
        
        if self.skipped > 0:
            print(f"Skipped {self.skipped} missing image files")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        try:
            image = Image.open(img_path).convert("RGB")
            image = self.transform(image)
            return image, label
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            return torch.zeros(3, 224, 224), label

# ============================================================================
# BASELINE 4 (B4) - SEQUENCE DATASET
# ============================================================================

class VolleyballSequenceDataset(Dataset):
    """B4 Dataset: 9-frame sequences per sample"""
    
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
                    label_str = parts[1]
                    
                    if label_str not in Categories:
                        continue
                    
                    label = Categories[label_str]
                    clip_id = frame_name.replace(".jpg", "")
                    
                    try:
                        frame_num = int(clip_id)
                    except:
                        self.skipped += 1
                        continue
                    
                    # 9 frames: 4 before + middle + 4 after
                    frame_numbers = [frame_num - 4 + i for i in range(9)]
                    frame_paths = []
                    clip_dir = os.path.join(video_dir, clip_id)
                    
                    all_exist = True
                    for fn in frame_numbers:
                        frame_path = os.path.join(clip_dir, f"{fn}.jpg")
                        if not os.path.isfile(frame_path):
                            all_exist = False
                            break
                        frame_paths.append(frame_path)
                    
                    if all_exist:
                        self.samples.append((frame_paths, label))
                    else:
                        self.skipped += 1
        
        print(f"Loaded {len(self.samples)} sequences | Skipped {self.skipped}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        frame_paths, label = self.samples[idx]
        frames = []
        
        for path in frame_paths:
            try:
                img = Image.open(path).convert("RGB")
                if self.transform:
                    img = self.transform(img)
                frames.append(img)
            except Exception as e:
                print(f"Error loading {path}: {e}")
                frames.append(torch.zeros(3, 224, 224))
        
        frames = torch.stack(frames)  # [9, 3, 224, 224]
        return frames, label

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
# B1 MODEL
# ============================================================================

class B1_Classifier(nn.Module):
    """Baseline 1: ResNet50 single-frame classifier"""
    
    def __init__(self, num_classes):
        super(B1_Classifier, self).__init__()
        self.resnet50 = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.resnet50.fc = nn.Linear(
            in_features=self.resnet50.fc.in_features, 
            out_features=num_classes
        )
    
    def forward(self, x):
        return self.resnet50(x)

# ============================================================================
# B4 MODELS
# ============================================================================

class B1_FeatureExtractor(nn.Module):
    """Load B1's trained ResNet50 as frozen feature extractor"""
    
    def __init__(self, checkpoint_path):
        super(B1_FeatureExtractor, self).__init__()
        
        print(f"\n  Loading B1 checkpoint from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Create B1 model with same architecture
        b1_model = B1_Classifier(NUM_CLASSES)
        
        # Load B1's weights
        b1_model.load_state_dict(checkpoint['model_state_dict'])
        print(f"  ✓ Loaded B1 from epoch {checkpoint['epoch']} with val_acc={checkpoint['val_acc']:.4f}")
        
        # Extract feature extractor from B1's resnet50 (remove final FC)
        self.feature_extractor = nn.Sequential(*list(b1_model.resnet50.children())[:-1])
        
        # Freeze all parameters
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
        
        self.feature_extractor.eval()
    
    def forward(self, x):
        """Input: [batch*9, 3, 224, 224] → Output: [batch*9, 2048]"""
        with torch.no_grad():
            features = self.feature_extractor(x)
            features = features.squeeze(-1).squeeze(-1)
        return features


class LSTM_Classifier(nn.Module):
    """LSTM classifier for sequence of features"""
    
    def __init__(self, input_size=2048, hidden_size=512, num_classes=8):
        super(LSTM_Classifier, self).__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True
        )
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        """Input: [batch, 9, 2048] → Output: [batch, 8]"""
        lstm_out, (h_n, c_n) = self.lstm(x)
        last_hidden = h_n[-1]
        out = self.fc(last_hidden)
        return out


class B4_Complete_Model(nn.Module):
    """Complete B4: B1 feature extractor + LSTM classifier"""
    
    def __init__(self, b1_checkpoint_path):
        super(B4_Complete_Model, self).__init__()
        self.feature_extractor = B1_FeatureExtractor(b1_checkpoint_path)
        self.lstm_classifier = LSTM_Classifier()
    
    def forward(self, x):
        """Input: [batch, 9, 3, 224, 224] → Output: [batch, 8]"""
        batch_size, seq_len, c, h, w = x.shape
        
        # Extract features from all frames
        x = x.view(batch_size * seq_len, c, h, w)
        features = self.feature_extractor(x)
        
        # Reshape to sequence
        features = features.view(batch_size, seq_len, -1)
        
        # LSTM classification
        out = self.lstm_classifier(features)
        return out

# ============================================================================
# TRAINING UTILITIES
# ============================================================================

def evaluate(model, loader, criterion):
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
    fig, axes = plt.subplots(1, 3, figsize=(20, 5))
    
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
    
    # Learning rate
    axes[2].plot(history['lr'], 'g-o', linewidth=2, markersize=4)
    axes[2].set_xlabel('Epoch', fontsize=12)
    axes[2].set_ylabel('Learning Rate', fontsize=12)
    axes[2].set_title(f'{title_prefix} - LR Schedule', fontsize=14, fontweight='bold')
    axes[2].set_yscale('log')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_confusion_matrix(labels, preds, title, save_path):
    """Plot confusion matrix"""
    cm = confusion_matrix(labels, preds)
    cm_norm = cm.astype("float") / cm.sum(axis=1, keepdims=True)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=list(Categories.keys()),
        yticklabels=list(Categories.keys()),
        cbar_kws={'label': 'Normalized Frequency'}
    )
    plt.xlabel("Predicted Label", fontsize=12)
    plt.ylabel("True Label", fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


# ============================================================================
# PHASE 1: TRAIN BASELINE B1
# ============================================================================

def train_b1():
    print("\n" + "="*70)
    print("PHASE 1: TRAINING BASELINE B1 (ResNet50 Single-Frame)")
    print("="*70)
    
    # Load datasets
    print("\nLoading B1 datasets...")
    train_ds = VolleyballImageDataset(train_data, transform)
    val_ds = VolleyballImageDataset(val_data, transform)
    test_ds = VolleyballImageDataset(test_data, transform)
    
    train_loader = DataLoader(train_ds, batch_size=B1_BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=B1_BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=B1_BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
    
    print(f"Train samples: {len(train_ds)}")
    print(f"Val samples:   {len(val_ds)}")
    print(f"Test samples:  {len(test_ds)}")
    
    # Class distribution
    print("\nClass distribution in training set:")
    labels = [label for _, label in train_ds.samples]
    counts = Counter(labels)
    for name, idx in sorted(Categories.items(), key=lambda x: x[1]):
        print(f"  {name:12s}: {counts[idx]} samples")
    
    # Build model
    print("\nBuilding B1 model...")
    model = B1_Classifier(num_classes=NUM_CLASSES).to(device)
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=B1_LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)
    scaler = GradScaler()
    
    # Training loop
    print("\n" + "="*70)
    print("STARTING B1 TRAINING")
    print("="*70)
    
    best_val_acc = 0.0
    best_model_state = None
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [], 'val_f1': [], 'lr': []
    }
    
    for epoch in range(B1_EPOCHS):
        model.train()
        running_loss = 0
        correct = 0
        total = 0
        
        for x, y in tqdm(train_loader, desc=f"B1 Epoch {epoch+1}/{B1_EPOCHS}"):
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
        val_acc, val_f1, val_loss, _, _ = evaluate(model, val_loader, criterion)
        
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_f1'].append(val_f1)
        history['lr'].append(current_lr)
        
        print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f} | Train Acc={train_acc:.4f} | "
              f"Val Loss={val_loss:.4f} | Val Acc={val_acc:.4f} | Val F1={val_f1:.4f} | LR={current_lr:.6f}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict()
            print(f"  ✓ New best: {best_val_acc:.4f}")
            save_checkpoint(f"{output_dir}/b1_best_checkpoint.pkl", model, optimizer, epoch+1, best_val_acc)
    
    # Test evaluation
    print("\n" + "="*70)
    print("B1 TEST EVALUATION")
    print("="*70)
    
    model.load_state_dict(best_model_state)
    test_acc, test_f1, test_loss, test_labels, test_preds = evaluate(model, test_loader, criterion)
    
    print(f"\nB1 Test Results:")
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
    print("\nGenerating B1 visualizations...")
    plot_training_curves(history, "B1", f"{output_dir}/b1_training_history.png")
    plot_confusion_matrix(test_labels, test_preds, "B1 - Confusion Matrix", f"{output_dir}/b1_confusion_matrix.png")
    
    # Save log
    with open(f"{output_dir}/b1_log.txt", "w") as f:
        f.write("BASELINE B1 - ResNet50 Single-Frame Classifier\n")
        f.write("="*70 + "\n\n")
        f.write(f"Best Val Acc: {best_val_acc:.4f}\n")
        f.write(f"Test Acc:     {test_acc:.4f}\n")
        f.write(f"Test F1:      {test_f1:.4f}\n\n")
        f.write(report)
    
    print("\n✓ B1 Training completed!")
    print(f"✓ Checkpoint saved: b1_best_checkpoint.pkl")
    
    return best_val_acc, test_acc, test_f1


# ============================================================================
# PHASE 2: TRAIN BASELINE B4
# ============================================================================

def train_b4(b1_checkpoint_path):
    print("\n" + "="*70)
    print("PHASE 2: TRAINING BASELINE B4 (LSTM with Temporal Sequences)")
    print("="*70)
    print(f"Using B1 checkpoint: {b1_checkpoint_path}")
    print(f"Sequence length: {SEQUENCE_LENGTH} frames (4 before + mid + 4 after)")
    
    # Check checkpoint exists
    if not os.path.isfile(b1_checkpoint_path):
        print(f"\n❌ ERROR: B1 checkpoint not found at {b1_checkpoint_path}")
        return None, None, None
    
    # Load datasets
    print("\nLoading B4 sequence datasets...")
    train_ds = VolleyballSequenceDataset(train_data, transform)
    val_ds = VolleyballSequenceDataset(val_data, transform)
    test_ds = VolleyballSequenceDataset(test_data, transform)
    
    train_loader = DataLoader(train_ds, batch_size=B4_BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=B4_BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=B4_BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
    
    print(f"Train: {len(train_ds)} sequences")
    print(f"Val:   {len(val_ds)} sequences")
    print(f"Test:  {len(test_ds)} sequences")
    
    # Class distribution
    print("\nClass distribution in training set:")
    labels = [label for _, label in train_ds.samples]
    counts = Counter(labels)
    for name, idx in sorted(Categories.items(), key=lambda x: x[1]):
        print(f"  {name:12s}: {counts[idx]} samples")
    
    # Build model
    print("\nBuilding B4 model...")
    model = B4_Complete_Model(b1_checkpoint_path).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters:     {total_params:,}")
    print(f"Trainable (LSTM):     {trainable_params:,}")
    print(f"Frozen (B1 features): {total_params - trainable_params:,}")
    
    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=B4_LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)
    scaler = GradScaler()
    
    # Training loop
    print("\n" + "="*70)
    print("STARTING B4 TRAINING")
    print("="*70)
    
    best_val_acc = 0.0
    best_model_state = None
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [], 'val_f1': [], 'lr': []
    }
    
    for epoch in range(B4_EPOCHS):
        model.train()
        running_loss = 0
        correct = 0
        total = 0
        
        for x, y in tqdm(train_loader, desc=f"B4 Epoch {epoch+1}/{B4_EPOCHS}"):
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
        val_acc, val_f1, val_loss, _, _ = evaluate(model, val_loader, criterion)
        
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_f1'].append(val_f1)
        history['lr'].append(current_lr)
        
        print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f} | Train Acc={train_acc:.4f} | "
              f"Val Loss={val_loss:.4f} | Val Acc={val_acc:.4f} | Val F1={val_f1:.4f} | LR={current_lr:.6f}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict()
            print(f"  ✓ New best: {best_val_acc:.4f}")
            save_checkpoint(f"{output_dir}/b4_best_checkpoint.pkl", model, optimizer, epoch+1, best_val_acc)
    
    # Test evaluation
    print("\n" + "="*70)
    print("B4 TEST EVALUATION")
    print("="*70)
    
    model.load_state_dict(best_model_state)
    test_acc, test_f1, test_loss, test_labels, test_preds = evaluate(model, test_loader, criterion)
    
    print(f"\nB4 Test Results:")
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
    print("\nGenerating B4 visualizations...")
    plot_training_curves(history, "B4", f"{output_dir}/b4_training_history.png")
    plot_confusion_matrix(test_labels, test_preds, "B4 LSTM - Confusion Matrix", f"{output_dir}/b4_confusion_matrix.png")
    
    # Save log
    with open(f"{output_dir}/b4_log.txt", "w") as f:
        f.write("BASELINE B4 - LSTM Temporal Classifier\n")
        f.write("="*70 + "\n\n")
        f.write(f"Sequence length: {SEQUENCE_LENGTH} frames\n")
        f.write(f"Feature extractor: Frozen B1 ResNet50\n\n")
        f.write(f"Best Val Acc: {best_val_acc:.4f}\n")
        f.write(f"Test Acc:     {test_acc:.4f}\n")
        f.write(f"Test F1:      {test_f1:.4f}\n\n")
        f.write(report)
    
    print("\n✓ B4 Training completed!")
    print(f"✓ Checkpoint saved: b4_best_checkpoint.pkl")
    
    return best_val_acc, test_acc, test_f1


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("STARTING SEQUENTIAL TRAINING: B1 → B4")
    print("="*70)
    
    # Phase 1: Train B1
    b1_val_acc, b1_test_acc, b1_test_f1 = train_b1()
    
    # Phase 2: Train B4 using B1's checkpoint
    b1_checkpoint = f"{output_dir}/b1_best_checkpoint.pkl"
    b4_val_acc, b4_test_acc, b4_test_f1 = train_b4(b1_checkpoint)
    
    # Final Summary
    print("\n" + "="*70)
    print("FINAL SUMMARY - ALL BASELINES")
    print("="*70)
    print("\nBaseline B1 (ResNet50 Single-Frame):")
    print(f"  Best Val Acc: {b1_val_acc:.4f}")
    print(f"  Test Acc:     {b1_test_acc:.4f} ({b1_test_acc*100:.2f}%)")
    print(f"  Test F1:      {b1_test_f1:.4f}")
    
    if b4_test_acc is not None:
        print("\nBaseline B4 (LSTM Temporal Sequences):")
        print(f"  Best Val Acc: {b4_val_acc:.4f}")
        print(f"  Test Acc:     {b4_test_acc:.4f} ({b4_test_acc*100:.2f}%)")
        print(f"  Test F1:      {b4_test_f1:.4f}")
        
        print("\n" + "-"*70)
        print("COMPARISON:")
        print("-"*70)
        diff = b4_test_acc - b1_test_acc
        if diff > 0:
            print(f"✓ B4 beats B1 by {diff*100:.2f}% (absolute)")
        else:
            print(f"⚠ B4 is {abs(diff)*100:.2f}% below B1")
    
    print("\n" + "="*70)
    print("FILES SAVED:")
    print("="*70)
    print("B1 Files:")
    print("  - b1_best_checkpoint.pkl")
    print("  - b1_training_history.png")
    print("  - b1_confusion_matrix.png")
    print("  - b1_log.txt")
    print("\nB4 Files:")
    print("  - b4_best_checkpoint.pkl")
    print("  - b4_training_history.png")
    print("  - b4_confusion_matrix.png")
    print("  - b4_log.txt")
    print("="*70)
    print("\n✓ ALL TRAINING COMPLETED SUCCESSFULLY!")
    print("Note: Download files from 'Output' tab on the right →")