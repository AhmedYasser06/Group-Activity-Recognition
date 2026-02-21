"""
Baseline B3 - Two-Phase Training Approach (OPTIMIZED)
======================================================
Phase A: Train person-level action classifier (9 classes) - 3-5 epochs with early stopping
Phase B: Train group activity classifier using pooled person features (8 classes)

KEY OPTIMIZATIONS:
1. Phase A: 5 epochs max, use BEST val checkpoint
2. NO augmentation (match B1's successful approach)
3. Use ALL frames (not just ones with 12 perfect crops)
4. Unfreeze last ResNet block in Phase B
5. One-hot encoding for proper training
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
# CONFIGURATIONS
# ============================================================================

dataset_root = "/kaggle/input/volleyball"
videos_root = f"{dataset_root}/volleyball_/videos"
output_dir = "/kaggle/working"

# Phase A settings - Train for 5 epochs, pick best
PHASE_A_EPOCHS = 5
PHASE_A_BATCH_SIZE = 64
PHASE_A_LR = 1e-3

# Phase B settings  
PHASE_B_EPOCHS = 30
PHASE_B_BATCH_SIZE = 32
PHASE_B_LR = 1e-3

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
# DATASET CLASSES
# ============================================================================

class PersonActionDataset(Dataset):
    """Phase A: Individual player crops with action labels"""
    
    def __init__(self, video_ids, transform, use_onehot=False):
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
        
        if self.transform:
            image = self.transform(image)
        
        if self.use_onehot:
            label_onehot = torch.zeros(9, dtype=torch.float32)
            label_onehot[label] = 1.0
            return image, label_onehot
        
        return image, label


class GroupActivityDataset(Dataset):
    """Phase B: All player crops per frame with group activity label"""
    
    def __init__(self, video_ids, transform, use_onehot=False):
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
                    
                    group_label = Categories[group_label_str]
                    clip_id = frame_name.replace(".jpg", "")
                    img_path = os.path.join(video_dir, clip_id, frame_name)
                    
                    if not os.path.isfile(img_path):
                        self.skipped += 1
                        continue
                    
                    # Collect ALL player bboxes (pad if needed)
                    bboxes = []
                    num_players = 12
                    start = 2
                    
                    for idx in range(num_players):
                        base = start + idx * 5
                        try:
                            x = int(parts[base])
                            y = int(parts[base + 1])
                            w = int(parts[base + 2])
                            h = int(parts[base + 3])
                            bboxes.append((x, y, w, h))
                        except:
                            # If missing, use a default small crop (will be padded)
                            bboxes.append((0, 0, 50, 50))
                    
                    # Always add the frame (even if some crops are invalid)
                    self.samples.append((img_path, bboxes, group_label, len([b for b in bboxes if b != (0, 0, 50, 50)])))
        
        print(f"Loaded {len(self.samples)} group samples | Skipped {self.skipped}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, bboxes, label, valid_count = self.samples[idx]
        
        image = Image.open(img_path).convert("RGB")
        img_w, img_h = image.size
        
        player_crops = []
        for (x, y, w, h) in bboxes:
            # Handle invalid/padding boxes
            if (x, y, w, h) == (0, 0, 50, 50):
                # Create black padding image
                crop = Image.new('RGB', (224, 224), (0, 0, 0))
            else:
                # Clip to image boundaries
                x = max(0, min(x, img_w-1))
                y = max(0, min(y, img_h-1))
                w = max(1, min(w, img_w-x))
                h = max(1, min(h, img_h-y))
                crop = image.crop((x, y, x + w, y + h))
            
            if self.transform:
                crop = self.transform(crop)
            player_crops.append(crop)
        
        player_crops = torch.stack(player_crops)
        
        if self.use_onehot:
            label_onehot = torch.zeros(8, dtype=torch.float32)
            label_onehot[label] = 1.0
            return player_crops, label_onehot
        
        return player_crops, label


def collate_fn(batch):
    """Collate function"""
    clips, labels = zip(*batch)
    clips = torch.stack(clips)
    labels = torch.stack(labels)
    return clips, labels


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


class Group_Activity_Classifier(nn.Module):
    """Phase B: Group activity classifier with PARTIAL unfreezing"""
    
    def __init__(self, person_feature_extractor, num_classes=8):
        super(Group_Activity_Classifier, self).__init__()
        
        # Extract backbone (without FC)
        self.fc_in_features = person_feature_extractor.in_features
        backbone_layers = list(person_feature_extractor.resnet50.children())[:-1]
        
        # Split into frozen and unfrozen parts
        # Freeze layers 0-7 (conv1, bn1, relu, maxpool, layer1, layer2, layer3, layer4[:2])
        # Unfreeze layer4[-1] (last residual block) + avgpool
        self.frozen_features = nn.Sequential(*backbone_layers[:-2])  # Up to layer4
        self.unfrozen_features = nn.Sequential(*backbone_layers[-2:])  # layer4 + avgpool
        
        # Freeze the frozen part
        for param in self.frozen_features.parameters():
            param.requires_grad = False
        
        # Unfreeze the unfrozen part
        for param in self.unfrozen_features.parameters():
            param.requires_grad = True
        
        # Max pooling across players
        self.pool = nn.AdaptiveMaxPool2d((1, 2048))
        
        # Larger classifier head for more capacity
        self.fc = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes),
        )
    
    def forward(self, x):
        b, num_players, c, h, w = x.shape
        
        x = x.view(b * num_players, c, h, w)
        
        # Pass through frozen then unfrozen parts
        with torch.no_grad():
            x = self.frozen_features(x)
        x = self.unfrozen_features(x)
        
        x = x.view(b, num_players, -1)
        x = self.pool(x)
        x = x.squeeze(dim=1)
        x = self.fc(x)
        return x


# ============================================================================
# TRANSFORMS - NO AUGMENTATION (match B1)
# ============================================================================

simple_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
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
            out = model(x)
            loss = criterion(out, y)
            total_loss += loss.item()
            
            p = out.argmax(1)
            y_class = y.argmax(1)
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


# ============================================================================
# PHASE A: TRAIN PERSON ACTION CLASSIFIER
# ============================================================================

def train_phase_a():
    print("\n" + "="*70)
    print("PHASE A: PERSON-LEVEL ACTION CLASSIFIER (9 CLASSES)")
    print("="*70)
    print("Strategy: Train 5 epochs, use BEST validation checkpoint")
    print("="*70)
    
    train_ds = PersonActionDataset(train_data, simple_transform, use_onehot=True)
    val_ds = PersonActionDataset(val_data, simple_transform, use_onehot=True)
    
    train_loader = DataLoader(train_ds, batch_size=PHASE_A_BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=PHASE_A_BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
    
    print(f"Train: {len(train_ds)} samples | Val: {len(val_ds)} samples")
    
    model = Person_Activity_Classifier(num_classes=9).to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=PHASE_A_LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=2)
    scaler = GradScaler()
    
    best_val_acc = 0.0
    best_model_state = None
    best_epoch = 0
    
    print("\nTraining Phase A...")
    for epoch in range(PHASE_A_EPOCHS):
        model.train()
        running_loss = 0
        correct = 0
        total = 0
        
        for x, y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{PHASE_A_EPOCHS}"):
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
            y_class = y.argmax(1)
            correct += (pred == y_class).sum().item()
            total += y.size(0)
        
        train_loss = running_loss / len(train_loader)
        train_acc = correct / total
        val_acc, val_f1, val_loss, _, _ = evaluate_model(model, val_loader, criterion)
        
        scheduler.step(val_acc)
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"Epoch {epoch+1}: Train Acc={train_acc:.4f} | Val Acc={val_acc:.4f} | Val F1={val_f1:.4f} | LR={current_lr:.6f}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict()
            best_epoch = epoch + 1
            print(f"  ✓ Best: {best_val_acc:.4f}")
            save_checkpoint(f"{output_dir}/phase_a_best.pkl", model, optimizer, epoch+1, best_val_acc)
    
    model.load_state_dict(best_model_state)
    print(f"\nPhase A Done: Best Val Acc={best_val_acc:.4f} (Epoch {best_epoch})")
    return model


# ============================================================================
# PHASE B: TRAIN GROUP ACTIVITY CLASSIFIER
# ============================================================================

def train_phase_b(person_classifier):
    print("\n" + "="*70)
    print("PHASE B: GROUP ACTIVITY CLASSIFIER (8 CLASSES)")
    print("="*70)
    
    train_ds = GroupActivityDataset(train_data, simple_transform, use_onehot=True)
    val_ds = GroupActivityDataset(val_data, simple_transform, use_onehot=True)
    test_ds = GroupActivityDataset(test_data, simple_transform, use_onehot=True)
    
    train_loader = DataLoader(train_ds, batch_size=PHASE_B_BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=PHASE_B_BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=PHASE_B_BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True, collate_fn=collate_fn)
    
    print(f"Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")
    
    model = Group_Activity_Classifier(person_classifier, num_classes=8).to(device)
    
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Total: {total:,} | Trainable: {trainable:,} | Frozen: {total-trainable:,}")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=PHASE_B_LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)
    scaler = GradScaler()
    
    best_val_acc = 0.0
    best_model_state = None
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'val_f1': []}
    
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
            y_class = y.argmax(1)
            correct += (pred == y_class).sum().item()
            total += y.size(0)
        
        train_loss = running_loss / len(train_loader)
        train_acc = correct / total
        val_acc, val_f1, val_loss, _, _ = evaluate_model(model, val_loader, criterion)
        
        scheduler.step(val_loss)
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_f1'].append(val_f1)
        
        print(f"Epoch {epoch+1}: Train Acc={train_acc:.4f} | Val Acc={val_acc:.4f} | Val F1={val_f1:.4f}")
        
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
    plt.title("Baseline B3 - Confusion Matrix")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/b3_confusion_matrix.png", dpi=150)
    plt.show()
    
    # Training curves
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    axes[0].plot(history['train_loss'], 'b-', label='Train')
    axes[0].plot(history['val_loss'], 'r-', label='Val')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Loss Curves')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(history['train_acc'], 'b-', label='Train')
    axes[1].plot(history['val_acc'], 'r-', label='Val')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Accuracy Curves')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/b3_training_history.png", dpi=150)
    plt.show()
    
    # Save log
    with open(f"{output_dir}/b3_log.txt", "w") as f:
        f.write(f"Baseline B3 - Optimized\n")
        f.write(f"="*70 + "\n")
        f.write(f"Phase A: 5 epochs, best val checkpoint\n")
        f.write(f"Phase B: Unfrozen last block, larger classifier\n")
        f.write(f"NO augmentation (match B1)\n\n")
        f.write(f"Best Val Acc: {best_val_acc:.4f}\n")
        f.write(f"Test Acc:     {test_acc:.4f}\n")
        f.write(f"Test F1:      {test_f1:.4f}\n\n")
        f.write(report)
    
    return test_acc, test_f1


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    person_classifier = train_phase_a()
    test_acc, test_f1 = train_phase_b(person_classifier)
    
    print("\n" + "="*70)
    print("FINAL RESULTS")
    print("="*70)
    print(f"Test Accuracy: {test_acc*100:.2f}%")
    print(f"Test F1-Score: {test_f1:.4f}")
    print(f"\nBaseline 1:    79.88%")
    print(f"Baseline 3:    {test_acc*100:.2f}%")
    