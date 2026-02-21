import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
from tqdm import tqdm
from collections import Counter
import torch.nn.functional as F

## Configurations 
dataset_root = "/kaggle/input/volleyball"
videos_root = f"{dataset_root}/volleyball_/videos"
batch_size = 64  
epochs = 30  
LR = 1e-3  
num_classes = 8
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Output directory
output_dir = "/kaggle/working"
print(f"Output directory: {output_dir}")
print("Note: Download files from 'Output' tab on the right after training →")

assert os.path.isdir(videos_root), "Videos path is WRONG"
print("Videos path OK:", videos_root)

## Class Categories
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

## train / val / test splits
train_data = ["1","3","6","7","10","13","15","16","18","22","23","31",
              "32","36","38","39","40","41","42","48","50","52","53","54"]
val_data = ["0","2","8","12","17","19","24","26","27","28","30","33","46","49","51"]
test_data = ["4","5","9","11","14","20","21","25","29","34","35","37","43","44","45","47"]

## Dataset class
class VolleyballImageDataset(Dataset):
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

# TRANSFORMS
train_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

val_test_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# DataLoaders
print("\nLoading datasets...")
train_ds = VolleyballImageDataset(train_data, train_transform)
val_ds = VolleyballImageDataset(val_data, val_test_transform)
test_ds = VolleyballImageDataset(test_data, val_test_transform)

train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

print(f"\nTrain samples: {len(train_ds)}")
print(f"Val samples  : {len(val_ds)}")
print(f"Test samples : {len(test_ds)}")

# MODEL 
print("\nBuilding model...")
class Group_Activity_Classifier(nn.Module):
    def __init__(self, num_classes):
        super(Group_Activity_Classifier, self).__init__()
        # Use DEFAULT weights like reference code
        self.resnet50 = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        # Simple replacement - NO dropout like reference
        self.resnet50.fc = nn.Linear(
            in_features=self.resnet50.fc.in_features, 
            out_features=num_classes
        )
    
    def forward(self, x):
        return self.resnet50(x)

model = Group_Activity_Classifier(num_classes=num_classes)
model = model.to(device)

print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

# NO CLASS WEIGHTS
print("\nClass distribution in training set:")
labels = [label for _, label in train_ds.samples]
counts = Counter(labels)
for name, idx in sorted(Categories.items(), key=lambda x: x[1]):
    print(f"  {name}: {counts[idx]} samples")

criterion = nn.CrossEntropyLoss()  # Simple, no weights, no label smoothing

# Optimizer 
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LR,
    weight_decay=1e-4
)

# Scheduler
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=0.1,
    patience=3
)
# Mixed precision training like reference
from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()

# Evaluation function
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def evaluate(model, loader):
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

# Save functions
def save_checkpoint(path, model, optimizer, epoch, val_acc):
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_acc": val_acc
    }, path)

# Training Loop
print("\n" + "="*60)
print("STARTING TRAINING - Reference-Based Approach")
print("="*60)

best_val_acc = 0.0
best_model_state = None

history = {
    'train_loss': [],
    'train_acc': [],
    'val_loss': [],
    'val_acc': [],
    'val_f1': [],
    'lr': []
}

for epoch in range(epochs):
    model.train()
    running_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (x, y) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")):
        x, y = x.to(device), y.to(device)
        
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        running_loss += loss.item()
        pred = out.argmax(1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    
    # Calculate metrics
    train_loss = running_loss / len(train_loader)
    train_acc = correct / total
    val_acc, val_f1, val_loss, _, _ = evaluate(model, val_loader)
    
    # Step scheduler based on validation loss
    scheduler.step(val_loss)
    current_lr = optimizer.param_groups[0]['lr']
    
    # Store metrics
    history['train_loss'].append(train_loss)
    history['train_acc'].append(train_acc)
    history['val_loss'].append(val_loss)
    history['val_acc'].append(val_acc)
    history['val_f1'].append(val_f1)
    history['lr'].append(current_lr)
    
    print(
        f"Epoch {epoch+1}/{epochs}: "
        f"Train Loss={train_loss:.4f} | Train Acc={train_acc:.4f} | "
        f"Val Loss={val_loss:.4f} | Val Acc={val_acc:.4f} | Val F1={val_f1:.4f} | "
        f"LR={current_lr:.6f}"
    )
    
    # Save best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_model_state = model.state_dict()
        print(f"  New best validation accuracy: {best_val_acc:.4f}")
        
        save_checkpoint(
            path=f"{output_dir}/best_model_checkpoint.pkl",
            model=model,
            optimizer=optimizer,
            epoch=epoch + 1,
            val_acc=best_val_acc
        )
        torch.save(model, f"{output_dir}/best_model_full.pth")

print("\n" + "="*60)
print("TRAINING COMPLETED")
print("="*60)

# Plot Training History
print("\nGenerating training visualizations...")

fig, axes = plt.subplots(1, 3, figsize=(20, 5))

# Plot 1: Loss curves
axes[0].plot(range(1, len(history['train_loss'])+1), history['train_loss'], 'b-o', label='Train Loss', linewidth=2, markersize=4)
axes[0].plot(range(1, len(history['val_loss'])+1), history['val_loss'], 'r-s', label='Val Loss', linewidth=2, markersize=4)
axes[0].set_xlabel('Epoch', fontsize=12)
axes[0].set_ylabel('Loss', fontsize=12)
axes[0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
axes[0].legend(fontsize=11)
axes[0].grid(True, alpha=0.3)

# Plot 2: Accuracy curves
axes[1].plot(range(1, len(history['train_acc'])+1), history['train_acc'], 'b-o', label='Train Accuracy', linewidth=2, markersize=4)
axes[1].plot(range(1, len(history['val_acc'])+1), history['val_acc'], 'r-s', label='Val Accuracy', linewidth=2, markersize=4)
axes[1].set_xlabel('Epoch', fontsize=12)
axes[1].set_ylabel('Accuracy', fontsize=12)
axes[1].set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
axes[1].legend(fontsize=11)
axes[1].grid(True, alpha=0.3)

# Plot 3: Learning Rate
axes[2].plot(range(1, len(history['lr'])+1), history['lr'], 'g-o', linewidth=2, markersize=4)
axes[2].set_xlabel('Epoch', fontsize=12)
axes[2].set_ylabel('Learning Rate', fontsize=12)
axes[2].set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
axes[2].set_yscale('log')
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f"{output_dir}/training_history.png", dpi=150, bbox_inches='tight')
plt.show()
print(f"  → Training history plot saved")

# Final Test Evaluation
print("\n" + "="*60)
print("FINAL EVALUATION ON TEST SET")
print("="*60)

model.load_state_dict(best_model_state)
print(f"Loaded best model (Val Acc = {best_val_acc:.4f})\n")

test_acc, test_f1, test_loss, test_labels, test_preds = evaluate(model, test_loader)

print("="*60)
print("TEST RESULTS:")
print("="*60)
print(f"Accuracy     : {test_acc*100:.2f}%")
print(f"Average Loss : {test_loss:.4f}")
print(f"F1 Score     : {test_f1:.4f}")
print()

# Classification Report
print("Classification Report:")
report = classification_report(
    test_labels, 
    test_preds, 
    target_names=list(Categories.keys()),
    digits=4
)
print(report)

# Per-class accuracy
print("\nPer-Class Accuracy:")
for name, idx in sorted(Categories.items(), key=lambda x: x[1]):
    class_mask = [l == idx for l in test_labels]
    if sum(class_mask) > 0:
        class_preds = [test_preds[i] for i, m in enumerate(class_mask) if m]
        class_labels = [test_labels[i] for i, m in enumerate(class_mask) if m]
        class_acc = accuracy_score(class_labels, class_preds)
        print(f"  {name:12s}: {class_acc:.4f} ({sum(class_mask)} samples)")

# Confusion Matrix
cm = confusion_matrix(test_labels, test_preds)
cm_normalized = cm.astype("float") / cm.sum(axis=1, keepdims=True)

plt.figure(figsize=(10, 8))
sns.heatmap(
    cm_normalized,
    annot=True,
    fmt=".2f",
    cmap="Blues",
    xticklabels=list(Categories.keys()),
    yticklabels=list(Categories.keys()),
    cbar_kws={'label': 'Normalized Frequency'}
)
plt.xlabel("Predicted Label", fontsize=12)
plt.ylabel("True Label", fontsize=12)
plt.title("Normalized Confusion Matrix - Test Set", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f"{output_dir}/confusion_matrix.png", dpi=150, bbox_inches='tight')
plt.show()
print(f"\n  → Confusion matrix saved")

# Save training log
with open(f"{output_dir}/training_log.txt", "w") as f:

    f.write(f"  - Epochs: {epochs}\n")
    f.write(f"  - Batch Size: {batch_size}\n")
    f.write(f"  - Learning Rate: {LR}\n")
    f.write(f"  - Train samples: {len(train_ds)}\n")
    f.write(f"  - Val samples: {len(val_ds)}\n")
    f.write(f"  - Test samples: {len(test_ds)}\n\n")
    
    f.write("FINAL RESULTS\n")
    f.write("="*60 + "\n")
    f.write(f"Best Validation Accuracy: {best_val_acc:.4f}\n")
    f.write(f"Test Loss:                {test_loss:.4f}\n")
    f.write(f"Test Accuracy:            {test_acc:.4f}\n")
    f.write(f"Test F1-score:            {test_f1:.4f}\n\n")
    f.write(report)

# Summary
print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"Best Validation Accuracy: {best_val_acc:.4f}")
print(f"Final Test Accuracy     : {test_acc:.4f} ({test_acc*100:.2f}%)")
print(f"Final Test F1-score     : {test_f1:.4f}")
