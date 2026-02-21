<div align="center">
  <img src="https://github.com/user-attachments/assets/22cc8c54-f3c7-4900-a9db-3e37fffac5ad" alt="Background Image" width="95%" />
</div>

<h1 align="center">Group Activity Recognition</h1>

<p align="center">
  A comprehensive PyTorch implementation inspired by the <strong>CVPR 2016 publication</strong>, <a href="https://arxiv.org/pdf/1607.02643"><em>A Hierarchical Deep Temporal Model for Group Activity Recognition</em></a>.  
  This framework utilizes a hierarchical LSTM design to classify team activities through the analysis of individual player dynamics and collective team behavior.
</p>

## Table of Contents
1. [Key Updates](#key-updates)
2. [Usage](#usage)
   - [Clone the Repository](#1-clone-the-repository)
   - [Install Dependencies](#2-install-the-required-dependencies)
   - [Download Model Checkpoint](#3-download-the-model-checkpoint)
3. [Dataset Overview](#dataset-overview)
   - [Example Annotations](#example-annotations)
   - [Train-Test Split](#train-test-split)
   - [Dataset Statistics](#dataset-statistics)
   - [Dataset Organization](#dataset-organization)
   - [Dataset Download Instructions](#dataset-download-instructions)
4. [Ablation Study](#ablation-study)
   - [Baselines](#baselines)
5. [Performance Comparison](#performance-comparison)
   - [Original Paper Baselines Score](#original-paper-baselines-score)
   - [My Scores (Accuracy and F1 Scores)](#my-scores-accuracy-and-f1-scores)
6. [Interesting Observations](#interesting-observations)
   - [Effect of Team Independent Pooling](#effect-of-team-independent-pooling)
7. [Model Architecture](#model-architecture-baseline-8)

## Key Updates

- Integration of ResNet50 as the primary feature extractor (superseding AlexNet).
- Comprehensive ablation experiments to evaluate architectural components.
- Development of a unified end-to-end training approach (Baseline 9).
- Superior classification performance demonstrated across all baseline configurations relative to the source publication.
- Complete PyTorch-based implementation (replacing the original Caffe framework).

-----
## Usage

---

### 1. Clone the Repository
```bash
git clone https://github.com/AhmedYasser06/Group-Activity-Recognition.git
```

### 2. Install the Required Dependencies
```bash
pip install -r requirements.txt
```

### 3. Download the Model Checkpoint
The pretrained model weights must be obtained separately through one of the following methods.

#### Option 1: Use Python Code
Automatically download and extract checkpoint files:
```python
import kagglehub

# Retrieve the most recent model version
checkpoint_path = kagglehub.model_download("AhmedYasser06/volleyball-gar/pyTorch/v1")

print("Model checkpoint location:", checkpoint_path)
```

#### Option 2: Download Directly
Access and download checkpoints manually from Kaggle:  
[Volleyball Activity Recognition - PyTorch Models](https://www.kaggle.com/models/)

-----
## Dataset Overview

This dataset comprises annotations extracted from publicly accessible volleyball game footage on YouTube. A total of 4,830 frames spanning 55 distinct videos were manually labeled, with individual player actions classified into 9 categories and collective team activities organized into 8 categories.

### Example Annotations

![image](https://github.com/user-attachments/assets/50f906ad-c68c-4882-b9cf-9200f5a380c7)

- **Figure**: An annotated frame displaying "Left Spike" classification, with detection boxes marking each visible player to illustrate team-level activity labeling.

![image](https://github.com/user-attachments/assets/cca9447a-8b40-4330-a11d-dbc0feb230ff)

### Train-Test Split

- **Training Partition**: 3,493 annotated frames
- **Testing Partition**: 1,337 annotated frames

### Dataset Statistics

#### Group Activity Labels
| Group Activity Class | Instances |
|-----------------------|-----------|
| Right set            | 644       |
| Right spike          | 623       |
| Right pass           | 801       |
| Right winpoint       | 295       |
| Left winpoint        | 367       |
| Left pass            | 826       |
| Left spike           | 642       |
| Left set             | 633       |

#### Player Action Labels
| Action Class | Instances |
|--------------|-----------|
| Waiting      | 3,601     |
| Setting      | 1,332     |
| Digging      | 2,333     |
| Falling      | 1,241     |
| Spiking      | 1,216     |
| Blocking     | 2,458     |
| Jumping      | 341       |
| Moving       | 5,121     |
| Standing     | 38,696    |

### Dataset Organization

- **Total Videos**: 55, identified by unique indices (0–54).
- **Training Split**: 1, 3, 6, 7, 10, 13, 15, 16, 18, 22, 23, 31, 32, 36, 38, 39, 40, 41, 42, 48, 50, 52, 53, 54.
- **Validation Split**: 0, 2, 8, 12, 17, 19, 24, 26, 27, 28, 30, 33, 46, 49, 51.
- **Testing Split**: 4, 5, 9, 11, 14, 20, 21, 25, 29, 34, 35, 37, 43, 44, 45, 47.

### Dataset Download Instructions

1. Configure Kaggle API credentials following the official guide: [Kaggle API Setup](https://www.kaggle.com/docs/api).  
2. Execute the download script provided:
```bash
  chmod 600 ~/.kaggle/kaggle.json 
  chmod +x scripts/download_volleyball_dataset.sh
  ./scripts/download_volleyball_dataset.sh
```
Additional dataset documentation is available in the original paper's repository:  
[Repository Link](https://github.com/mostafa-saad/deep-activity-rec)

-----
## [Ablation Study](https://en.wikipedia.org/wiki/Ablation_(artificial_intelligence)#:~:text=In%20artificial%20intelligence%20(AI)%2C,resultant%20performance%20of%20the%20system)

### Baselines

- **B1: Image Classification:**  
   A basic ResNet-50 classifier optimized for group activity prediction from individual video frames without temporal context.

- **B3: Fine-tuned Person Classification:**  
   ResNet-50 applied independently to each player's bounding box, extracting 2048-dimensional features that undergo max-pooling aggregation across all detected individuals before softmax classification for team activity recognition.

- **B4: Temporal Model with Image Features:**  
   Incorporates temporal information through LSTM processing of full-frame feature sequences. Each video segment contains 9 consecutive frames, with the LSTM modeling temporal dependencies across the complete sequence.

- **B5: Temporal Model with Person Features:**  
   Builds upon B3 by introducing player-level temporal modeling. Individual LSTM networks process each person's crop sequence, and the resulting temporal features are aggregated for team activity classification.

- **B6: Two-stage Model without LSTM 1:**  
  Aggregated person-level features from all visible players serve as input to a frame-level LSTM network that models collective temporal dynamics.

- **B7: Two-stage Model without LSTM 2:**  
   Complete hierarchical architecture (Version 1) where player-level LSTMs extract individual temporal patterns from 9-frame sequences. Max-pooling combines all player representations, followed by frame-level LSTM processing for final classification.

- **B8: Two-stage Hierarchical Model:**  
   Enhanced hierarchical design (Version 2) featuring team-aware aggregation. Player-level LSTMs generate individual temporal features, which are then pooled separately for each team. Concatenated team representations feed into a frame-level LSTM for activity classification.

- **B9: Unified Hierarchical Model:**  
   Consolidates person-level and group-level objectives into a single end-to-end trainable network. Unlike prior two-stage approaches, this unified architecture optimizes both classification tasks simultaneously through joint gradient backpropagation. Architecture employs `ResNet34` and `GRU` units to **reduce parameter count and enhance generalization**.

---
## Performance comparison

### Original Paper Baselines Score

![{83C0D210-27DA-4A7F-8126-D9407823B766}](https://github.com/user-attachments/assets/c62ee368-8027-4e83-a5a4-687b7adebe5a)

### My Scores (Accuracy and F1 Scores)

| **Baseline** | **Accuracy** | **F1 Score** |
|--------------|--------------|--------------|
| Baseline 1   | 79.88%       | 79.81%       |
| Baseline 3   | 80.25%       | 80.24%       |
| Baseline 4   | 76.59%       | 76.67%       |
| Baseline 5   | 77.04%       | 77.07%       |
| Baseline 6   | 84.52%       | 83.99%       |
| Baseline 7   | 89.15%       | 89.14%       |
| Baseline 8   | 92.30%       | 92.29%       |
| Baseline 9   | 93.12%       | 93.11%       |

---

## Interesting Observations

### Effect of Team Independent Pooling

Analysis of confusion matrices from Baselines 5 and 6 reveals critical insights into aggregation strategies:

#### Baseline 5 Confusion Matrix
<img src="Baselines/baseline%205/outputs/Group_Activity_Baseline_5_eval_on_testset_confusion_matrix.png" alt="Baseline 5 confusion matrix" width="60%">

#### Baseline 6 Confusion Matrix
<img src="Baselines/baseline%206/outputs/Group_Activity_Baseline_6_eval_on_testset_confusion_matrix.png" alt="Baseline 6 confusion matrix" width="60%">

- Predominant classification errors occur between symmetric activity pairs:
  - Right winpoint vs. left winpoint
  - Right pass vs. left pass
  - Right set vs. left set
  - Right spike vs. left spike

This pattern stems from indiscriminate pooling across all 12 players during the transition from individual to group-level representations. Treating both teams as a single unified group discards critical spatial configuration data about player locations on the court.

Maintaining team-based separation during feature aggregation preserves geometric spatial relationships. This preservation of positional information demonstrates measurable benefits in Baselines 8 and 9, where team-aware processing significantly reduces symmetric misclassification.

#### Baseline 8 Confusion Matrix
<img src="Baselines/baseline%208/outputs/Group_Activity_Baseline_8_eval_on_testset_confusion_matrix.png" alt="Baseline 8 confusion matrix" width="60%">

#### Baseline 9 Confusion Matrix
<img src="Baselines/baseline 9 (end to end)/outputs/Group_Activity_Baseline_9_eval_on_testset_confusion_matrix.png" alt="Baseline 9 confusion matrix" width="60%">

--- 

### Model Architecture (Baseline 8)

This hierarchical temporal architecture integrates individual player representations with team-level dynamics across temporal dimensions. The following sections describe the computational pipeline and architectural components.

1. **Player-Level Feature Extraction**: ResNet-50 and LSTM networks process individual player trajectories to extract temporal behavioral patterns.
2. **Team-Level Feature Integration**: Team-specific aggregation followed by frame-level LSTM processing enables collective activity classification.

#### **1. Player Activity Temporal Classifier**
The `Person_Activity_Temporal_Classifier` module processes individual player sequences through the following stages:

- **ResNet-50 Backbone**: Pre-trained ResNet-50 network (fully-connected layer removed) generates spatial feature representations from player crop regions.
- **Layer Normalization**: Normalizes activations to stabilize training and improve convergence.
- **Temporal Modeling with LSTM**: LSTM unit captures temporal evolution of individual player features across the video sequence.
- **Fully Connected Layers**: Multi-layer perceptron maps LSTM hidden states to player action class probabilities.

#### **2. Group Activity Temporal Classifier**
The `Group_Activity_Temporal_Classifier` module extends individual modeling to team-level prediction:

- **Shared ResNet-50 and LSTM**: Inherits frozen ResNet-50 and LSTM parameters from the person-level classifier to transfer learned representations.
- **Pooling and Feature Concatenation**:
  - ResNet-50 spatial features and LSTM temporal features are concatenated for each player.
  - Player features are partitioned into two team groups (players 1–6 for Team A, players 7–12 for Team B).
  - Adaptive max-pooling reduces each team's player set to a single representative feature vector.
  - Team feature vectors are concatenated to form the complete frame representation.
- **Team-Level LSTM**: Second LSTM network processes concatenated team features temporally, modeling inter-team dynamics over time.
- **Classification Layers**: Dense layers with softmax activation produce final team activity predictions.

#### Training Configuration

- **Compute Resources**: Experiments conducted on Kaggle's complimentary GPU allocation (P100 with 16GB memory).
- **Optimization**: AdamW optimizer with cosine learning rate decay.
- **Batch Configuration:** 8 samples per iteration
                  
---

### Model Architecture: Hierarchical Group Activity Classifier (Baseline 9)

The `Hierarchical_Group_Activity_Classifier` architecture unifies spatial encoding, temporal sequence modeling, and hierarchical feature aggregation to generate predictions at both individual and collective levels simultaneously.

1. **Feature Extraction**: 
   - Pre-trained ResNet-34 backbone extracts spatial representations from individual video frame regions.
2. **Individual-Level Classification**:
   - Layer normalization stabilizes spatial features before temporal processing.
   - Gated Recurrent Unit (GRU) models temporal dependencies for each detected person across the frame sequence.
   - Multi-layer fully-connected network with normalization, non-linear activations, and dropout regularization classifies individual activities.

3. **Group-Level Classification**:
   - Adaptive max-pooling aggregates individual features into team-based representations, maintaining team structure.
   - Layer normalization prepares team features for temporal modeling.
   - Second GRU network captures higher-order temporal patterns at the team level.
   - Parallel fully-connected classifier generates team activity predictions.

4. **Model Outputs**:
   - `person_output`: Class probability distributions for individual player activities.
   - `group_output`: Class probability distributions for collective team activities.

#### Training Configuration

- **Compute Resources**: Training performed on Kaggle's free-tier GPU resources (dual T4 GPUs with 15GB memory each) [Training Notebook](https://www.kaggle.com/code/AhmedYasser06/baseline-9-training).
- **Optimization**: AdamW with scheduled learning rate adjustment.
- **Batch Configuration:** 4 samples per GPU device
