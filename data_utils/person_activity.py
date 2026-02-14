"""
Person Activity Dataset for individual action recognition.
Used by: Baseline 3-A, 5-A, 7-A, 8-A

Supports both single-frame and temporal sequence modes.
"""
import cv2
import pickle
import torch
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from torch.utils.data import Dataset

from .base_dataset import BoxInfo, person_activity_labels


class Person_Activity_DataSet(Dataset):
    """
    Dataset for person-level activity classification.
    
    Operates in two modes:
    1. Single-frame mode (seq=False): Returns individual person crops
    2. Temporal mode (seq=True): Returns temporal sequences of person crops
    
    Args:
        videos_path: Directory containing video frame subfolders
        annot_path: Path to pickle file with frame annotations
        seq: Enable temporal sequence mode
        split: Video IDs to include in dataset
        labels: Mapping from activity names to class indices
        only_tar: Only use target frames (where frame_id == clip_dir)
        transform: Albumentations augmentation pipeline
    
    Returns:
        Single mode: (crop, label) where crop is (C,H,W)
        Temporal mode: (crops, labels) where crops is (N,T,C,H,W)
    """
    
    def __init__(
        self,
        videos_path: str,
        annot_path: str,
        seq: bool = False,
        split: list = [],
        labels: dict = {},
        only_tar: bool = False,
        transform=None
    ):
        self.frames_dir = Path(videos_path)
        self.augment = transform
        self.temporal_mode = seq
        self.target_frames_only = only_tar
        self.activity_mapping = labels
        
        # Build sample index
        self.data_index = self._construct_index(annot_path, split)
    
    def _construct_index(self, annot_file: str, video_ids: List[int]) -> List[Dict]:
        """Parse annotations and build dataset index."""
        with open(annot_file, 'rb') as f:
            raw_annotations = pickle.load(f)
        
        index = []
        
        for vid_id in video_ids:
            video_clips = raw_annotations[str(vid_id)]
            
            for clip_id, clip_data in video_clips.items():
                frame_annotations = clip_data['frame_boxes_dct']
                
                if self.temporal_mode:
                    # Build temporal sequence
                    frame_sequence = []
                    for frame_id, person_boxes in frame_annotations.items():
                        if self.target_frames_only and str(frame_id) != str(clip_id):
                            continue
                        
                        frame_sequence.append({
                            'video_id': vid_id,
                            'clip_id': clip_id,
                            'frame_id': frame_id,
                            'people': person_boxes
                        })
                    
                    if frame_sequence:
                        index.append({
                            'type': 'temporal',
                            'frames': frame_sequence
                        })
                
                else:
                    # Build single-frame samples
                    for frame_id, person_boxes in frame_annotations.items():
                        if self.target_frames_only and str(frame_id) != str(clip_id):
                            continue
                        
                        for person_box in person_boxes:
                            index.append({
                                'type': 'single',
                                'video_id': vid_id,
                                'clip_id': clip_id,
                                'frame_id': frame_id,
                                'detection': person_box
                            })
        
        return index
    
    def _read_frame(self, video_id: int, clip_id: str, frame_id: str) -> np.ndarray:
        """Load frame image from disk."""
        path = self.frames_dir / str(video_id) / str(clip_id) / f"{frame_id}.jpg"
        image = cv2.imread(str(path))
        if image is None:
            raise FileNotFoundError(f"Cannot read frame: {path}")
        return image
    
    def _extract_person_crop(
        self,
        image: np.ndarray,
        detection: BoxInfo
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract person region and create activity label.
        
        Returns:
            crop: Person crop (potentially augmented)
            label: One-hot activity label
        """
        x1, y1, x2, y2 = detection.box
        person_region = image[y1:y2, x1:x2]
        
        # Apply augmentation
        if self.augment is not None:
            augmented = self.augment(image=person_region)
            person_region = augmented['image']
        
        # Create label vector
        num_activities = len(self.activity_mapping)
        label_vector = np.zeros(num_activities)
        activity_idx = self.activity_mapping[detection.category]
        label_vector[activity_idx] = 1
        
        return person_region, label_vector
    
    def __len__(self) -> int:
        return len(self.data_index)
    
    def __getitem__(self, idx: int):
        sample = self.data_index[idx]
        
        if sample['type'] == 'single':
            # Single-frame mode
            image = self._read_frame(
                sample['video_id'],
                sample['clip_id'],
                sample['frame_id']
            )
            crop, label = self._extract_person_crop(image, sample['detection'])
            return crop, torch.from_numpy(label)
        
        else:  # temporal mode
            # Process temporal sequence
            all_people_sequences = {}  # person_id -> list of (crop, label)
            
            for frame_info in sample['frames']:
                frame = self._read_frame(
                    frame_info['video_id'],
                    frame_info['clip_id'],
                    frame_info['frame_id']
                )
                
                # Process each person in this frame
                for person_idx, person_det in enumerate(frame_info['people']):
                    crop, label = self._extract_person_crop(frame, person_det)
                    
                    if person_idx not in all_people_sequences:
                        all_people_sequences[person_idx] = {'crops': [], 'labels': []}
                    
                    all_people_sequences[person_idx]['crops'].append(crop)
                    all_people_sequences[person_idx]['labels'].append(label)
            
            # Stack into tensors
            person_sequences = []
            label_sequences = []
            
            for person_data in all_people_sequences.values():
                person_crops = np.stack(person_data['crops'])
                person_labels = np.stack(person_data['labels'])
                person_sequences.append(person_crops)
                label_sequences.append(person_labels)
            
            if person_sequences:
                # Shape: (num_people, num_frames, C, H, W)
                crops_array = np.stack(person_sequences)
                labels_array = np.stack(label_sequences)
                
                return (
                    torch.from_numpy(crops_array),
                    torch.from_numpy(labels_array)
                )
            
            return None, None


def person_collate_fn(batch):
    """
    Collate function for person activity batches.
    Pads number of people to 12 and extracts last frame labels.
    
    Args:
        batch: List of (clips, labels) tuples
    
    Returns:
        Padded clips: (B, 12, T, C, H, W)
        Flattened labels: (B*12, num_classes)
    """
    clips_list, labels_list = zip(*batch)
    
    target_num_people = 12
    padded_clips = []
    padded_labels = []
    
    for clip, label in zip(clips_list, labels_list):
        num_people = clip.size(0)
        
        if num_people < target_num_people:
            deficit = target_num_people - num_people
            
            # Pad clips
            clip_pad_dims = list(clip.shape)
            clip_pad_dims[0] = deficit
            clip_pad = torch.zeros(clip_pad_dims)
            clip = torch.cat([clip, clip_pad], dim=0)
            
            # Pad labels
            label_pad_dims = list(label.shape)
            label_pad_dims[0] = deficit
            label_pad = torch.zeros(label_pad_dims)
            label = torch.cat([label, label_pad], dim=0)
        
        padded_clips.append(clip)
        padded_labels.append(label)
    
    # Create batches
    batched_clips = torch.stack(padded_clips)
    batched_labels = torch.stack(padded_labels)
    
    # Extract last frame labels and flatten
    # From (B, N, T, C) to (B*N, C)
    last_frame_labels = batched_labels[:, :, -1, :]
    batch_size, num_people, num_classes = last_frame_labels.shape
    flat_labels = last_frame_labels.view(batch_size * num_people, num_classes)
    
    return batched_clips, flat_labels
