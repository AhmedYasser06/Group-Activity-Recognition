"""
Group Activity Dataset for volleyball group activity recognition.
Used by: Baseline 1, 3-B, 4, 5-B, 6, 7-B, 8-B

This dataset supports multiple modes:
- Full frame or person crops
- Single frame or temporal sequences
- Sorted or unsorted person ordering
"""
import cv2
import pickle
import torch
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from torch.utils.data import Dataset

from .base_dataset import BoxInfo, group_activity_labels


class Group_Activity_DataSet(Dataset):
    """
    Flexible dataset for group activity recognition with multiple output modes.
    
    Args:
        videos_path: Root directory containing video frame folders
        annot_path: Path to pickle file with annotations
        seq: If True, returns temporal sequences; if False, single frames
        crops: If True, returns person crops; if False, full frames
        sort: If True, sorts people by x-coordinate (for team-aware models)
        split: List of video IDs to include in this split
        only_tar: If True, uses only target frames (frame_id == clip_dir)
        labels: Dictionary mapping activity names to class indices
        transform: Albumentations transform pipeline
    
    Output shapes depend on mode:
        seq=False, crops=False: (C, H, W), (num_classes,)
        seq=True,  crops=False: (T, C, H, W), (T, num_classes)
        seq=False, crops=True:  (N, C, H, W), (num_classes,)
        seq=True,  crops=True:  (N, T, C, H, W), (T, num_classes)
    where N=num_people, T=num_frames, C=channels, H=height, W=width
    """
    
    def __init__(
        self,
        videos_path: str,
        annot_path: str,
        seq: bool = False,
        crops: bool = False,
        sort: bool = False,
        split: list = [],
        only_tar: bool = False,
        labels: dict = {},
        transform=None
    ):
        self.video_root = Path(videos_path)
        self.apply_transform = transform
        self.is_temporal = seq
        self.use_person_crops = crops
        self.sort_by_position = sort
        self.target_only = only_tar
        self.class_mapping = labels
        
        # Load and parse annotations
        self.samples = self._build_sample_index(annot_path, split)
    
    def _build_sample_index(self, annot_file: str, video_ids: List[int]) -> List[Dict]:
        """Construct dataset index from annotation file."""
        with open(annot_file, 'rb') as f:
            annotations = pickle.load(f)
        
        dataset_samples = []
        
        for vid_id in video_ids:
            video_data = annotations[str(vid_id)]
            
            for scene_id, scene_info in video_data.items():
                activity_class = scene_info['category']
                frame_annotations = scene_info['frame_boxes_dct']
                
                # Mode 1: Full frame, single time step
                if not self.use_person_crops and not self.is_temporal:
                    for fid, detections in frame_annotations.items():
                        if self.target_only and str(fid) != str(scene_id):
                            continue
                        
                        img_path = self.video_root / str(vid_id) / str(scene_id) / f"{fid}.jpg"
                        dataset_samples.append({
                            'mode': 'single_frame',
                            'image_path': str(img_path),
                            'activity': activity_class
                        })
                
                # Mode 2: Full frame, temporal sequence
                elif not self.use_person_crops and self.is_temporal:
                    frame_sequence = []
                    for fid, detections in frame_annotations.items():
                        if self.target_only and str(fid) != str(scene_id):
                            continue
                        img_path = self.video_root / str(vid_id) / str(scene_id) / f"{fid}.jpg"
                        frame_sequence.append(str(img_path))
                    
                    if frame_sequence:
                        dataset_samples.append({
                            'mode': 'temporal_frames',
                            'image_paths': frame_sequence,
                            'activity': activity_class
                        })
                
                # Mode 3: Person crops, single time step
                elif self.use_person_crops and not self.is_temporal:
                    for fid, detections in frame_annotations.items():
                        if self.target_only and str(fid) != str(scene_id):
                            continue
                        
                        img_path = self.video_root / str(vid_id) / str(scene_id) / f"{fid}.jpg"
                        dataset_samples.append({
                            'mode': 'single_crops',
                            'image_path': str(img_path),
                            'detections': detections,
                            'activity': activity_class
                        })
                
                # Mode 4: Person crops, temporal sequence
                else:
                    temporal_data = []
                    for fid, detections in frame_annotations.items():
                        if self.target_only and str(fid) != str(scene_id):
                            continue
                        
                        img_path = self.video_root / str(vid_id) / str(scene_id) / f"{fid}.jpg"
                        temporal_data.append((str(img_path), detections))
                    
                    if temporal_data:
                        dataset_samples.append({
                            'mode': 'temporal_crops',
                            'frame_detections': temporal_data,
                            'activity': activity_class
                        })
        
        return dataset_samples
    
    def _read_image(self, path: str) -> np.ndarray:
        """Load image from disk."""
        img = cv2.imread(path)
        if img is None:
            raise FileNotFoundError(f"Cannot load image: {path}")
        return img
    
    def _get_bbox_center_x(self, bbox: List[int]) -> float:
        """Calculate horizontal center of bounding box."""
        x1, _, x2, _ = bbox
        return (x1 + x2) / 2.0
    
    def _crop_and_process_person(
        self, 
        image: np.ndarray, 
        detection: BoxInfo
    ) -> torch.Tensor:
        """Extract person region and apply transformations."""
        x1, y1, x2, y2 = detection.box
        person_region = image[y1:y2, x1:x2]
        
        if self.apply_transform is not None:
            augmented = self.apply_transform(image=person_region)
            person_region = augmented['image']
        
        return person_region
    
    def _extract_people(
        self, 
        image: np.ndarray, 
        detections: List[BoxInfo]
    ) -> Tuple[List[torch.Tensor], List[float]]:
        """Extract all person crops from an image."""
        person_crops = []
        x_positions = []
        
        for det in detections:
            crop = self._crop_and_process_person(image, det)
            person_crops.append(crop)
            x_positions.append(self._get_bbox_center_x(det.box))
        
        return person_crops, x_positions
    
    def _create_activity_label(self, activity_name: str) -> torch.Tensor:
        """Convert activity name to one-hot encoded tensor."""
        num_classes = len(self.class_mapping)
        label_vector = torch.zeros(num_classes)
        class_idx = self.class_mapping[activity_name]
        label_vector[class_idx] = 1.0
        return label_vector
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sample = self.samples[index]
        mode = sample['mode']
        
        # Single full frame
        if mode == 'single_frame':
            frame = self._read_image(sample['image_path'])
            if self.apply_transform is not None:
                frame = self.apply_transform(image=frame)['image']
            
            label = self._create_activity_label(sample['activity'])
            return frame, label
        
        # Temporal full frames
        elif mode == 'temporal_frames':
            frame_list = []
            label_list = []
            base_label = self._create_activity_label(sample['activity'])
            
            for path in sample['image_paths']:
                frame = self._read_image(path)
                if self.apply_transform is not None:
                    frame = self.apply_transform(image=frame)['image']
                frame_list.append(frame)
                label_list.append(base_label)
            
            frames_tensor = torch.stack(frame_list)
            labels_tensor = torch.stack(label_list)
            return frames_tensor, labels_tensor
        
        # Single frame with person crops
        elif mode == 'single_crops':
            frame = self._read_image(sample['image_path'])
            crops, _ = self._extract_people(frame, sample['detections'])
            crops_tensor = torch.stack(crops)
            label = self._create_activity_label(sample['activity'])
            return crops_tensor, label
        
        # Temporal sequence with person crops
        else:  # mode == 'temporal_crops'
            all_frame_crops = []
            label_list = []
            base_label = self._create_activity_label(sample['activity'])
            
            for img_path, detections in sample['frame_detections']:
                frame = self._read_image(img_path)
                crops, positions = self._extract_people(frame, detections)
                
                # Sort by x-position if required (for team-aware models)
                if self.sort_by_position:
                    sorted_indices = sorted(range(len(positions)), key=lambda k: positions[k])
                    crops = [crops[i] for i in sorted_indices]
                
                frame_crops_tensor = torch.stack(crops)
                all_frame_crops.append(frame_crops_tensor)
                label_list.append(base_label)
            
            # Reshape to (num_people, num_frames, C, H, W)
            sequence = torch.stack(all_frame_crops)  # (T, N, C, H, W)
            sequence = sequence.permute(1, 0, 2, 3, 4)  # (N, T, C, H, W)
            
            labels_tensor = torch.stack(label_list)
            return sequence, labels_tensor


def group_collate_fn(batch):
    """
    Collate function for group activity batches with person crops.
    Pads the number of people to a fixed size (12 players).
    
    Args:
        batch: List of (clips, labels) tuples
    
    Returns:
        Padded clips and labels tensors
    """
    clips, labels = zip(*batch)
    
    max_people = 12
    padded_clips = []
    
    for clip in clips:
        num_people = clip.size(0)
        if num_people < max_people:
            # Create padding tensor with same shape as clip but different first dim
            pad_shape = list(clip.shape)
            pad_shape[0] = max_people - num_people
            padding = torch.zeros(pad_shape)
            padded_clip = torch.cat([clip, padding], dim=0)
        else:
            padded_clip = clip
        
        padded_clips.append(padded_clip)
    
    batched_clips = torch.stack(padded_clips)
    batched_labels = torch.stack(labels)
    
    # Use label from last frame for temporal sequences
    if batched_labels.dim() > 2:
        batched_labels = batched_labels[:, -1, :]
    
    return batched_clips, batched_labels
