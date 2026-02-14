"""
Hierarchical Dataset for end-to-end person and group activity recognition.
Used by: Baseline 9

Returns both person-level and group-level labels for joint training.
"""
import cv2
import pickle
import torch
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict
from torch.utils.data import Dataset

from .base_dataset import BoxInfo, activities_labels


class Hierarchical_Group_Activity_DataSet(Dataset):
    """
    Dataset for hierarchical activity recognition with dual-level labels.
    
    This dataset returns:
    - Person crops in temporal sequences
    - Person-level activity labels for each person at each timestep
    - Group-level activity label for the entire scene
    
    Args:
        videos_path: Root directory with video frames
        annot_path: Path to annotations pickle file
        split: List of video IDs for this data split
        labels: Dictionary with 'person' and 'group' label mappings
        transform: Albumentations transform to apply to crops
    
    Returns:
        tuple: (clips, person_labels, group_labels)
            - clips: (N, T, C, H, W) - N people, T frames
            - person_labels: (N, T, num_person_classes)
            - group_labels: (T, num_group_classes)
    """
    
    def __init__(
        self,
        videos_path: str,
        annot_path: str,
        split: list = [],
        labels: dict = {},
        transform=None
    ):
        self.frame_root = Path(videos_path)
        self.augmentation = transform
        self.label_maps = labels
        
        # Build dataset index
        self.scene_samples = self._parse_annotations(annot_path, split)
    
    def _parse_annotations(self, annot_file: str, video_list: List[int]) -> List[Dict]:
        """Build index of scenes with temporal annotations."""
        with open(annot_file, 'rb') as f:
            annotation_data = pickle.load(f)
        
        samples = []
        
        for video_idx in video_list:
            video_scenes = annotation_data[str(video_idx)]
            
            for scene_key, scene_data in video_scenes.items():
                group_activity = scene_data['category']
                frame_data = scene_data['frame_boxes_dct']
                
                # Collect temporal sequence
                temporal_sequence = []
                for frame_key, person_detections in frame_data.items():
                    frame_path = self.frame_root / str(video_idx) / str(scene_key) / f"{frame_key}.jpg"
                    temporal_sequence.append({
                        'path': str(frame_path),
                        'people': person_detections
                    })
                
                samples.append({
                    'frames': temporal_sequence,
                    'group_activity': group_activity
                })
        
        return samples
    
    def _load_image(self, filepath: str) -> np.ndarray:
        """Read image from filesystem."""
        image = cv2.imread(filepath)
        if image is None:
            raise IOError(f"Failed to read image: {filepath}")
        return image
    
    def _compute_horizontal_center(self, bbox: List[int]) -> float:
        """Get x-coordinate of bounding box center."""
        left, _, right, _ = bbox
        return (left + right) * 0.5
    
    def _process_person_detection(
        self,
        full_image: np.ndarray,
        detection: BoxInfo
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract person crop and create person activity label.
        
        Returns:
            crop: Processed person crop tensor
            label: One-hot person activity label
        """
        x1, y1, x2, y2 = detection.box
        crop_region = full_image[y1:y2, x1:x2]
        
        # Apply augmentation if provided
        if self.augmentation is not None:
            transformed = self.augmentation(image=crop_region)
            crop_region = transformed['image']
        
        # Create person activity label
        person_class_count = len(self.label_maps['person'])
        activity_label = torch.zeros(person_class_count)
        activity_idx = self.label_maps['person'][detection.category]
        activity_label[activity_idx] = 1.0
        
        return crop_region, activity_label
    
    def _process_frame(
        self,
        frame_info: Dict
    ) -> Tuple[List[torch.Tensor], List[float], List[torch.Tensor]]:
        """
        Process a single frame and extract all person data.
        
        Returns:
            crops: List of person crop tensors
            x_coords: List of x-coordinates for sorting
            labels: List of person activity labels
        """
        image = self._load_image(frame_info['path'])
        detections = frame_info['people']
        
        crop_list = []
        position_list = []
        label_list = []
        
        for person_det in detections:
            crop, label = self._process_person_detection(image, person_det)
            x_pos = self._compute_horizontal_center(person_det.box)
            
            crop_list.append(crop)
            position_list.append(x_pos)
            label_list.append(label)
        
        return crop_list, position_list, label_list
    
    def _create_group_label(self, activity_name: str) -> torch.Tensor:
        """Convert group activity name to one-hot tensor."""
        num_group_classes = len(self.label_maps['group'])
        label_tensor = torch.zeros(num_group_classes)
        class_index = self.label_maps['group'][activity_name]
        label_tensor[class_index] = 1.0
        return label_tensor
    
    def __len__(self) -> int:
        return len(self.scene_samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        sample = self.scene_samples[idx]
        
        # Process each frame in the temporal sequence
        sequence_crops = []
        sequence_person_labels = []
        sequence_group_labels = []
        
        group_label = self._create_group_label(sample['group_activity'])
        
        for frame_data in sample['frames']:
            crops, positions, person_labels = self._process_frame(frame_data)
            
            # Sort people by x-coordinate (left to right)
            sorted_data = sorted(
                zip(positions, crops, person_labels),
                key=lambda item: item[0]
            )
            
            # Unpack sorted data
            sorted_crops = [item[1] for item in sorted_data]
            sorted_person_labels = [item[2] for item in sorted_data]
            
            # Stack into tensors
            frame_crops = torch.stack(sorted_crops)
            frame_person_labels = torch.stack(sorted_person_labels)
            
            sequence_crops.append(frame_crops)
            sequence_person_labels.append(frame_person_labels)
            sequence_group_labels.append(group_label)
        
        # Combine temporal dimension
        # clips: (T, N, C, H, W) -> (N, T, C, H, W)
        clips_tensor = torch.stack(sequence_crops).permute(1, 0, 2, 3, 4)
        
        # person_labels: (T, N, num_classes) -> (N, T, num_classes)
        person_labels_tensor = torch.stack(sequence_person_labels).permute(1, 0, 2)
        
        # group_labels: (T, num_classes)
        group_labels_tensor = torch.stack(sequence_group_labels)
        
        return clips_tensor, person_labels_tensor, group_labels_tensor


def hierarchical_collate_fn(batch):
    """
    Collate function for hierarchical batches.
    Pads people dimension to 12 and handles both person and group labels.
    
    Args:
        batch: List of (clips, person_labels, group_labels) tuples
    
    Returns:
        Padded clips, person labels, and group labels
    """
    clips_list, person_labels_list, group_labels_list = zip(*batch)
    
    fixed_num_people = 12
    padded_clips = []
    padded_person_labels = []
    
    for clip, person_labels in zip(clips_list, person_labels_list):
        current_num_people = clip.size(0)
        
        if current_num_people < fixed_num_people:
            deficit = fixed_num_people - current_num_people
            
            # Pad clips: (N, T, C, H, W)
            clip_pad_shape = list(clip.shape)
            clip_pad_shape[0] = deficit
            clip_padding = torch.zeros(clip_pad_shape)
            padded_clip = torch.cat([clip, clip_padding], dim=0)
            
            # Pad person labels: (N, T, num_classes)
            label_pad_shape = list(person_labels.shape)
            label_pad_shape[0] = deficit
            label_padding = torch.zeros(label_pad_shape)
            padded_labels = torch.cat([person_labels, label_padding], dim=0)
        else:
            padded_clip = clip
            padded_labels = person_labels
        
        padded_clips.append(padded_clip)
        padded_person_labels.append(padded_labels)
    
    # Stack into batches
    batched_clips = torch.stack(padded_clips)
    batched_person_labels = torch.stack(padded_person_labels)
    batched_group_labels = torch.stack(group_labels_list)
    
    # Reshape person labels: (B, N, T, C) -> (B*N, C) for loss computation
    batch_size, num_people, num_frames, num_classes = batched_person_labels.shape
    flat_person_labels = batched_person_labels.view(batch_size * num_people, num_frames, num_classes)
    flat_person_labels = flat_person_labels[:, -1, :]  # Use last frame
    
    # Use last frame for group labels: (B, T, C) -> (B, C)
    final_group_labels = batched_group_labels[:, -1, :]
    
    return batched_clips, flat_person_labels, final_group_labels
