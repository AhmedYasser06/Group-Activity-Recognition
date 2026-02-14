"""
Base dataset utilities and common classes.
"""
from dataclasses import dataclass
from typing import List


@dataclass
class BoxInfo:
    """Information about a bounding box."""
    box: List[int]  # [x_min, y_min, x_max, y_max]
    category: str
    
    def __post_init__(self):
        """Validate box coordinates."""
        if len(self.box) != 4:
            raise ValueError(f"Box must have 4 coordinates, got {len(self.box)}")


# Activity labels
person_activity_classes = [
    "Waiting", "Setting", "Digging", "Falling", 
    "Spiking", "Blocking", "Jumping", "Moving", "Standing"
]
person_activity_labels = {
    class_name.lower(): i for i, class_name in enumerate(person_activity_classes)
}

group_activity_classes = [
    "r_set", "r_spike", "r-pass", "r_winpoint", 
    "l_winpoint", "l-pass", "l-spike", "l_set"
]
group_activity_labels = {
    class_name: i for i, class_name in enumerate(group_activity_classes)
}

activities_labels = {
    "person": person_activity_labels, 
    "group": group_activity_labels
}
