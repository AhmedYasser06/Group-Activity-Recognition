"""
Centralized path configuration for the project.
Handles different environments (local, Kaggle, cloud, etc.)
"""
import os
from pathlib import Path


class PathConfig:
    """Centralized path management for different environments."""
    
    def __init__(self, environment='local', custom_root=None):
        """
        Initialize path configuration.
        
        Args:
            environment: str, one of ['local', 'kaggle', 'colab', 'custom']
            custom_root: str, custom root path (used when environment='custom')
        """
        self.environment = environment
        
        # Determine root based on environment
        if environment == 'local':
            self.project_root = Path(__file__).parent.parent.absolute()
        elif environment == 'kaggle':
            self.project_root = Path('/kaggle/working/Group-Activity-Recognition')
        elif environment == 'colab':
            self.project_root = Path('/content/Group-Activity-Recognition')
        elif environment == 'custom' and custom_root:
            self.project_root = Path(custom_root)
        else:
            # Default to local
            self.project_root = Path(__file__).parent.parent.absolute()
        
        # Data paths
        self.data_root = self.project_root / 'data'
        self.videos_path = self.data_root / 'videos'
        self.annot_path = self.data_root / 'annot_all.pkl'
        
        # Output paths
        self.output_root = self.project_root / 'outputs'
        self.checkpoint_root = self.output_root / 'checkpoints'
        self.log_root = self.output_root / 'logs'
        self.tensorboard_root = self.output_root / 'tensorboard'
        
        # Baseline paths
        self.baselines_root = self.project_root / 'baselines'
        
        # Utils paths
        self.utils_root = self.project_root / 'utils'
        
        # Create necessary directories
        self._create_directories()
    
    def _create_directories(self):
        """Create necessary output directories if they don't exist."""
        directories = [
            self.output_root,
            self.checkpoint_root,
            self.log_root,
            self.tensorboard_root
        ]
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    def get_baseline_path(self, baseline_name):
        """Get path to specific baseline directory."""
        return self.baselines_root / baseline_name
    
    def get_output_path(self, baseline_name, experiment_name=None):
        """Get output path for a specific baseline and experiment."""
        if experiment_name:
            return self.output_root / baseline_name / experiment_name
        return self.output_root / baseline_name
    
    def get_checkpoint_path(self, baseline_name, checkpoint_name):
        """Get full path to a checkpoint file."""
        return self.checkpoint_root / baseline_name / checkpoint_name
    
    def get_config_path(self, baseline_name, config_file='config.yml'):
        """Get path to baseline config file."""
        baseline_path = self.get_baseline_path(baseline_name)
        # Handle two-stage baselines
        if (baseline_path / 'step_a_person' / config_file).exists():
            return baseline_path / 'step_a_person' / config_file
        return baseline_path / config_file
    
    def __repr__(self):
        return f"PathConfig(environment='{self.environment}', root='{self.project_root}')"


# Global path config instance
# Users can override this by creating a new instance
path_config = PathConfig()


def get_paths(environment='local', custom_root=None):
    """
    Get path configuration for specified environment.
    
    Args:
        environment: str, one of ['local', 'kaggle', 'colab', 'custom']
        custom_root: str, custom root path (used when environment='custom')
    
    Returns:
        PathConfig instance
    
    Examples:
        >>> # Local development
        >>> paths = get_paths('local')
        >>> print(paths.data_root)
        
        >>> # Kaggle
        >>> paths = get_paths('kaggle')
        >>> print(paths.videos_path)
        
        >>> # Custom
        >>> paths = get_paths('custom', '/my/custom/path')
    """
    return PathConfig(environment=environment, custom_root=custom_root)


if __name__ == '__main__':
    # Test different environments
    print("Local environment:")
    local_paths = get_paths('local')
    print(f"  Project root: {local_paths.project_root}")
    print(f"  Data root: {local_paths.data_root}")
    print(f"  Output root: {local_paths.output_root}")
    
    print("\nKaggle environment:")
    kaggle_paths = get_paths('kaggle')
    print(f"  Project root: {kaggle_paths.project_root}")
    print(f"  Videos path: {kaggle_paths.videos_path}")
