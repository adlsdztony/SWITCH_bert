"""
Configuration Management for Social Worker BERT

This module handles all configuration parameters for the social worker skill classification.
Supports YAML configuration files and programmatic configuration.
"""

import yaml
import torch
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from pathlib import Path


@dataclass
class DataConfig:
    """Data-related configuration parameters."""
    # Support both single file and separate train/test files
    data_file: Optional[str] = './tri.csv'  # Single file mode (legacy)
    train_file: Optional[str] = None  # Separate train file
    test_file: Optional[str] = None   # Separate test file
    val_file: Optional[str] = None    # Separate validation file (optional)
    
    # Data processing parameters
    annotator_columns: List[str] = field(default_factory=lambda: ['Anthea', 'Karen', 'Kimmy'])
    message_column: str = 'message'
    
    # Split parameters (only used in single file mode)
    test_size: float = 0.1
    val_size: float = 0.1
    random_state: int = 42
    
    # Label processing parameters
    label_strategy: str = 'ensemble'  # 'ensemble', 'majority', 'individual'
    min_annotator_agreement: int = 2
    
    # Data mode - automatically determined or can be set explicitly
    mode: str = 'auto'  # 'auto', 'single_file', 'separate_files'


@dataclass
class ModelConfig:
    """Model-related configuration parameters."""
    model_name: str = 'bert-base-uncased'
    max_length: int = 512
    num_labels: Optional[int] = None  # Will be set automatically
    dropout_prob: float = 0.1


@dataclass
class TrainingConfig:
    """Training-related configuration parameters."""
    batch_size: int = 16
    learning_rate: float = 2e-5
    num_epochs: int = 5
    warmup_steps: int = 500
    weight_decay: float = 0.01
    use_gpu: bool = True
    save_model: bool = True
    model_save_path: str = 'social_worker_bert_model'
    gradient_clip_norm: float = 1.0


@dataclass
class ClassImbalanceConfig:
    """Class imbalance handling configuration."""
    use_class_weights: bool = True
    weight_method: str = 'balanced'  # 'balanced', 'sqrt_balanced', 'none'
    min_samples_threshold: int = 5
    rare_label_handling: str = 'remove'  # 'remove', 'upsample', 'keep'
    focal_loss: bool = True
    focal_alpha: float = 0.25
    focal_gamma: float = 2.0


@dataclass
class DataAugmentationConfig:
    """Data augmentation configuration."""
    enable: bool = True
    rare_threshold: float = 1.0
    augment_factor: int = 3
    methods: List[str] = field(default_factory=lambda: ['paraphrase', 'synonym_replacement'])


@dataclass
class EvaluationConfig:
    """Evaluation configuration."""
    classification_threshold: float = 0.5
    metrics: List[str] = field(default_factory=lambda: ['f1_macro', 'f1_micro', 'precision', 'recall'])
    save_detailed_predictions: bool = True
    multi_thresholds: Optional[Dict[str, float]] = None


class Config:
    """Main configuration class that combines all configuration sections."""
    
    def __init__(self, config_path: Optional[str] = None, **kwargs):
        """
        Initialize configuration.
        
        Args:
            config_path: Path to YAML configuration file
            **kwargs: Override any configuration parameters
        """
        # Initialize with defaults
        self.data = DataConfig()
        self.model = ModelConfig()
        self.training = TrainingConfig()
        self.class_imbalance = ClassImbalanceConfig()
        self.data_augmentation = DataAugmentationConfig()
        self.evaluation = EvaluationConfig()
        
        # Load from file if provided
        if config_path:
            self.load_from_file(config_path)
        
        # Override with any provided kwargs
        self.update_from_dict(kwargs)
        
        # Set device
        self.device = torch.device('cuda' if torch.cuda.is_available() and self.training.use_gpu else 'cpu')
        
        # Set random seeds
        self.set_random_seeds()
    
    def load_from_file(self, config_path: str):
        """Load configuration from YAML file."""
        config_file = Path(config_path)
        if not config_file.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_file, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        self.update_from_dict(config_dict)
    
    def update_from_dict(self, config_dict: Dict[str, Any]):
        """Update configuration from dictionary."""
        for section_name, section_config in config_dict.items():
            if hasattr(self, section_name):
                section = getattr(self, section_name)
                for key, value in section_config.items():
                    if hasattr(section, key):
                        setattr(section, key, value)
                    else:
                        print(f"Warning: Unknown config key {section_name}.{key}")
            else:
                print(f"Warning: Unknown config section {section_name}")
    
    def save_to_file(self, config_path: str):
        """Save current configuration to YAML file."""
        config_dict = self.to_dict()
        with open(config_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'data': self.data.__dict__,
            'model': self.model.__dict__,
            'training': self.training.__dict__,
            'class_imbalance': self.class_imbalance.__dict__,
            'data_augmentation': self.data_augmentation.__dict__,
            'evaluation': self.evaluation.__dict__
        }
    
    def set_random_seeds(self):
        """Set random seeds for reproducibility."""
        torch.manual_seed(self.data.random_state)
        np.random.seed(self.data.random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.data.random_state)
            torch.cuda.manual_seed_all(self.data.random_state)
    
    def print_summary(self):
        """Print configuration summary."""
        print("🔧 Configuration Summary:")
        print("=" * 50)
        print(f"📱 Device: {self.device}")
        
        # Print data configuration based on mode
        self.determine_data_mode()
        if self.data.mode == 'separate_files':
            print("📊 Data mode: Separate files")
            print(f"  📈 Train file: {self.data.train_file}")
            print(f"  📉 Test file: {self.data.test_file}")
        else:
            print("📊 Data mode: Single file")
            print(f"  📊 Data file: {self.data.data_file}")
            print(f"  📉 Test size: {self.data.test_size}")
            print(f"  📊 Val size: {self.data.val_size}")
        
        print(f"🤖 Model: {self.model.model_name}")
        print(f"📏 Max length: {self.model.max_length}")
        print(f"🎯 Batch size: {self.training.batch_size}")
        print(f"📈 Learning rate: {self.training.learning_rate}")
        print(f"🔄 Epochs: {self.training.num_epochs}")
        print(f"🏷️  Label strategy: {self.data.label_strategy}")
        print(f"🎲 Random state: {self.data.random_state}")
        print(f"⚖️  Class weights: {self.class_imbalance.use_class_weights}")
        print(f"🔥 Focal loss: {self.class_imbalance.focal_loss}")
        print(f"🔀 Data augmentation: {self.data_augmentation.enable}")
        print("=" * 50)
    
    def determine_data_mode(self):
        """Determine data mode based on configuration."""
        if self.data.mode == 'auto':
            if self.data.train_file and self.data.test_file:
                self.data.mode = 'separate_files'
            elif self.data.data_file:
                self.data.mode = 'single_file'
            else:
                raise ValueError("Either data_file or both train_file and test_file must be provided")
        
        # Validate configuration
        if self.data.mode == 'separate_files':
            if not self.data.train_file or not self.data.test_file:
                raise ValueError("Both train_file and test_file must be provided for separate_files mode")
        elif self.data.mode == 'single_file':
            if not self.data.data_file:
                raise ValueError("data_file must be provided for single_file mode")
    
    def get_data_files(self) -> Dict[str, str]:
        """Get data files based on mode."""
        self.determine_data_mode()
        
        if self.data.mode == 'separate_files':
            files = {
                'train': self.data.train_file,
                'test': self.data.test_file
            }
            if self.data.val_file:
                files['val'] = self.data.val_file
            return files
        else:
            return {
                'single': self.data.data_file
            }


def create_default_config_file(file_path: str = "config.yaml"):
    """Create a default configuration YAML file."""
    config = Config()
    config.save_to_file(file_path)
    print(f"✅ Default configuration saved to {file_path}")


if __name__ == "__main__":
    # Example usage
    config = Config()
    config.print_summary()
    
    # Create default config file
    create_default_config_file()
