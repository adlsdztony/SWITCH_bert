"""
Social Worker BERT: Multi-label Classification Library for Social Worker Skills

This library provides a complete pipeline for training and inference of BERT-based
multi-label classification models for social worker skill detection.

Main Components:
- config: Configuration management
- data: Data preprocessing and loading
- models: BERT model definitions and loss functions
- training: Training pipeline and utilities
- inference: Prediction and evaluation utilities
- pipeline: End-to-end pipeline orchestration

Usage:
    # Quick start - full pipeline
    from social_worker_bert.pipeline import SocialWorkerBERTPipeline
    
    pipeline = SocialWorkerBERTPipeline('config.yaml')
    pipeline.run()
    
    # Individual components
    from social_worker_bert.data import DataProcessor
    from social_worker_bert.models import MultiLabelBERTClassifier
    from social_worker_bert.training import Trainer
    from social_worker_bert.inference import Predictor
"""

__version__ = "1.0.0"
__author__ = "Social Worker BERT Team"

# Import main classes for easy access
from .config import Config
from .data import DataProcessor
from .models import MultiLabelBERTClassifier, FocalLoss
from .training import Trainer
from .inference import Predictor
from .pipeline import SocialWorkerBERTPipeline

__all__ = [
    'Config',
    'DataProcessor', 
    'MultiLabelBERTClassifier',
    'FocalLoss',
    'Trainer',
    'Predictor',
    'SocialWorkerBERTPipeline'
]
