"""
Model Definitions for Social Worker BERT

This module contains:
- BERT-based multi-label classifier
- Focal Loss for handling class imbalance
- Model utilities and helper functions
"""

import torch
import torch.nn as nn
from transformers import BertForSequenceClassification, BertTokenizer
from typing import Dict, Any, Optional

from .config import Config


class FocalLoss(nn.Module):
    """
    Focal Loss for multi-label classification to handle class imbalance.
    
    Focal Loss addresses class imbalance by down-weighting easy examples
    and focusing on hard examples.
    
    Args:
        alpha: Weighting factor for rare class (default: 0.25)
        gamma: Focusing parameter to down-weight easy examples (default: 2.0)
        reduction: Specifies the reduction to apply to the output
    """
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = 'mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for focal loss.
        
        Args:
            inputs: Logits from model
            targets: Ground truth labels
        
        Returns:
            Computed focal loss
        """
        # Apply sigmoid to get probabilities
        p = torch.sigmoid(inputs)
        
        # Calculate focal loss
        ce_loss = nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        p_t = p * targets + (1 - p) * (1 - targets)
        
        # Calculate alpha factor
        alpha_factor = targets * self.alpha + (1 - targets) * (1 - self.alpha)
        
        # Calculate modulating factor
        modulating_factor = (1.0 - p_t) ** self.gamma
        
        # Calculate focal loss
        focal_loss = alpha_factor * modulating_factor * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class MultiLabelBERTClassifier(nn.Module):
    """
    Multi-label BERT classifier for social worker skill detection.
    
    This model uses BERT as the backbone and adds a classification head
    for multi-label classification.
    """
    
    def __init__(self, model_name: str, num_labels: int, dropout_prob: float = 0.1):
        """
        Initialize the multi-label BERT classifier.
        
        Args:
            model_name: Name of the BERT model to use
            num_labels: Number of classification labels
            dropout_prob: Dropout probability for regularization
        """
        super(MultiLabelBERTClassifier, self).__init__()
        
        self.num_labels = num_labels
        self.model_name = model_name
        
        # Load BERT model with classification head
        self.bert = BertForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
            problem_type="multi_label_classification"
        )
        
        # Add dropout for regularization
        self.dropout = nn.Dropout(dropout_prob)
        
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, 
                labels: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the model.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            labels: Ground truth labels (optional)
        
        Returns:
            Dictionary containing model outputs
        """
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        return outputs
    
    def get_predictions(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, 
                       threshold: float = 0.5) -> torch.Tensor:
        """
        Get predictions from the model.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            threshold: Classification threshold
        
        Returns:
            Binary predictions
        """
        with torch.no_grad():
            outputs = self.forward(input_ids, attention_mask)
            logits = outputs.logits
            probabilities = torch.sigmoid(logits)
            predictions = (probabilities > threshold).int()
        return predictions
    
    def get_probabilities(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Get class probabilities from the model.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
        
        Returns:
            Class probabilities
        """
        with torch.no_grad():
            outputs = self.forward(input_ids, attention_mask)
            logits = outputs.logits
            probabilities = torch.sigmoid(logits)
        return probabilities


def get_loss_function(config: Config) -> nn.Module:
    """
    Get the appropriate loss function based on configuration.
    
    Args:
        config: Configuration object
    
    Returns:
        Loss function instance
    """
    if config.class_imbalance.focal_loss:
        print(f"📊 Using Focal Loss (alpha={config.class_imbalance.focal_alpha}, "
              f"gamma={config.class_imbalance.focal_gamma})")
        return FocalLoss(
            alpha=config.class_imbalance.focal_alpha,
            gamma=config.class_imbalance.focal_gamma
        )
    else:
        print("📊 Using Standard Binary Cross Entropy Loss")
        return nn.BCEWithLogitsLoss()


def create_model_and_tokenizer(config: Config) -> tuple[MultiLabelBERTClassifier, BertTokenizer]:
    """
    Create model and tokenizer instances.
    
    Args:
        config: Configuration object
    
    Returns:
        Tuple of (model, tokenizer)
    """
    print(f"🤖 Initializing model: {config.model.model_name}")
    
    if config.model.num_labels is None:
        raise ValueError("num_labels must be set in config before creating model")
    
    # Initialize tokenizer
    tokenizer = BertTokenizer.from_pretrained(config.model.model_name)
    
    # Initialize model
    num_labels = config.model.num_labels
    assert num_labels is not None, "num_labels must be set in config"
    
    model = MultiLabelBERTClassifier(
        model_name=config.model.model_name,
        num_labels=num_labels,
        dropout_prob=config.model.dropout_prob
    )
    
    # Move to device
    model.to(config.device)
    
    print("✅ Model initialized!")
    print(f"  Model: {config.model.model_name}")
    print(f"  Number of labels: {config.model.num_labels}")
    print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print(f"  Device: {config.device}")
    
    return model, tokenizer


def save_model(model: MultiLabelBERTClassifier, tokenizer: BertTokenizer, 
               config: Config, additional_data: Dict[str, Any], 
               save_path: str) -> None:
    """
    Save model and associated data.
    
    Args:
        model: Trained model
        tokenizer: Tokenizer
        config: Configuration
        additional_data: Additional data to save (mlb, skill_labels, etc.)
        save_path: Path to save the model
    """
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_config': {
            'model_name': config.model.model_name,
            'num_labels': config.model.num_labels,
            'dropout_prob': config.model.dropout_prob,
        },
        'tokenizer_name': config.model.model_name,
        'config': config.to_dict(),
        **additional_data
    }, save_path)
    
    print(f"💾 Model saved to {save_path}")


def load_model(save_path: str, device: torch.device) -> tuple[MultiLabelBERTClassifier, BertTokenizer, Dict[str, Any]]:
    """
    Load saved model and associated data.
    
    Args:
        save_path: Path to saved model
        device: Device to load model on
    
    Returns:
        Tuple of (model, tokenizer, additional_data)
    """
    print(f"📂 Loading model from {save_path}")
    
    # Load with weights_only=False to handle sklearn objects
    # This is safe for trusted model files
    checkpoint = torch.load(save_path, map_location=device, weights_only=False)
    
    # Recreate model
    model_config = checkpoint['model_config']
    model = MultiLabelBERTClassifier(
        model_name=model_config['model_name'],
        num_labels=model_config['num_labels'],
        dropout_prob=model_config.get('dropout_prob', 0.1)
    )
    
    # Load model state
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    # Load tokenizer
    tokenizer = BertTokenizer.from_pretrained(checkpoint['tokenizer_name'])
    
    # Extract additional data
    additional_data = {
        key: value for key, value in checkpoint.items() 
        if key not in ['model_state_dict', 'model_config', 'tokenizer_name']
    }
    
    print("✅ Model loaded successfully!")
    print(f"  Model: {model_config['model_name']}")
    print(f"  Number of labels: {model_config['num_labels']}")
    print(f"  Device: {device}")
    
    return model, tokenizer, additional_data


if __name__ == "__main__":
    # Example usage
    from .config import Config
    
    config = Config()
    config.model.num_labels = 10  # Example
    
    model, tokenizer = create_model_and_tokenizer(config)
    print(f"Model created with {config.model.num_labels} labels")
