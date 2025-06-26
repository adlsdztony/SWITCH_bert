"""
Training Module for Social Worker BERT

This module handles:
- Training loop implementation
- Validation and evaluation
- Optimizer and scheduler setup
- Metrics calculation
- Model checkpointing
"""

import torch
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import f1_score
import numpy as np
from typing import List, Tuple, Dict, Any, Optional
import time

from .config import Config
from .models import MultiLabelBERTClassifier, get_loss_function

# Import AdamW optimizer
try:
    from torch.optim import AdamW
except ImportError:
    # Fallback for older versions - this may not work in newer transformers
    print("Warning: AdamW not available from torch.optim")
    raise ImportError("AdamW optimizer not available")


class Trainer:
    """Main trainer class for social worker skill classification."""
    
    def __init__(self, config: Config, model: MultiLabelBERTClassifier):
        """
        Initialize trainer.
        
        Args:
            config: Configuration object
            model: Model to train
        """
        self.config = config
        self.model = model
        self.device = config.device
        
        # Training tracking
        self.train_losses = []
        self.val_losses = []
        self.val_f1_scores = []
        self.best_val_f1 = 0.0
        self.best_model_state = None
        
        # Initialize optimizer, scheduler, and loss function
        self.optimizer = None
        self.scheduler = None
        self.criterion = None
        
    def setup_training(self, train_loader: DataLoader):
        """
        Setup optimizer, scheduler, and loss function.
        
        Args:
            train_loader: Training data loader
        """
        print("⚙️  Setting up training components...")
        
        # Optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.config.training.learning_rate,
            weight_decay=self.config.training.weight_decay
        )
        
        # Scheduler
        total_steps = len(train_loader) * self.config.training.num_epochs
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.config.training.warmup_steps,
            num_training_steps=total_steps
        )
        
        # Loss function
        self.criterion = get_loss_function(self.config)
        
        print("✅ Training setup completed!")
        print(f"  Optimizer: AdamW (lr={self.config.training.learning_rate})")
        print(f"  Total training steps: {total_steps}")
        print(f"  Warmup steps: {self.config.training.warmup_steps}")
        print(f"  Loss function: {type(self.criterion).__name__}")
    
    def calculate_f1_score(self, predictions: torch.Tensor, labels: torch.Tensor, 
                          threshold: float = 0.5) -> Tuple[float, float, np.ndarray]:
        """
        Calculate F1 score for multi-label classification.
        
        Args:
            predictions: Model predictions (logits)
            labels: Ground truth labels
            threshold: Classification threshold
        
        Returns:
            Tuple of (f1_macro, f1_micro, f1_per_class)
        """
        pred_binary = (predictions > threshold).int()
        
        # Convert to numpy for sklearn metrics
        labels_np = labels.cpu().numpy()
        pred_np = pred_binary.cpu().numpy()
        
        # Calculate macro and micro F1
        f1_macro = f1_score(labels_np, pred_np, average='macro', zero_division=0)
        f1_micro = f1_score(labels_np, pred_np, average='micro', zero_division=0)
        
        # Calculate per-class metrics
        f1_per_class = f1_score(labels_np, pred_np, average=None, zero_division=0)
        
        return f1_macro, f1_micro, f1_per_class
    
    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, float, float]:
        """
        Train for one epoch.
        
        Args:
            train_loader: Training data loader
        
        Returns:
            Tuple of (avg_loss, avg_f1_macro, avg_f1_micro)
        """
        self.model.train()
        total_loss = 0
        total_f1_macro = 0
        total_f1_micro = 0
        batch_count = 0
        
        for batch_idx, batch in enumerate(train_loader):
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            self.optimizer.zero_grad()
            
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            
            # Calculate loss
            loss = self.criterion(logits, labels)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), 
                max_norm=self.config.training.gradient_clip_norm
            )
            
            self.optimizer.step()
            self.scheduler.step()
            
            total_loss += loss.item()
            batch_count += 1
            
            # Calculate F1 scores
            with torch.no_grad():
                f1_macro, f1_micro, _ = self.calculate_f1_score(
                    torch.sigmoid(logits), labels, self.config.evaluation.classification_threshold
                )
                total_f1_macro += f1_macro
                total_f1_micro += f1_micro
            
            # Print progress every 10 batches
            if batch_idx % 10 == 0:
                print(f"    Batch {batch_idx}/{len(train_loader)}, "
                      f"Loss: {loss.item():.4f}, F1: {f1_macro:.3f}")
        
        avg_loss = total_loss / batch_count
        avg_f1_macro = total_f1_macro / batch_count
        avg_f1_micro = total_f1_micro / batch_count
        
        return avg_loss, avg_f1_macro, avg_f1_micro
    
    def validate_epoch(self, val_loader: DataLoader) -> Tuple[float, float, float]:
        """
        Validate for one epoch.
        
        Args:
            val_loader: Validation data loader
        
        Returns:
            Tuple of (avg_loss, avg_f1_macro, avg_f1_micro)
        """
        self.model.eval()
        total_loss = 0
        total_f1_macro = 0
        total_f1_micro = 0
        batch_count = 0
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                
                loss = self.criterion(logits, labels)
                total_loss += loss.item()
                batch_count += 1
                
                # Calculate F1 scores
                f1_macro, f1_micro, _ = self.calculate_f1_score(
                    torch.sigmoid(logits), labels, self.config.evaluation.classification_threshold
                )
                total_f1_macro += f1_macro
                total_f1_micro += f1_micro
        
        avg_loss = total_loss / batch_count
        avg_f1_macro = total_f1_macro / batch_count
        avg_f1_micro = total_f1_micro / batch_count
        
        return avg_loss, avg_f1_macro, avg_f1_micro
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader) -> Dict[str, List[float]]:
        """
        Main training loop.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
        
        Returns:
            Dictionary containing training history
        """
        print("🚀 Starting training...")
        print("=" * 80)
        
        # Setup training components
        self.setup_training(train_loader)
        
        start_time = time.time()
        
        for epoch in range(self.config.training.num_epochs):
            epoch_start_time = time.time()
            
            print(f"\n📈 Epoch {epoch + 1}/{self.config.training.num_epochs}")
            print("-" * 40)
            
            # Training
            print("  🔄 Training...")
            train_loss, train_f1_macro, train_f1_micro = self.train_epoch(train_loader)
            
            # Validation
            print("  🔍 Validating...")
            val_loss, val_f1_macro, val_f1_micro = self.validate_epoch(val_loader)
            
            # Track metrics
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.val_f1_scores.append(val_f1_macro)
            
            # Calculate epoch time
            epoch_time = time.time() - epoch_start_time
            
            # Print epoch results
            print(f"\n  📊 Epoch {epoch + 1} Results:")
            print(f"    Train Loss: {train_loss:.4f}")
            print(f"    Train F1 (Macro): {train_f1_macro:.4f}")
            print(f"    Train F1 (Micro): {train_f1_micro:.4f}")
            print(f"    Val Loss: {val_loss:.4f}")
            print(f"    Val F1 (Macro): {val_f1_macro:.4f}")
            print(f"    Val F1 (Micro): {val_f1_micro:.4f}")
            print(f"    Epoch Time: {epoch_time:.2f}s")
            
            # Save best model
            if val_f1_macro > self.best_val_f1:
                self.best_val_f1 = val_f1_macro
                self.best_model_state = self.model.state_dict().copy()
                print(f"    🎯 New best validation F1: {self.best_val_f1:.4f}")
            
            print("-" * 40)
        
        # Training completion
        total_time = time.time() - start_time
        print(f"\n🎉 Training completed in {total_time:.2f}s!")
        print(f"Best validation F1 score: {self.best_val_f1:.4f}")
        
        # Load best model
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            print("✅ Best model loaded!")
        
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_f1_scores': self.val_f1_scores,
            'best_val_f1': self.best_val_f1
        }
    
    def save_checkpoint(self, epoch: int, additional_data: Optional[Dict[str, Any]] = None):
        """
        Save training checkpoint.
        
        Args:
            epoch: Current epoch
            additional_data: Additional data to save
        """
        if additional_data is None:
            additional_data = {}
            
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_f1_scores': self.val_f1_scores,
            'best_val_f1': self.best_val_f1,
            'config': self.config.to_dict()
        }
        
        checkpoint.update(additional_data)
        
        save_path = f"{self.config.training.model_save_path}_checkpoint_epoch_{epoch}.pt"
        torch.save(checkpoint, save_path)
        print(f"💾 Checkpoint saved to {save_path}")
    
    def load_checkpoint(self, checkpoint_path: str) -> int:
        """
        Load training checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
        
        Returns:
            Epoch to resume from
        """
        print(f"📂 Loading checkpoint from {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.train_losses = checkpoint['train_losses']
        self.val_losses = checkpoint['val_losses']
        self.val_f1_scores = checkpoint['val_f1_scores']
        self.best_val_f1 = checkpoint['best_val_f1']
        
        epoch = checkpoint['epoch']
        print(f"✅ Checkpoint loaded! Resuming from epoch {epoch + 1}")
        
        return epoch + 1


def quick_train(config: Config, model: MultiLabelBERTClassifier, 
                train_loader: DataLoader, val_loader: DataLoader) -> Tuple[MultiLabelBERTClassifier, Dict[str, List[float]]]:
    """
    Quick training function for convenience.
    
    Args:
        config: Configuration object
        model: Model to train
        train_loader: Training data loader
        val_loader: Validation data loader
    
    Returns:
        Tuple of (trained_model, training_history)
    """
    trainer = Trainer(config, model)
    history = trainer.train(train_loader, val_loader)
    return model, history


if __name__ == "__main__":
    # Example usage
    from .config import Config
    from .models import create_model_and_tokenizer
    
    config = Config()
    config.model.num_labels = 10  # Example
    
    model, tokenizer = create_model_and_tokenizer(config)
    trainer = Trainer(config, model)
    print("Trainer initialized successfully!")
