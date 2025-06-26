"""
Inference Module for Social Worker BERT

This module handles:
- Model prediction on new data
- Batch inference
- Evaluation metrics calculation
- Results formatting and export
"""

import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, precision_score, recall_score
from typing import List, Tuple, Dict, Any, Optional
from transformers import BertTokenizer

from .config import Config
from .models import MultiLabelBERTClassifier


class Predictor:
    """Main predictor class for social worker skill classification."""
    
    def __init__(self, config: Config, model: MultiLabelBERTClassifier, 
                 tokenizer: BertTokenizer, skill_labels: List[str]):
        """
        Initialize predictor.
        
        Args:
            config: Configuration object
            model: Trained model
            tokenizer: BERT tokenizer
            skill_labels: List of skill label names
        """
        self.config = config
        self.model = model
        self.tokenizer = tokenizer
        self.skill_labels = skill_labels
        self.device = config.device
        
        # Set model to evaluation mode
        self.model.eval()
    
    def predict_single(self, text: str, threshold: Optional[float] = None, verbose: bool = True) -> Tuple[List[str], Dict[str, float]]:
        """
        Predict skills for a single text.
        
        Args:
            text: Input text to classify
            threshold: Classification threshold (uses config default if None)
            verbose: Whether to print detailed output
        
        Returns:
            Tuple of (predicted_skills, all_probabilities)
        """
        if threshold is None:
            threshold = self.config.evaluation.classification_threshold
        
        # Tokenize
        encoding = self.tokenizer(
            str(text),
            truncation=True,
            padding='max_length',
            max_length=self.config.model.max_length,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        
        # Predict
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probabilities = torch.sigmoid(logits).cpu().numpy()[0]
        
        # Create probability dictionary
        prob_dict = {skill: float(prob) for skill, prob in zip(self.skill_labels, probabilities)}
        
        if verbose:
            # Print confidence levels in a clear format
            print(f"\n📊 CONFIDENCE LEVELS (Threshold: {threshold:.2f})")
            print("=" * 60)
            
            # Sort by probability (highest first)
            sorted_probs = sorted(prob_dict.items(), key=lambda x: x[1], reverse=True)
            
            for skill, prob in sorted_probs:
                status = "✓" if prob > threshold else " "
                confidence_bar = "█" * int(prob * 20)  # Visual bar (max 20 chars)
                print(f"{status} {skill:<30} {prob:.4f} |{confidence_bar:<20}|")
            
            print("=" * 60)
        
        # Get predicted skills
        predicted_indices = np.where(probabilities > threshold)[0]
        predicted_skills = [self.skill_labels[i] for i in predicted_indices]
        
        if verbose:
            # Print final predictions
            if predicted_skills:
                print("🎯 PREDICTIONS ABOVE THRESHOLD:")
                for skill in predicted_skills:
                    print(f"   • {skill} ({prob_dict[skill]:.4f})")
            else:
                print("🎯 PREDICTIONS ABOVE THRESHOLD: None")
            print()
        
        return predicted_skills, prob_dict
    
    def predict_batch(self, texts: List[str], threshold: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict skills for a batch of texts.
        
        Args:
            texts: List of input texts
            threshold: Classification threshold
        
        Returns:
            Tuple of (predictions, probabilities)
        """
        if threshold is None:
            threshold = self.config.evaluation.classification_threshold
        
        predictions = []
        probabilities = []
        
        with torch.no_grad():
            for text in texts:
                # Tokenize
                encoding = self.tokenizer(
                    str(text),
                    truncation=True,
                    padding='max_length',
                    max_length=self.config.model.max_length,
                    return_tensors='pt'
                )
                
                input_ids = encoding['input_ids'].to(self.device)
                attention_mask = encoding['attention_mask'].to(self.device)
                
                # Predict
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                probs = torch.sigmoid(logits).cpu().numpy()[0]
                
                # Apply threshold
                preds = (probs > threshold).astype(int)
                
                predictions.append(preds)
                probabilities.append(probs)
        
        return np.array(predictions), np.array(probabilities)
    
    def evaluate_model(self, data_loader: DataLoader, threshold: Optional[float] = None) -> Dict[str, Any]:
        """
        Comprehensive model evaluation on a dataset.
        
        Args:
            data_loader: DataLoader containing evaluation data
            threshold: Classification threshold
        
        Returns:
            Dictionary containing evaluation metrics
        """
        if threshold is None:
            threshold = self.config.evaluation.classification_threshold
        
        self.model.eval()
        all_predictions = []
        all_labels = []
        all_probabilities = []
        
        with torch.no_grad():
            for batch in data_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                probabilities = torch.sigmoid(logits)
                
                all_probabilities.extend(probabilities.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
                # Apply threshold
                predictions = (probabilities > threshold).int()
                all_predictions.extend(predictions.cpu().numpy())
        
        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)
        all_probabilities = np.array(all_probabilities)
        
        # Calculate metrics
        metrics = self.calculate_metrics(all_predictions, all_labels, all_probabilities)
        
        return {
            'metrics': metrics,
            'predictions': all_predictions,
            'labels': all_labels,
            'probabilities': all_probabilities
        }
    
    def calculate_metrics(self, predictions: np.ndarray, labels: np.ndarray, 
                         probabilities: np.ndarray) -> Dict[str, Any]:
        """
        Calculate comprehensive evaluation metrics.
        
        Args:
            predictions: Binary predictions
            labels: Ground truth labels
            probabilities: Prediction probabilities
        
        Returns:
            Dictionary containing various metrics
        """
        # Overall metrics
        f1_macro = f1_score(labels, predictions, average='macro', zero_division=0)
        f1_micro = f1_score(labels, predictions, average='micro', zero_division=0)
        f1_weighted = f1_score(labels, predictions, average='weighted', zero_division=0)
        
        precision_macro = precision_score(labels, predictions, average='macro', zero_division=0)
        precision_micro = precision_score(labels, predictions, average='micro', zero_division=0)
        
        recall_macro = recall_score(labels, predictions, average='macro', zero_division=0)
        recall_micro = recall_score(labels, predictions, average='micro', zero_division=0)
        
        # Per-class metrics
        f1_per_class = f1_score(labels, predictions, average=None, zero_division=0)
        precision_per_class = precision_score(labels, predictions, average=None, zero_division=0)
        recall_per_class = recall_score(labels, predictions, average=None, zero_division=0)
        
        # Additional metrics
        sample_accuracy = np.mean(np.all(predictions == labels, axis=1))
        hamming_loss = np.mean(predictions != labels)
        
        # Support (number of true instances for each class)
        support = labels.sum(axis=0)
        
        return {
            'f1_macro': f1_macro,
            'f1_micro': f1_micro,
            'f1_weighted': f1_weighted,
            'precision_macro': precision_macro,
            'precision_micro': precision_micro,
            'recall_macro': recall_macro,
            'recall_micro': recall_micro,
            'sample_accuracy': sample_accuracy,
            'hamming_loss': hamming_loss,
            'per_class': {
                'f1': f1_per_class.tolist(),
                'precision': precision_per_class.tolist(),
                'recall': recall_per_class.tolist(),
                'support': support.tolist(),
                'labels': self.skill_labels
            }
        }
    
    def create_detailed_prediction_table(self, texts: List[str], predictions: np.ndarray, 
                                       probabilities: np.ndarray, true_labels: Optional[np.ndarray] = None,
                                       threshold: Optional[float] = None) -> pd.DataFrame:
        """
        Create a detailed table with all predictions and probabilities.
        
        Args:
            texts: Input texts
            predictions: Binary predictions
            probabilities: Prediction probabilities
            true_labels: Ground truth labels (optional)
            threshold: Classification threshold
        
        Returns:
            Detailed DataFrame with predictions
        """
        if threshold is None:
            threshold = self.config.evaluation.classification_threshold
        
        detailed_results = []
        
        for i, text in enumerate(texts):
            # Basic info
            result = {
                'sample_id': i + 1,
                'text': text,
                'text_preview': text[:100] + '...' if len(text) > 100 else text,
            }
            
            # Predicted labels (above threshold)
            predicted_skills = [self.skill_labels[j] for j in range(len(self.skill_labels)) 
                              if predictions[i][j] == 1]
            result['predicted_labels'] = '|'.join(predicted_skills) if predicted_skills else 'None'
            result['num_predicted_labels'] = len(predicted_skills)
            
            # True labels if provided
            if true_labels is not None:
                true_skills = [self.skill_labels[j] for j in range(len(self.skill_labels)) 
                             if true_labels[i][j] == 1]
                result['true_labels'] = '|'.join(true_skills) if true_skills else 'None'
                result['num_true_labels'] = len(true_skills)
            
            # All probabilities
            for j, skill in enumerate(self.skill_labels):
                result[f'{skill}'] = probabilities[i][j]
            
            # Top 3 most likely skills
            top_3_indices = np.argsort(probabilities[i])[-3:][::-1]
            top_3_skills = [(self.skill_labels[idx], probabilities[i][idx]) for idx in top_3_indices]
            result['top_3_skills'] = ', '.join([f"{skill}({prob:.3f})" for skill, prob in top_3_skills])
            
            # Confidence metrics
            result['max_probability'] = probabilities[i].max()
            result['avg_probability'] = probabilities[i].mean()
            result['confidence_spread'] = probabilities[i].max() - probabilities[i].min()
            
            detailed_results.append(result)
        
        return pd.DataFrame(detailed_results)
    
    def print_evaluation_report(self, metrics: Dict[str, Any]):
        """
        Print a comprehensive evaluation report.
        
        Args:
            metrics: Metrics dictionary from calculate_metrics
        """
        print("\n📊 Evaluation Report")
        print("=" * 60)
        
        print("\n🎯 Overall Metrics:")
        print(f"  F1 Score (Macro): {metrics['f1_macro']:.4f}")
        print(f"  F1 Score (Micro): {metrics['f1_micro']:.4f}")
        print(f"  F1 Score (Weighted): {metrics['f1_weighted']:.4f}")
        print(f"  Precision (Macro): {metrics['precision_macro']:.4f}")
        print(f"  Precision (Micro): {metrics['precision_micro']:.4f}")
        print(f"  Recall (Macro): {metrics['recall_macro']:.4f}")
        print(f"  Recall (Micro): {metrics['recall_micro']:.4f}")
        print(f"  Sample Accuracy: {metrics['sample_accuracy']:.4f}")
        print(f"  Hamming Loss: {metrics['hamming_loss']:.4f}")
        
        print("\n🏷️  Per-Class Metrics:")
        per_class = metrics['per_class']
        for i, label in enumerate(per_class['labels']):
            print(f"  {label}:")
            print(f"    F1: {per_class['f1'][i]:.3f}, "
                  f"Precision: {per_class['precision'][i]:.3f}, "
                  f"Recall: {per_class['recall'][i]:.3f}, "
                  f"Support: {per_class['support'][i]}")
        
        print("=" * 60)
    
    def save_predictions(self, texts: List[str], predictions: np.ndarray, 
                        probabilities: np.ndarray, save_path: str,
                        true_labels: Optional[np.ndarray] = None):
        """
        Save detailed predictions to CSV file.
        
        Args:
            texts: Input texts
            predictions: Binary predictions
            probabilities: Prediction probabilities
            save_path: Path to save the CSV file
            true_labels: Ground truth labels (optional)
        """
        detailed_table = self.create_detailed_prediction_table(
            texts, predictions, probabilities, true_labels
        )
        
        detailed_table.to_csv(save_path, index=False)
        print(f"💾 Detailed predictions saved to {save_path}")


def quick_predict(config: Config, model: MultiLabelBERTClassifier, 
                  tokenizer: BertTokenizer, skill_labels: List[str],
                  texts: List[str]) -> Tuple[List[List[str]], List[Dict[str, float]]]:
    """
    Quick prediction function for convenience.
    
    Args:
        config: Configuration object
        model: Trained model
        tokenizer: BERT tokenizer
        skill_labels: List of skill labels
        texts: Texts to predict
    
    Returns:
        Tuple of (predicted_skills_list, probabilities_list)
    """
    predictor = Predictor(config, model, tokenizer, skill_labels)
    
    results = []
    probs = []
    
    for text in texts:
        pred_skills, prob_dict = predictor.predict_single(text)
        results.append(pred_skills)
        probs.append(prob_dict)
    
    return results, probs


if __name__ == "__main__":
    # Example usage
    print("Inference module loaded successfully!")
