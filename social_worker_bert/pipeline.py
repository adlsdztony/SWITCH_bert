"""
Pipeline Module for Social Worker BERT

This module provides a unified pipeline that orchestrates all components:
- Data processing
- Model creation and training
- Evaluation and inference
- Result export

The pipeline can be run end-to-end or individual components can be used separately.
"""

import os
import time
from typing import Dict, Any, Optional, List

from .config import Config
from .data import DataProcessor
from .models import create_model_and_tokenizer, save_model, load_model
from .training import Trainer
from .inference import Predictor


class SocialWorkerBERTPipeline:
    """
    Main pipeline class that orchestrates the entire workflow.
    
    This class provides both a complete end-to-end pipeline and access to
    individual components for custom workflows.
    """
    
    def __init__(self, config_path: Optional[str] = None, config: Optional[Config] = None):
        """
        Initialize the pipeline.
        
        Args:
            config_path: Path to configuration YAML file
            config: Configuration object (alternative to config_path)
        """
        if config is not None:
            self.config = config
        elif config_path is not None:
            self.config = Config(config_path)
        else:
            self.config = Config()  # Use defaults
        
        # Pipeline state
        self.data_processor = None
        self.model = None
        self.tokenizer = None
        self.trainer = None
        self.predictor = None
        
        # Data storage
        self.processed_data = None
        self.training_history = None
        self.evaluation_results = None
        
        print("🚀 Social Worker BERT Pipeline initialized!")
        self.config.print_summary()
    
    def process_data(self) -> Dict[str, Any]:
        """
        Process the data using the data processor.
        
        Returns:
            Dictionary containing processed data
        """
        print("\n" + "="*60)
        print("📊 STEP 1: DATA PROCESSING")
        print("="*60)
        
        self.data_processor = DataProcessor(self.config)
        
        # Create tokenizer for data processing
        from transformers import BertTokenizer
        temp_tokenizer = BertTokenizer.from_pretrained(self.config.model.model_name)
        
        # Process all data
        self.processed_data = self.data_processor.process_all(temp_tokenizer)
        
        print("\n✅ Data processing completed!")
        return self.processed_data
    
    def create_model(self) -> tuple:
        """
        Create the model and tokenizer.
        
        Returns:
            Tuple of (model, tokenizer)
        """
        print("\n" + "="*60)
        print("🤖 STEP 2: MODEL CREATION")
        print("="*60)
        
        if self.processed_data is None:
            raise ValueError("Data must be processed before creating model. Run process_data() first.")
        
        # Update config with number of labels
        self.config.model.num_labels = len(self.processed_data['skill_labels'])
        
        # Create model and tokenizer
        self.model, self.tokenizer = create_model_and_tokenizer(self.config)
        
        print("\n✅ Model creation completed!")
        return self.model, self.tokenizer
    
    def train_model(self) -> Dict[str, List[float]]:
        """
        Train the model.
        
        Returns:
            Training history dictionary
        """
        print("\n" + "="*60)
        print("🎯 STEP 3: MODEL TRAINING")
        print("="*60)
        
        if self.model is None:
            raise ValueError("Model must be created before training. Run create_model() first.")
        
        if self.processed_data is None:
            raise ValueError("Data must be processed before training. Run process_data() first.")
        
        # Create trainer
        self.trainer = Trainer(self.config, self.model)
        
        # Train the model
        self.training_history = self.trainer.train(
            self.processed_data['loaders']['train'],
            self.processed_data['loaders']['val']
        )
        
        print("\n✅ Model training completed!")
        return self.training_history
    
    def evaluate_model(self) -> Dict[str, Any]:
        """
        Evaluate the trained model.
        
        Returns:
            Evaluation results dictionary
        """
        print("\n" + "="*60)
        print("📈 STEP 4: MODEL EVALUATION")
        print("="*60)
        
        if self.model is None:
            raise ValueError("Model must be created before evaluation. Run create_model() first.")
        
        if self.processed_data is None:
            raise ValueError("Data must be processed before evaluation. Run process_data() first.")
        
        # Create predictor
        self.predictor = Predictor(
            self.config, 
            self.model, 
            self.tokenizer, 
            self.processed_data['skill_labels']
        )
        
        # Evaluate on test set
        test_results = self.predictor.evaluate_model(
            self.processed_data['loaders']['test']
        )
        
        # Print evaluation report
        self.predictor.print_evaluation_report(test_results['metrics'])
        
        # Store results
        self.evaluation_results = test_results
        
        print("\n✅ Model evaluation completed!")
        return test_results
    
    def save_results(self, output_dir: str = "./results") -> None:
        """
        Save all results and model artifacts.
        
        Args:
            output_dir: Directory to save results
        """
        print("\n" + "="*60)
        print("💾 STEP 5: SAVING RESULTS")
        print("="*60)
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Save model if training was completed
        if self.config.training.save_model and self.model is not None:
            model_path = os.path.join(output_dir, f"{self.config.training.model_save_path}.pt")
            save_model(
                self.model,
                self.tokenizer,
                self.config,
                {
                    'skill_labels': self.processed_data['skill_labels'],
                    'mlb': self.processed_data['mlb'],
                    'training_history': self.training_history
                },
                model_path
            )
        
        # Save detailed predictions if evaluation was completed
        if self.evaluation_results is not None and self.config.evaluation.save_detailed_predictions:
            # Test set predictions
            test_predictions_path = os.path.join(output_dir, "test_detailed_predictions.csv")
            self.predictor.save_predictions(
                self.processed_data['splits']['X_test'],
                self.evaluation_results['predictions'],
                self.evaluation_results['probabilities'],
                test_predictions_path,
                self.evaluation_results['labels']
            )
            
            # Training set predictions for analysis
            train_results = self.predictor.evaluate_model(
                self.processed_data['loaders']['train']
            )
            train_predictions_path = os.path.join(output_dir, "train_detailed_predictions.csv")
            self.predictor.save_predictions(
                self.processed_data['splits']['X_train'],
                train_results['predictions'],
                train_results['probabilities'],
                train_predictions_path,
                train_results['labels']
            )

            # Validation set predictions if available
            if 'val' in self.processed_data['loaders']:
                val_results = self.predictor.evaluate_model(
                    self.processed_data['loaders']['val']
                )
                val_predictions_path = os.path.join(output_dir, "val_detailed_predictions.csv")
                self.predictor.save_predictions(
                    self.processed_data['splits']['X_val'],
                    val_results['predictions'],
                    val_results['probabilities'],
                    val_predictions_path,
                    val_results['labels']
                )
        
        # Save configuration
        config_path = os.path.join(output_dir, "config.yaml")
        self.config.save_to_file(config_path)
        
        print(f"\n✅ Results saved to {output_dir}")
    
    def run(self, output_dir: str = "./results") -> Dict[str, Any]:
        """
        Run the complete pipeline end-to-end.
        
        Args:
            output_dir: Directory to save results
        
        Returns:
            Dictionary containing all results
        """
        start_time = time.time()
        
        print("\n🚀 STARTING SOCIAL WORKER BERT PIPELINE")
        print("="*80)
        
        # Step 1: Process data
        processed_data = self.process_data()
        
        # Step 2: Create model
        model, tokenizer = self.create_model()
        
        # Step 3: Train model
        training_history = self.train_model()
        
        # Step 4: Evaluate model
        evaluation_results = self.evaluate_model()
        
        # Step 5: Save results
        self.save_results(output_dir)
        
        # Pipeline completion
        total_time = time.time() - start_time
        print("\n" + "="*80)
        print("🎉 PIPELINE COMPLETED SUCCESSFULLY!")
        print(f"⏱️  Total time: {total_time:.2f} seconds")
        print(f"🎯 Best validation F1: {self.training_history['best_val_f1']:.4f}")
        print(f"📊 Test F1 (macro): {evaluation_results['metrics']['f1_macro']:.4f}")
        print(f"💾 Results saved to: {output_dir}")
        print("="*80)
        
        return {
            'processed_data': processed_data,
            'model': model,
            'tokenizer': tokenizer,
            'training_history': training_history,
            'evaluation_results': evaluation_results,
            'config': self.config
        }
    
    def predict(self, texts: List[str], threshold: Optional[float] = None) -> List[List[str]]:
        """
        Make predictions on new texts.
        
        Args:
            texts: List of texts to predict
            threshold: Classification threshold
        
        Returns:
            List of predicted skill lists for each text
        """
        if self.predictor is None:
            if self.model is None or self.tokenizer is None:
                raise ValueError("Model must be trained or loaded before making predictions.")
            
            if self.processed_data is None:
                raise ValueError("Data must be processed to get skill labels.")
            
            self.predictor = Predictor(
                self.config,
                self.model,
                self.tokenizer,
                self.processed_data['skill_labels']
            )
        
        results = []
        for text in texts:
            predicted_skills, _ = self.predictor.predict_single(text, threshold)
            results.append(predicted_skills)
        
        return results
    
    def load_trained_model(self, model_path: str) -> None:
        """
        Load a previously trained model.
        
        Args:
            model_path: Path to saved model
        """
        print(f"📂 Loading trained model from {model_path}")
        
        self.model, self.tokenizer, additional_data = load_model(model_path, self.config.device)
        
        # Update processed data with saved information
        if self.processed_data is None:
            self.processed_data = {}
        
        self.processed_data.update({
            'skill_labels': additional_data.get('skill_labels', []),
            'mlb': additional_data.get('mlb', None)
        })
        
        # Update config
        if 'config' in additional_data:
            saved_config = additional_data['config']
            self.config.update_from_dict(saved_config)
        
        # Create predictor with loaded model
        self.predictor = Predictor(
            self.config,
            self.model,
            self.tokenizer,
            self.processed_data['skill_labels']
        )
        
        print("✅ Model loaded successfully!")
        print(f"📊 Available skills: {len(self.processed_data['skill_labels'])} skills loaded")


def create_default_pipeline(data_file: str = "./tri.csv") -> SocialWorkerBERTPipeline:
    """
    Create a pipeline with default configuration.
    
    Args:
        data_file: Path to data file
    
    Returns:
        Configured pipeline
    """
    config = Config()
    config.data.data_file = data_file
    return SocialWorkerBERTPipeline(config=config)


def run_training_pipeline(config_path: str, output_dir: str = "./results") -> Dict[str, Any]:
    """
    Convenience function to run complete training pipeline.
    
    Args:
        config_path: Path to configuration file
        output_dir: Output directory for results
    
    Returns:
        Pipeline results
    """
    pipeline = SocialWorkerBERTPipeline(config_path)
    return pipeline.run(output_dir)


if __name__ == "__main__":
    # Example usage
    print("Pipeline module loaded successfully!")
    
    # Create a default pipeline
    pipeline = create_default_pipeline()
    print("Default pipeline created!")
