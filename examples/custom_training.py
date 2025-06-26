#!/usr/bin/env python3
"""
Custom Training Example for Social Worker BERT

This example shows how to use individual components for custom workflows.
"""

import sys
from pathlib import Path

# Add the package to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from social_worker_bert import Config, DataProcessor, Trainer, Predictor
from social_worker_bert.models import create_model_and_tokenizer


def main():
    """Run custom training example."""
    print("🔧 Social Worker BERT - Custom Training Example")
    print("=" * 60)
    
    # Step 1: Create and customize configuration
    print("\n📋 Step 1: Configuration")
    config = Config()
    config.data.data_file = './tri.csv'
    config.training.num_epochs = 3
    config.training.batch_size = 16
    config.class_imbalance.focal_loss = True
    config.print_summary()
    
    # Step 2: Process data
    print("\n📊 Step 2: Data Processing")
    data_processor = DataProcessor(config)
    
    # Create tokenizer
    from transformers import BertTokenizer
    tokenizer = BertTokenizer.from_pretrained(config.model.model_name)
    
    # Process data
    processed_data = data_processor.process_all(tokenizer)
    
    # Step 3: Create model
    print("\n🤖 Step 3: Model Creation")
    config.model.num_labels = len(processed_data['skill_labels'])
    model, tokenizer = create_model_and_tokenizer(config)
    
    # Step 4: Custom training
    print("\n🎯 Step 4: Training")
    trainer = Trainer(config, model)
    
    # You can add custom callbacks or modify training here
    training_history = trainer.train(
        processed_data['loaders']['train'],
        processed_data['loaders']['val']
    )
    
    # Step 5: Custom evaluation
    print("\n📈 Step 5: Evaluation")
    predictor = Predictor(config, model, tokenizer, processed_data['skill_labels'])
    
    # Evaluate on test set
    test_results = predictor.evaluate_model(processed_data['loaders']['test'])
    predictor.print_evaluation_report(test_results['metrics'])
    
    # Step 6: Save results
    print("\n💾 Step 6: Saving Results")
    predictor.save_predictions(
        processed_data['splits']['X_test'],
        test_results['predictions'],
        test_results['probabilities'],
        'custom_test_predictions.csv',
        test_results['labels']
    )
    
    print("\n✅ Custom training example completed!")
    print(f"Best validation F1: {training_history['best_val_f1']:.4f}")
    print(f"Test F1 (macro): {test_results['metrics']['f1_macro']:.4f}")


if __name__ == "__main__":
    main()
