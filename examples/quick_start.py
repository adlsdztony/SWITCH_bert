#!/usr/bin/env python3
"""
Quick Start Example for Social Worker BERT

This example shows the simplest way to train and evaluate a model.
"""

import sys
from pathlib import Path

# Add the package to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from social_worker_bert import SocialWorkerBERTPipeline


def main():
    """Run quick start example."""
    print("🚀 Social Worker BERT - Quick Start Example")
    print("=" * 50)
    
    # Create pipeline with default configuration
    pipeline = SocialWorkerBERTPipeline(config_path='./example_config.yaml')
    
    
    # Run the complete pipeline
    results = pipeline.run(output_dir='./results')
    
    print("\n✅ Quick start example completed!")
    print(f"Best validation F1: {results['training_history']['best_val_f1']:.4f}")
    print(f"Test F1 (macro): {results['evaluation_results']['metrics']['f1_macro']:.4f}")
    
    # Example prediction on new text
    print("\n🔍 Example prediction:")
    example_text = "How are you feeling today? Can you tell me more about what's been troubling you?"
    predictions = pipeline.predict([example_text])
    print(f"Input: {example_text}")
    print(f"Predicted skills: {', '.join(predictions[0]) if predictions[0] else 'None'}")


if __name__ == "__main__":
    main()
