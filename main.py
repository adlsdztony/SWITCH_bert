#!/usr/bin/env python3
"""
Main entry point for Social Worker BERT

This script provides command-line interface for running different pipeline components.

Usage:
    python main.py train --config config.yaml --output results/
    python main.py predict --model model.pt --text "Hello, how are you feeling today?"
    python main.py evaluate --model model.pt --data test_data.csv
"""

import argparse
import sys
import os
from pathlib import Path

# Add the package to Python path
sys.path.insert(0, str(Path(__file__).parent))

from social_worker_bert import SocialWorkerBERTPipeline
from social_worker_bert.config import create_default_config_file


def train_command(args):
    """Execute training command."""
    print("🚀 Starting training pipeline...")
    
    # Load configuration
    if args.config:
        pipeline = SocialWorkerBERTPipeline(config_path=args.config)
    else:
        pipeline = SocialWorkerBERTPipeline()
    
    # Override config with command line arguments
    if args.data:
        pipeline.config.data.data_file = args.data
        pipeline.config.data.mode = 'single_file'
    elif args.train_file and args.test_file:
        pipeline.config.data.train_file = args.train_file
        pipeline.config.data.test_file = args.test_file
        pipeline.config.data.mode = 'separate_files'
    elif args.train_file or args.test_file:
        raise ValueError("Both --train-file and --test-file must be provided together")
    
    if args.epochs:
        pipeline.config.training.num_epochs = args.epochs
    if args.batch_size:
        pipeline.config.training.batch_size = args.batch_size
    if args.learning_rate:
        pipeline.config.training.learning_rate = args.learning_rate
    
    # Run pipeline
    results = pipeline.run(args.output)
    
    print("✅ Training completed successfully!")
    return results


def predict_command(args):
    """Execute prediction command."""
    print("🔍 Making predictions...")
    
    if not args.model:
        raise ValueError("Model path is required for prediction")
    
    # Create pipeline and load model
    pipeline = SocialWorkerBERTPipeline()
    pipeline.load_trained_model(args.model)
    
    if args.text:
        # Single text prediction
        texts = [args.text]
        predictions = pipeline.predict(texts, threshold=args.threshold)
        
        print(f"\nInput: {args.text}")
        print(f"Predicted skills: {', '.join(predictions[0]) if predictions[0] else 'None'}")
    
    elif args.file:
        # Batch prediction from file
        import pandas as pd
        
        if args.file.endswith('.csv'):
            df = pd.read_csv(args.file)
            if 'message' in df.columns:
                texts = df['message'].tolist()
            elif 'text' in df.columns:
                texts = df['text'].tolist()
            else:
                raise ValueError("CSV file must contain 'message' or 'text' column")
        else:
            # Plain text file
            with open(args.file, 'r') as f:
                texts = [line.strip() for line in f if line.strip()]
        
        predictions = pipeline.predict(texts, threshold=args.threshold)
        
        # Save results
        results_df = pd.DataFrame({
            'text': texts,
            'predicted_skills': ['|'.join(pred) if pred else 'None' for pred in predictions]
        })
        
        output_file = args.output or 'predictions.csv'
        results_df.to_csv(output_file, index=False)
        print(f"Predictions saved to {output_file}")
    
    else:
        raise ValueError("Either --text or --file must be provided for prediction")


def evaluate_command(args):
    """Execute evaluation command."""
    print("📊 Evaluating model...")
    
    if not args.model:
        raise ValueError("Model path is required for evaluation")
    
    # Create pipeline and load model
    pipeline = SocialWorkerBERTPipeline()
    pipeline.load_trained_model(args.model)
    
    if args.data:
        # Load test data and evaluate
        pipeline.config.data.data_file = args.data
        pipeline.process_data()
        pipeline.evaluate_model()
        
        # Save detailed results if requested
        if args.output:
            os.makedirs(args.output, exist_ok=True)
            pipeline.save_results(args.output)
    else:
        print("Warning: No test data provided. Please use --data to specify test file.")


def create_config_command(args):
    """Create default configuration file."""
    print("📄 Creating default configuration file...")
    create_default_config_file(args.output or "config.yaml")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Social Worker BERT - Multi-label Classification for Social Worker Skills",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train with single file (legacy mode)
  python main.py train --data ./tri.csv --output ./results

  # Train with separate train/test files
  python main.py train --train-file ./train.csv --test-file ./test.csv --output ./results

  # Train with custom configuration
  python main.py train --config config.yaml --output ./results

  # Train with command line overrides
  python main.py train --data ./tri.csv --epochs 10 --batch-size 32

  # Predict single text
  python main.py predict --model model.pt --text "How are you feeling today?"

  # Predict batch from file
  python main.py predict --model model.pt --file texts.csv --output predictions.csv

  # Evaluate model
  python main.py evaluate --model model.pt --data test_data.csv --output eval_results/

  # Create default config file
  python main.py create-config --output my_config.yaml
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train a new model')
    train_parser.add_argument('--config', '-c', type=str, help='Path to configuration YAML file')
    
    # Data input options (mutually exclusive groups)
    data_group = train_parser.add_mutually_exclusive_group()
    data_group.add_argument('--data', '-d', type=str, help='Path to single training data CSV file (will be split internally)')
    
    train_parser.add_argument('--train-file', type=str, help='Path to training data CSV file (use with --test-file)')
    train_parser.add_argument('--test-file', type=str, help='Path to test data CSV file (use with --train-file)')
    
    train_parser.add_argument('--output', '-o', type=str, default='./results', 
                             help='Output directory for results')
    train_parser.add_argument('--epochs', type=int, help='Number of training epochs')
    train_parser.add_argument('--batch-size', type=int, help='Training batch size')
    train_parser.add_argument('--learning-rate', type=float, help='Learning rate')
    
    # Predict command
    predict_parser = subparsers.add_parser('predict', help='Make predictions with trained model')
    predict_parser.add_argument('--model', '-m', type=str, required=True, 
                               help='Path to trained model file')
    predict_parser.add_argument('--text', '-t', type=str, help='Single text to predict')
    predict_parser.add_argument('--file', '-f', type=str, help='File containing texts to predict')
    predict_parser.add_argument('--output', '-o', type=str, help='Output file for predictions')
    predict_parser.add_argument('--threshold', type=float, default=0.5, 
                               help='Classification threshold')
    
    # Evaluate command
    evaluate_parser = subparsers.add_parser('evaluate', help='Evaluate trained model')
    evaluate_parser.add_argument('--model', '-m', type=str, required=True, 
                                help='Path to trained model file')
    evaluate_parser.add_argument('--data', '-d', type=str, help='Path to test data CSV file')
    evaluate_parser.add_argument('--output', '-o', type=str, help='Output directory for results')
    
    # Create config command
    config_parser = subparsers.add_parser('create-config', help='Create default configuration file')
    config_parser.add_argument('--output', '-o', type=str, default='config.yaml',
                              help='Output path for configuration file')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    try:
        if args.command == 'train':
            train_command(args)
        elif args.command == 'predict':
            predict_command(args)
        elif args.command == 'evaluate':
            evaluate_command(args)
        elif args.command == 'create-config':
            create_config_command(args)
        else:
            parser.print_help()
    
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
