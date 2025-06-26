#!/usr/bin/env python3
"""
Example script demonstrating the new separate files functionality
"""

from social_worker_bert import SocialWorkerBERTPipeline, Config

def example_separate_files():
    """Example using separate train and test files."""
    print("=== Example: Separate Train/Test Files ===")
    
    # Create configuration for separate files mode
    config = Config()
    config.data.train_file = './train.csv'
    config.data.test_file = './test.csv'
    config.data.mode = 'separate_files'
    config.data.val_size = 0.1  # 10% of training data for validation
    
    # Set training parameters
    config.training.num_epochs = 3
    config.training.batch_size = 16
    config.training.learning_rate = 2e-5
    
    # Create pipeline
    pipeline = SocialWorkerBERTPipeline(config=config)
    
    # Run training
    results = pipeline.run('./results_separate_files')
    
    print("Training completed!")
    return results

def example_single_file():
    """Example using single file with automatic splitting."""
    print("=== Example: Single File (Legacy Mode) ===")
    
    # Create configuration for single file mode
    config = Config()
    config.data.data_file = './tri.csv'
    config.data.mode = 'single_file'
    config.data.test_size = 0.1
    config.data.val_size = 0.1
    
    # Set training parameters
    config.training.num_epochs = 3
    config.training.batch_size = 16
    
    # Create pipeline
    pipeline = SocialWorkerBERTPipeline(config=config)
    
    # Run training
    results = pipeline.run('./results_single_file')
    
    print("Training completed!")
    return results

def example_with_config_file():
    """Example using a YAML configuration file."""
    print("=== Example: Using Configuration File ===")
    
    # Load configuration from file
    pipeline = SocialWorkerBERTPipeline(config_path='./example_config.yaml')
    
    # Override some settings programmatically if needed
    pipeline.config.training.num_epochs = 2
    
    # Run training
    results = pipeline.run('./results_from_config')
    
    print("Training completed!")
    return results

def example_prediction():
    """Example of making predictions with a trained model."""
    print("=== Example: Making Predictions ===")
    
    # Create pipeline and load trained model
    pipeline = SocialWorkerBERTPipeline()
    
    # Load a trained model (adjust path as needed)
    model_path = './results_separate_files/social_worker_bert_model.pt'
    try:
        pipeline.load_trained_model(model_path)
        
        # Make predictions
        texts = [
            "How are you feeling today?",
            "Can you tell me more about that situation?",
            "I understand that must be difficult for you.",
            "What would you like to focus on in our session?"
        ]
        
        predictions = pipeline.predict(texts)
        
        for text, skills in zip(texts, predictions):
            print(f"Text: {text}")
            print(f"Predicted skills: {', '.join(skills) if skills else 'None'}")
            print()
            
    except FileNotFoundError:
        print(f"Model file not found at {model_path}")
        print("Please train a model first using one of the training examples above.")

if __name__ == "__main__":
    print("Social Worker BERT - New Features Example")
    print("=" * 50)
    
    # Run examples
    try:
        # Example 1: Separate files (requires train.csv and test.csv)
        # example_separate_files()
        
        # Example 2: Single file (requires tri.csv)
        # example_single_file()
        
        # Example 3: Configuration file (requires example_config.yaml)
        # example_with_config_file()
        
        # Example 4: Prediction (requires trained model)
        example_prediction()
        
    except Exception as e:
        print(f"Error: {e}")
        print("\nNote: Make sure you have the required data files:")
        print("- For separate files mode: train.csv and test.csv")
        print("- For single file mode: tri.csv") 
        print("- For prediction: a trained model file")
