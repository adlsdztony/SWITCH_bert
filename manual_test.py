#!/usr/bin/env python3
"""
Manual Test Script for Social Worker BERT

This script allows you to:
1. Load a well-finetuned model
2. Interactively test predictions on custom input text
3. Adjust classification thresholds
4. View detailed probability scores for all skills

Usage:
    python manual_test.py
"""

import os
import sys
from pathlib import Path

# Add the package to Python path
sys.path.insert(0, str(Path(__file__).parent))

from social_worker_bert import SocialWorkerBERTPipeline


class ManualTester:
    """Interactive tester for Social Worker BERT models."""
    
    def __init__(self):
        self.pipeline = None
        self.current_threshold = 0.5
        self.model_loaded = False
    
    def load_model(self, model_path: str) -> bool:
        """
        Load a trained model for testing.
        
        Args:
            model_path: Path to the trained model file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            print(f"🔄 Loading model from: {model_path}")
            
            # Create pipeline
            self.pipeline = SocialWorkerBERTPipeline()
            
            # Load the trained model
            self.pipeline.load_trained_model(model_path)
            
            print("✅ Model loaded successfully!")
            print(f"📊 Available skills: {', '.join(self.pipeline.processed_data['skill_labels'])}")
            print(f"🎯 Current threshold: {self.current_threshold}")
            
            self.model_loaded = True
            return True
            
        except Exception as e:
            print(f"❌ Error loading model: {str(e)}")
            return False
    
    def predict_text(self, text: str, show_probabilities: bool = True) -> None:
        """
        Make prediction on input text and display results.
        
        Args:
            text: Input text to classify
            show_probabilities: Whether to show detailed confidence levels (always True now)
        """
        if not self.model_loaded:
            print("❌ No model loaded. Please load a model first.")
            return
        
        try:
            print(f"\n{'='*60}")
            print(f"📝 Input: {text}")
            print(f"{'='*60}")
            
            # Make prediction with verbose output
            predicted_skills, probabilities = self.pipeline.predictor.predict_single(
                text, threshold=self.current_threshold, verbose=show_probabilities
            )
            
        except Exception as e:
            print(f"❌ Error making prediction: {str(e)}")
    
    def set_threshold(self, threshold: float) -> None:
        """
        Set the classification threshold.
        
        Args:
            threshold: New threshold value (0.0 to 1.0)
        """
        if 0.0 <= threshold <= 1.0:
            self.current_threshold = threshold
            print(f"🎯 Threshold set to: {threshold}")
        else:
            print("❌ Threshold must be between 0.0 and 1.0")
    
    def show_help(self) -> None:
        """Display help information."""
        print("\n" + "="*60)
        print("🔧 MANUAL TESTER COMMANDS")
        print("="*60)
        print("Commands:")
        print("  help                    - Show this help message")
        print("  load <model_path>       - Load a trained model")
        print("  threshold <value>       - Set classification threshold (0.0-1.0)")
        print("  quit or exit            - Exit the program")
        print("  <any text>              - Predict skills for the text")
        print("\nFeatures:")
        print("  • Confidence levels are always displayed with visual bars")
        print("  • Shows predictions above threshold or 'None' if below")
        print("  • Skills are sorted by confidence (highest first)")
        print("\nExamples:")
        print("  > load ./results/social_worker_bert_model.pt")
        print("  > threshold 0.3")
        print("  > How are you feeling today?")
        print("="*60)
    
    def run_interactive(self) -> None:
        """Run the interactive testing loop."""
        print("🚀 Social Worker BERT - Manual Testing Interface")
        print("Type 'help' for available commands or 'quit' to exit.")
        
        while True:
            try:
                # Get user input
                user_input = input("\n> ").strip()
                
                if not user_input:
                    continue
                
                # Handle commands
                if user_input.lower() in ['quit', 'exit']:
                    print("👋 Goodbye!")
                    break
                
                elif user_input.lower() == 'help':
                    self.show_help()
                
                elif user_input.lower().startswith('load '):
                    model_path = user_input[5:].strip()
                    if not model_path:
                        print("❌ Please provide a model path. Example: load ./results/model.pt")
                        continue
                    
                    if not os.path.exists(model_path):
                        print(f"❌ Model file not found: {model_path}")
                        continue
                    
                    self.load_model(model_path)
                
                elif user_input.lower().startswith('threshold '):
                    try:
                        threshold_str = user_input[10:].strip()
                        threshold = float(threshold_str)
                        self.set_threshold(threshold)
                    except ValueError:
                        print("❌ Invalid threshold value. Please provide a number between 0.0 and 1.0")
                
                else:
                    # Treat as text input for prediction
                    if not self.model_loaded:
                        print("❌ No model loaded. Use 'load <model_path>' to load a model first.")
                        continue
                    
                    self.predict_text(user_input)
            
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            
            except Exception as e:
                print(f"❌ Unexpected error: {str(e)}")


def main():
    """Main entry point."""
    tester = ManualTester()
    
    # Check for command line arguments
    if True:
        model_path = "/home/switch/switch_research/bert_classifier/SWITCH_bert/results/social_worker_bert_model.pt"  # Default model path
        if os.path.exists(model_path):
            print(f"🔄 Loading model from command line argument: {model_path}")
            if tester.load_model(model_path):
                print("✅ Model loaded successfully! You can now start testing.")
            else:
                print("❌ Failed to load model. You can try loading another model using 'load <path>'.")
        else:
            print(f"❌ Model file not found: {model_path}")
            print("You can load a model using 'load <path>' command.")
    else:
        print("💡 Tip: You can provide a model path as argument: python manual_test.py ./results/model.pt")
        print("Or use the 'load <path>' command to load a model.")
    
    # Start interactive mode
    tester.run_interactive()


if __name__ == "__main__":
    main()
