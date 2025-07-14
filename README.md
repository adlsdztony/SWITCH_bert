# Social Worker BERT: Multi-label Classification Library

A comprehensive Python library for training and deploying BERT-based multi-label classification models specifically designed for social worker skill detection.

## 🌟 Features

- **Modular Design**: Use individual components or complete pipeline
- **Multi-label Classification**: Detect multiple social worker skills simultaneously
- **BERT-based Architecture**: Leverages pre-trained BERT models
- **Class Imbalance Handling**: Focal loss and other techniques for imbalanced datasets
- **Data Augmentation**: Automatic text augmentation for rare classes using synonym replacement and paraphrasing
- **Flexible Configuration**: YAML-based configuration with command-line overrides
- **Comprehensive Evaluation**: Detailed metrics and per-class analysis
- **Easy Deployment**: Save/load trained models for inference
- **Command-line Interface**: Train, predict, and evaluate from command line

## 📦 Installation

### Requirements

- Python 3.8+
- PyTorch 1.9+
- Transformers 4.20+
- scikit-learn 1.0+
- pandas 1.3+
- numpy 1.21+

### Setup

1. Clone the repository:
```bash
git clone <repository_url>
cd SWITCH_bert
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Install the package in development mode:
```bash
pip install -e .
```

## 🚀 Quick Start

### Option 1: Complete Pipeline (Easiest)

```python
from social_worker_bert import SocialWorkerBERTPipeline

# Create pipeline with default configuration
pipeline = SocialWorkerBERTPipeline()

# Set your data file
pipeline.config.data.data_file = './tri.csv'

# Run complete pipeline: data processing → training → evaluation
results = pipeline.run(output_dir='./results')

# Make predictions on new text
predictions = pipeline.predict(["How are you feeling today?"])
print(f"Predicted skills: {predictions[0]}")
```

### Option 2: Command Line Interface

**Single file mode (legacy):**
```bash
# Train a model with automatic splitting
python main.py train --data ./tri.csv --output ./results --epochs 5
```

**Separate files mode (recommended):**
```bash
# Train a model with separate train/test files
python main.py train --train-file ./train.csv --test-file ./test.csv --output ./results --epochs 5
```

**Common operations:**
```bash
# Make predictions
python main.py predict --model ./results/social_worker_bert_model.pt --text "How are you feeling today?"

# Evaluate model
python main.py evaluate --model ./results/social_worker_bert_model.pt --data ./test_data.csv

# Create default configuration file
python main.py create-config --output my_config.yaml
```

### Option 3: Custom Components

```python
from social_worker_bert import Config, DataProcessor, Trainer, Predictor
from social_worker_bert.models import create_model_and_tokenizer

# 1. Configure
config = Config()
config.data.data_file = './tri.csv'
config.training.num_epochs = 5

# 2. Process data
data_processor = DataProcessor(config)
processed_data = data_processor.process_all(tokenizer)

# 3. Create model
model, tokenizer = create_model_and_tokenizer(config)

# 4. Train
trainer = Trainer(config, model)
history = trainer.train(train_loader, val_loader)

# 5. Evaluate
predictor = Predictor(config, model, tokenizer, skill_labels)
results = predictor.evaluate_model(test_loader)
```

## 📊 Data Format

Your CSV file should contain:
- `message`: Text messages from social workers
- Annotator columns (e.g., `Anthea`, `Karen`, `Kimmy`): Comma-separated skill labels

Example:
```csv
message,Anthea,Karen,Kimmy
"How are you feeling today?","Active Listening,Empathy","Active Listening","Empathy,Open-Ended Questions"
"Can you tell me more about that?","Open-Ended Questions,Clarifying","Open-Ended Questions","Active Listening,Open-Ended Questions"
```

## 📊 Data Input Modes

The library now supports two data input modes:

### 1. Single File Mode (Legacy)
Use a single CSV file that will be automatically split into train/validation/test sets:

**Command Line:**
```bash
python main.py train --data ./my_data.csv --output ./results
```

**Configuration:**
```yaml
data:
  data_file: './my_data.csv'
  mode: 'single_file'
  test_size: 0.1
  val_size: 0.1
```

### 2. Separate Files Mode (New)
Use separate CSV files for training and testing:

**Command Line:**
```bash
python main.py train --train-file ./train.csv --test-file ./test.csv --output ./results
```

**Configuration:**
```yaml
data:
  train_file: './train.csv'
  test_file: './test.csv'
  mode: 'separate_files' 
  val_size: 0.1  # Validation split from training data
```

**Benefits of Separate Files Mode:**
- Better control over train/test distribution
- Consistent evaluation across experiments
- Prevents data leakage
- Matches standard ML practices

## ⚙️ Configuration

### YAML Configuration

Create a `config.yaml` file:

```yaml
data:
  data_file: './tri.csv'
  annotator_columns: ['Anthea', 'Karen', 'Kimmy']
  message_column: 'message'
  test_size: 0.1
  val_size: 0.1
  label_strategy: 'ensemble'  # 'ensemble', 'majority', 'individual'

model:
  model_name: 'bert-base-uncased'
  max_length: 512
  dropout_prob: 0.1

training:
  batch_size: 16
  learning_rate: 2e-5
  num_epochs: 5
  use_gpu: true
  save_model: true

class_imbalance:
  focal_loss: true
  focal_alpha: 0.25
  focal_gamma: 2.0

evaluation:
  classification_threshold: 0.5
  save_detailed_predictions: true
```

### Programmatic Configuration

```python
from social_worker_bert import Config

config = Config()
config.data.data_file = './tri.csv'
config.training.num_epochs = 10
config.training.batch_size = 32
config.class_imbalance.focal_loss = True
```

## 📁 Library Structure

```
social_worker_bert/
├── __init__.py           # Main package exports
├── config.py             # Configuration management
├── data.py              # Data processing and loading
├── models.py            # BERT model definitions
├── training.py          # Training loop and utilities
├── inference.py         # Prediction and evaluation
└── pipeline.py          # End-to-end pipeline orchestration

examples/
├── quick_start.py       # Basic usage example
├── custom_training.py   # Custom component usage
└── batch_prediction.py  # Batch inference example

main.py                  # Command-line interface
requirements.txt         # Dependencies
```

## 🎯 Available Skills

The library can detect various social worker skills including:

- **Active Listening**: Demonstrating attentive listening
- **Empathy**: Showing understanding and compassion
- **Open-Ended Questions**: Asking questions that encourage elaboration
- **Paraphrasing**: Restating client's words for clarity
- **Reflecting**: Mirroring client's emotions and content
- **Clarifying**: Seeking additional information or understanding
- **Summarizing**: Condensing key points from conversation
- **Supportive**: Providing encouragement and validation
- **Assessment**: Evaluating client's situation or needs

*Note: Actual skills depend on your training data labels.*

## 📈 Evaluation Metrics

The library provides comprehensive evaluation including:

- **F1 Score**: Macro, micro, and weighted averages
- **Precision & Recall**: Overall and per-class metrics
- **Sample Accuracy**: Exact match accuracy for multi-label
- **Hamming Loss**: Average per-label classification error
- **Per-class Analysis**: Individual skill performance
- **Confusion Analysis**: Detailed prediction breakdowns

## 💡 Advanced Features

### Class Imbalance Handling

```python
config.class_imbalance.focal_loss = True
config.class_imbalance.focal_alpha = 0.25
config.class_imbalance.focal_gamma = 2.0
```

### Label Strategies

- **Ensemble**: Use all annotations from all annotators
- **Majority**: Require agreement from multiple annotators
- **Individual**: Use specific annotator's labels

### Custom Loss Functions

```python
from social_worker_bert.models import FocalLoss

# Focal loss for handling class imbalance
criterion = FocalLoss(alpha=0.25, gamma=2.0)
```

### Model Checkpointing

```python
# Save checkpoint during training
trainer.save_checkpoint(epoch=5, additional_data={'custom_info': 'value'})

# Load checkpoint to resume training
resume_epoch = trainer.load_checkpoint('checkpoint_epoch_5.pt')
```

### Data Augmentation

The library includes sophisticated data augmentation techniques to handle class imbalance and improve model performance:

```python
# Enable data augmentation
config.data_augmentation.enable = True
config.data_augmentation.rare_threshold = 1.0  # Classes with <1% of samples
config.data_augmentation.augment_factor = 3    # Generate 3x more samples
config.data_augmentation.methods = ['paraphrase', 'synonym_replacement']
```

**Available Augmentation Methods:**
- **Paraphrase**: Combines multiple techniques for natural text variation
- **Synonym Replacement**: Replaces words with synonyms using WordNet
- **Random Insertion**: Inserts synonyms at random positions
- **Random Deletion**: Removes words with low probability
- **Random Swap**: Swaps word positions randomly

**How it works:**
1. Identifies rare classes based on the threshold percentage
2. Finds samples containing rare classes
3. Generates augmented samples using selected methods
4. Applies augmentation only to training data (not validation/test)

```python
# Use data augmentation directly
from social_worker_bert.augmentation import DataAugmenter

augmenter = DataAugmenter(config)
augmented_texts, augmented_labels = augmenter.augment_dataset(texts, labels)
```

## 🔧 Command Line Interface

### Training Commands

```bash
# Basic training
python main.py train --data ./tri.csv

# Custom configuration
python main.py train --config config.yaml --output ./my_results

# Override parameters
python main.py train --data ./tri.csv --epochs 10 --batch-size 32 --learning-rate 1e-5
```

### Prediction Commands

```bash
# Single text prediction
python main.py predict --model model.pt --text "How are you feeling?"

# Batch prediction from file
python main.py predict --model model.pt --file texts.csv --output predictions.csv

# Custom threshold
python main.py predict --model model.pt --text "Hello" --threshold 0.3
```

### Python Library Inference

You can also use the library directly in Python for inference:

#### Single Text Prediction

```python
from social_worker_bert import SocialWorkerBERTPipeline
from social_worker_bert.models import load_model
from social_worker_bert import Config, Predictor

# Option 1: Using trained pipeline
pipeline = SocialWorkerBERTPipeline.load_from_checkpoint('./results/social_worker_bert_model.pt')
predictions = pipeline.predict(["How are you feeling today?"])
print(f"Predicted skills: {predictions[0]}")

# Option 2: Using Predictor directly
config = Config()
model, tokenizer, data = load_model('./results/social_worker_bert_model.pt', config.device)
predictor = Predictor(config, model, tokenizer, data['skill_labels'])

# Predict with custom threshold
predicted_skills, probabilities = predictor.predict_single(
    "How are you feeling today?", 
    threshold=0.3
)
print(f"Predicted skills: {predicted_skills}")
print(f"Probabilities: {probabilities}")
```

#### Batch Prediction

```python
# Predict multiple texts at once
texts = [
    "How are you feeling today?",
    "Can you tell me more about that?",
    "I understand this must be difficult for you."
]

# Using pipeline
predictions = pipeline.predict(texts)
for i, pred in enumerate(predictions):
    print(f"Text {i+1}: {pred}")

# Using predictor for more control
predictions, probabilities = predictor.predict_batch(texts, threshold=0.4)

# Create detailed results table
import pandas as pd
results_df = predictor.create_detailed_prediction_table(texts, predictions, probabilities)
results_df.to_csv('batch_predictions.csv', index=False)
```

#### Model Evaluation on Custom Data

```python
from torch.utils.data import DataLoader
from social_worker_bert.data import SocialWorkSkillDataset

# Prepare your test data
test_texts = ["Your test messages here..."]
test_labels = [[1, 0, 1, 0, 0]]  # Binary labels for each skill

# Create dataset and dataloader
test_dataset = SocialWorkSkillDataset(test_texts, test_labels, tokenizer, config.model.max_length)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# Evaluate
evaluation_results = predictor.evaluate_model(test_loader)

# Print comprehensive report
predictor.print_evaluation_report(evaluation_results['metrics'])

# Save detailed predictions
predictor.save_predictions(
    test_texts, 
    evaluation_results['predictions'], 
    evaluation_results['probabilities'],
    'detailed_evaluation.csv',
    true_labels=test_labels
)
```

#### Real-time Inference Server

```python
from flask import Flask, request, jsonify

app = Flask(__name__)

# Load model once at startup
pipeline = SocialWorkerBERTPipeline.load_from_checkpoint('./results/social_worker_bert_model.pt')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    text = data.get('text', '')
    threshold = data.get('threshold', 0.5)
    
    if not text:
        return jsonify({'error': 'No text provided'}), 400
    
    # Make prediction
    predictions = pipeline.predict([text], threshold=threshold)
    
    return jsonify({
        'text': text,
        'predicted_skills': predictions[0],
        'threshold': threshold
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)
```

#### Integration with Custom Applications

```python
class SocialWorkerAssistant:
    def __init__(self, model_path):
        self.pipeline = SocialWorkerBERTPipeline.load_from_checkpoint(model_path)
    
    def analyze_conversation(self, messages):
        """Analyze a conversation and return skill usage summary."""
        predictions = self.pipeline.predict(messages)
        
        # Aggregate skills across conversation
        all_skills = []
        for pred in predictions:
            all_skills.extend(pred)
        
        skill_counts = {}
        for skill in all_skills:
            skill_counts[skill] = skill_counts.get(skill, 0) + 1
        
        return {
            'total_messages': len(messages),
            'skill_usage': skill_counts,
            'unique_skills': len(set(all_skills)),
            'detailed_predictions': predictions
        }
    
    def get_skill_recommendations(self, text):
        """Get skill recommendations for improving a message."""
        predicted_skills, probabilities = self.predictor.predict_single(text)
        
        # Find skills with low probability that could be improved
        recommendations = []
        for skill, prob in probabilities.items():
            if prob < 0.3:  # Threshold for recommendation
                recommendations.append(f"Consider adding {skill} (confidence: {prob:.2f})")
        
        return recommendations

# Usage
assistant = SocialWorkerAssistant('./results/social_worker_bert_model.pt')
analysis = assistant.analyze_conversation([
    "How are you feeling today?",
    "Can you tell me more about that?"
])
print(analysis)
```
