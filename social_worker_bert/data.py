"""
Data Processing Module for Social Worker BERT

This module handles all data preprocessing tasks including:
- Text extraction and cleaning
- Label processing and normalization
- Multi-label encoding
- Data splitting
- Dataset creation
"""

import pandas as pd
import numpy as np
from collections import Counter
from typing import List, Tuple, Dict, Any
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer

from .config import Config


class SocialWorkSkillDataset(Dataset):
    """PyTorch Dataset for social worker skill classification."""
    
    def __init__(self, texts: List[str], labels: np.ndarray, tokenizer: BertTokenizer, max_length: int = 512):
        """
        Initialize dataset.
        
        Args:
            texts: List of text messages
            labels: Multi-label binary array
            tokenizer: BERT tokenizer
            max_length: Maximum sequence length
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self) -> int:
        return len(self.texts)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = str(self.texts[idx])
        labels = self.labels[idx]
        
        # Tokenize the text
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.FloatTensor(labels)
        }


class DataProcessor:
    """Main data processing class for social worker skill classification."""
    
    def __init__(self, config: Config):
        """
        Initialize data processor.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.mlb = None
        self.skill_labels = None
        self.class_weights = None
        
    def extract_social_worker_message(self, message: str) -> str:
        """Extract and clean social worker message."""
        return message.strip()
    
    def normalize_skill_label(self, skill: str) -> str:
        """Normalize skill labels to handle case inconsistencies and variations."""
        if not skill or pd.isna(skill):
            return ""
        
        # Convert to string and strip whitespace
        skill = str(skill).strip()
        
        # Convert to title case for consistency
        skill = skill.title()
        
        # Handle common variations and synonyms
        skill_mapping = {
            # Case variations
            'Active Listening': 'Active Listening',
            'active listening': 'Active Listening',
            'ACTIVE LISTENING': 'Active Listening',
            
            # Common synonyms and variations
            'Open-Ended Questions': 'Open-Ended Questions',
            'Open Ended Questions': 'Open-Ended Questions',
            'Open Questions': 'Open-Ended Questions',
            'Openended Questions': 'Open-Ended Questions',
            
            'Paraphrasing': 'Paraphrasing',
            'Paraphrase': 'Paraphrasing',
            
            'Reflecting': 'Reflecting',
            'Reflection': 'Reflecting',
            'Reflective Listening': 'Reflecting',
            
            'Empathy': 'Empathy',
            'Empathic': 'Empathy',
            'Empathetic': 'Empathy',
            
            'Clarifying': 'Clarifying',
            'Clarification': 'Clarifying',
            
            'Summarizing': 'Summarizing',
            'Summary': 'Summarizing',
            'Summarising': 'Summarizing',
            
            'Supportive': 'Supportive',
            'Support': 'Supportive',
            'Supportive Questioning': 'Supportive',
            
            'Assessment': 'Assessment',
            'Assess': 'Assessment',
            'Assessing': 'Assessment',
        }
        
        # Apply mapping if exists, otherwise return normalized version
        return skill_mapping.get(skill, skill)
    
    def process_skill_labels(self, skill_str: str) -> List[str]:
        """Process skill labels from annotator columns with normalization."""
        if pd.isna(skill_str) or skill_str == "":
            return []
        
        # Split by comma and clean up
        skills = [skill.strip() for skill in str(skill_str).split(',')]
        # Remove empty strings and normalize
        skills = [self.normalize_skill_label(skill) for skill in skills if skill.strip() != ""]
        # Remove duplicates while preserving order
        unique_skills = []
        for skill in skills:
            if skill and skill not in unique_skills:
                unique_skills.append(skill)
        
        return unique_skills
    
    def create_ensemble_labels(self, row: pd.Series, annotator_columns: List[str], strategy: str = 'ensemble') -> List[str]:
        """Create labels based on different strategies."""
        all_skills = []
        
        for col in annotator_columns:
            skills = self.process_skill_labels(row[col])
            all_skills.extend(skills)
        
        if strategy == 'ensemble':
            # Use all skills from all annotators
            return list(set(all_skills))
        
        elif strategy == 'majority':
            # Use skills that appear in at least min_annotator_agreement annotators
            skill_counts = Counter(all_skills)
            return [skill for skill, count in skill_counts.items() 
                    if count >= self.config.data.min_annotator_agreement]
        
        else:
            return list(set(all_skills))
    
    def load_and_preprocess_data(self):
        """
        Load and preprocess the dataset.
        
        Returns:
            For single file mode: Tuple of (texts, labels, skill_labels)
            For separate files mode: Dict with 'train' and 'test' keys, each containing (texts, labels, skill_labels)
        """
        print("📂 Loading dataset...")
        
        # Determine data mode
        self.config.determine_data_mode()
        data_files = self.config.get_data_files()
        
        if self.config.data.mode == 'separate_files':
            return self._load_separate_files(data_files)
        else:
            return self._load_single_file(data_files['single'])
    
    def _load_single_file(self, data_file: str) -> Tuple[List[str], np.ndarray, List[str]]:
        """Load and preprocess data from a single file."""
        try:
            df = pd.read_csv(data_file)
            print(f"✅ Dataset loaded successfully! Shape: {df.shape}")
        except FileNotFoundError:
            raise FileNotFoundError(f"Dataset file not found at {data_file}")
        
        return self._preprocess_dataframe(df, "single file dataset")
    
    def _load_separate_files(self, data_files: Dict[str, str]) -> Dict[str, Tuple[List[str], np.ndarray, List[str]]]:
        """Load and preprocess data from separate train and test files."""
        results = {}
        all_skills = set()
        
        # First pass: collect all unique skills from both files
        for split_name, file_path in data_files.items():
            try:
                df = pd.read_csv(file_path)
                print(f"✅ {split_name.capitalize()} dataset loaded successfully! Shape: {df.shape}")
                
                # Process skill labels to collect all unique skills
                for col in self.config.data.annotator_columns:
                    df[f'{col}_skills'] = df[col].apply(self.process_skill_labels)
                
                # Create ensemble labels
                df['final_labels'] = df.apply(
                    lambda row: self.create_ensemble_labels(
                        row, self.config.data.annotator_columns, self.config.data.label_strategy
                    ), 
                    axis=1
                )
                
                # Collect all skills
                for labels in df['final_labels']:
                    all_skills.update(labels)
                
                # Store dataframe for second pass
                results[f'{split_name}_df'] = df
                
            except FileNotFoundError:
                raise FileNotFoundError(f"{split_name.capitalize()} dataset file not found at {file_path}")
        
        # Set unified skill labels
        self.skill_labels = sorted(list(all_skills))
        print(f"\n🏷️  Total unique skills across all files: {len(self.skill_labels)}")
        print(f"  Skills: {self.skill_labels}")
        
        # Create multi-label binarizer with unified labels
        self.mlb = MultiLabelBinarizer(classes=self.skill_labels)
        self.mlb.fit([self.skill_labels])  # Fit with all possible labels
        
        # Second pass: preprocess each split with unified skill labels
        final_results = {}
        for split_name in ['train', 'test']:
            df = results[f'{split_name}_df']
            texts, labels, _ = self._preprocess_dataframe(df, f"{split_name} dataset", use_existing_mlb=True)
            final_results[split_name] = (texts, labels, self.skill_labels)
        
        return final_results
    
    def _preprocess_dataframe(self, df: pd.DataFrame, dataset_name: str, use_existing_mlb: bool = False) -> Tuple[List[str], np.ndarray, List[str]]:
        """Preprocess a single dataframe."""
        print(f"\n🔄 Preprocessing {dataset_name}...")
        
        # Extract social worker messages
        df['sw_message'] = df[self.config.data.message_column].apply(self.extract_social_worker_message)
        
        # Process skill labels if not already done
        if 'final_labels' not in df.columns:
            # Process skill labels for each annotator
            for col in self.config.data.annotator_columns:
                df[f'{col}_skills'] = df[col].apply(self.process_skill_labels)
            
            # Create ensemble labels based on strategy
            df['final_labels'] = df.apply(
                lambda row: self.create_ensemble_labels(
                    row, self.config.data.annotator_columns, self.config.data.label_strategy
                ), 
                axis=1
            )
        
        # Remove rows with empty social worker messages or no labels
        initial_length = len(df)
        df = df[(df['sw_message'].str.len() > 0) & (df['final_labels'].str.len() > 0)]
        removed_count = initial_length - len(df)
        if removed_count > 0:
            print(f"📉 Removed {removed_count} rows with empty messages or labels from {dataset_name}")
        
        if not use_existing_mlb:
            # Get all unique skills for this dataset
            all_skills = set()
            for labels in df['final_labels']:
                all_skills.update(labels)
            
            self.skill_labels = sorted(list(all_skills))
            
            print(f"\n🏷️  Label Information for {dataset_name}:")
            print(f"  Total unique skills: {len(self.skill_labels)}")
            print(f"  Skills: {self.skill_labels}")
            
            # Create multi-label binarizer
            self.mlb = MultiLabelBinarizer(classes=self.skill_labels)
            y_multilabel = self.mlb.fit_transform(df['final_labels'])
        else:
            # Use existing MLB (for separate files mode)
            y_multilabel = self.mlb.transform(df['final_labels'])
        
        # Prepare final dataset
        texts = df['sw_message'].tolist()
        labels = y_multilabel
        
        print(f"\n✅ Data preprocessing completed for {dataset_name}!")
        print(f"  Final dataset size: {len(texts)}")
        print(f"  Number of skills: {len(self.skill_labels)}")
        print(f"  Average skills per conversation: {np.mean(labels.sum(axis=1)):.2f}")
        print(f"  Max skills per conversation: {labels.sum(axis=1).max()}")
        print(f"  Min skills per conversation: {labels.sum(axis=1).min()}")
        
        return texts, labels, self.skill_labels
    
    def split_data(self, data_input) -> Tuple[List[str], List[str], List[str], np.ndarray, np.ndarray, np.ndarray]:
        """
        Split data into train, validation, and test sets.
        
        Args:
            data_input: Either (texts, labels) for single file mode or 
                       dict with 'train' and 'test' keys for separate files mode
        
        Returns:
            Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        print("\n🔄 Splitting data...")
        
        if isinstance(data_input, dict):
            # Separate files mode
            return self._split_separate_files(data_input)
        else:
            # Single file mode
            texts, labels = data_input
            return self._split_single_file(texts, labels)
    
    def _split_single_file(self, texts: List[str], labels: np.ndarray) -> Tuple[List[str], List[str], List[str], np.ndarray, np.ndarray, np.ndarray]:
        """Split data from single file into train, validation, and test sets."""
        # First split: train+val vs test
        X_temp, X_test, y_temp, y_test = train_test_split(
            texts, labels, 
            test_size=self.config.data.test_size, 
            random_state=self.config.data.random_state, 
            stratify=None
        )
        
        # Second split: train vs val
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, 
            test_size=self.config.data.val_size, 
            random_state=self.config.data.random_state, 
            stratify=None
        )
        
        self._print_split_summary(X_train, X_val, X_test, y_train, y_val, y_test)
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def _split_separate_files(self, data_dict: Dict[str, Tuple[List[str], np.ndarray, List[str]]]) -> Tuple[List[str], List[str], List[str], np.ndarray, np.ndarray, np.ndarray]:
        """Handle data that's already split into train and test files."""
        train_texts, train_labels, _ = data_dict['train']
        test_texts, test_labels, _ = data_dict['test']
        
        # Create validation set from training data
        X_train, X_val, y_train, y_val = train_test_split(
            train_texts, train_labels,
            test_size=self.config.data.val_size,
            random_state=self.config.data.random_state,
            stratify=None
        )
        
        X_test, y_test = test_texts, test_labels
        
        self._print_split_summary(X_train, X_val, X_test, y_train, y_val, y_test)
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def _print_split_summary(self, X_train, X_val, X_test, y_train, y_val, y_test):
        """Print summary of data splits."""
        print("✅ Data split completed!")
        print(f"  Training set: {len(X_train)} samples")
        print(f"  Validation set: {len(X_val)} samples")
        print(f"  Test set: {len(X_test)} samples")
        
        # Analyze split distributions
        train_skills_per_sample = y_train.sum(axis=1).mean()
        val_skills_per_sample = y_val.sum(axis=1).mean()
        test_skills_per_sample = y_test.sum(axis=1).mean()
        
        print("\n📊 Skills distribution in splits:")
        print(f"  Train avg skills per sample: {train_skills_per_sample:.2f}")
        print(f"  Val avg skills per sample: {val_skills_per_sample:.2f}")
        print(f"  Test avg skills per sample: {test_skills_per_sample:.2f}")
    
    def create_data_loaders(self, X_train: List[str], X_val: List[str], X_test: List[str],
                           y_train: np.ndarray, y_val: np.ndarray, y_test: np.ndarray,
                           tokenizer: BertTokenizer) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Create PyTorch data loaders.
        
        Args:
            X_train, X_val, X_test: Text data splits
            y_train, y_val, y_test: Label data splits
            tokenizer: BERT tokenizer
        
        Returns:
            Tuple of (train_loader, val_loader, test_loader)
        """
        print("\n📦 Creating data loaders...")
        
        train_dataset = SocialWorkSkillDataset(X_train, y_train, tokenizer, self.config.model.max_length)
        val_dataset = SocialWorkSkillDataset(X_val, y_val, tokenizer, self.config.model.max_length)
        test_dataset = SocialWorkSkillDataset(X_test, y_test, tokenizer, self.config.model.max_length)
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.config.training.batch_size, 
            shuffle=True, 
            num_workers=0
        )
        
        val_loader = DataLoader(
            val_dataset, 
            batch_size=self.config.training.batch_size, 
            shuffle=False, 
            num_workers=0
        )
        
        test_loader = DataLoader(
            test_dataset, 
            batch_size=self.config.training.batch_size, 
            shuffle=False, 
            num_workers=0
        )
        
        print("✅ Data loaders created!")
        print(f"  Training batches: {len(train_loader)}")
        print(f"  Validation batches: {len(val_loader)}")
        print(f"  Test batches: {len(test_loader)}")
        
        return train_loader, val_loader, test_loader
    
    def process_all(self, tokenizer: BertTokenizer) -> Dict[str, Any]:
        """
        Run complete data processing pipeline.
        
        Args:
            tokenizer: BERT tokenizer
        
        Returns:
            Dictionary containing processed data and loaders
        """
        # Load and preprocess data
        data_result = self.load_and_preprocess_data()
        
        # Handle different data modes
        if isinstance(data_result, dict):
            # Separate files mode
            skill_labels = data_result['train'][2]  # skill_labels from train set
            X_train, X_val, X_test, y_train, y_val, y_test = self.split_data(data_result)
        else:
            # Single file mode
            texts, labels, skill_labels = data_result
            X_train, X_val, X_test, y_train, y_val, y_test = self.split_data((texts, labels))
        
        # Create data loaders
        train_loader, val_loader, test_loader = self.create_data_loaders(
            X_train, X_val, X_test, y_train, y_val, y_test, tokenizer
        )
        
        # Update config with number of labels
        self.config.model.num_labels = len(skill_labels)
        
        return {
            'texts': X_train + X_val + X_test,  # All texts combined
            'labels': np.vstack([y_train, y_val, y_test]),  # All labels combined
            'skill_labels': skill_labels,
            'splits': {
                'X_train': X_train, 'X_val': X_val, 'X_test': X_test,
                'y_train': y_train, 'y_val': y_val, 'y_test': y_test
            },
            'loaders': {
                'train': train_loader,
                'val': val_loader,
                'test': test_loader
            },
            'mlb': self.mlb,
            'class_weights': self.class_weights
        }
