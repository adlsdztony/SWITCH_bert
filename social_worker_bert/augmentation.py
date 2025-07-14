"""
Data Augmentation Module for Social Worker BERT

This module implements various data augmentation techniques to improve model performance,
especially for handling class imbalance and improving generalization.

Supported augmentation methods:
- Paraphrasing using back-translation
- Synonym replacement
- Random insertion
- Random deletion
- Random swap
"""

import random
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
from collections import Counter
import nltk

# Download required NLTK data if not already present
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.corpus import wordnet as wn

from .config import Config


class DataAugmenter:
    """Main data augmentation class for social worker skill classification."""
    
    def __init__(self, config: Config):
        """
        Initialize data augmenter.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.stop_words = set(stopwords.words('english'))
        self.random_state = config.data.random_state
        random.seed(self.random_state)
        np.random.seed(self.random_state)
        
    def get_synonyms(self, word: str) -> List[str]:
        """
        Get synonyms for a given word using WordNet.
        
        Args:
            word: Input word
            
        Returns:
            List of synonyms
        """
        synonyms = set()
        
        # Get synsets for the word
        for syn in wn.synsets(word):
            for lemma in syn.lemmas():
                synonym = lemma.name().replace('_', ' ')
                if synonym.lower() != word.lower():
                    synonyms.add(synonym)
        
        return list(synonyms)
    
    def synonym_replacement(self, text: str, num_replacements: int = 1) -> str:
        """
        Replace random words with their synonyms.
        
        Args:
            text: Input text
            num_replacements: Number of words to replace
            
        Returns:
            Augmented text
        """
        words = word_tokenize(text.lower())
        
        # Filter out stop words and punctuation
        replaceable_words = [word for word in words 
                           if word.isalpha() and word not in self.stop_words and len(word) > 2]
        
        if not replaceable_words:
            return text
        
        # Randomly select words to replace
        words_to_replace = random.sample(
            replaceable_words, 
            min(num_replacements, len(replaceable_words))
        )
        
        augmented_words = words.copy()
        
        for word in words_to_replace:
            synonyms = self.get_synonyms(word)
            if synonyms:
                synonym = random.choice(synonyms)
                # Replace first occurrence
                for i, w in enumerate(augmented_words):
                    if w == word:
                        augmented_words[i] = synonym
                        break
        
        return ' '.join(augmented_words)
    
    def random_insertion(self, text: str, num_insertions: int = 1) -> str:
        """
        Insert random synonyms at random positions.
        
        Args:
            text: Input text
            num_insertions: Number of words to insert
            
        Returns:
            Augmented text
        """
        words = word_tokenize(text.lower())
        
        # Get words that can have synonyms
        replaceable_words = [word for word in words 
                           if word.isalpha() and word not in self.stop_words and len(word) > 2]
        
        if not replaceable_words:
            return text
        
        augmented_words = words.copy()
        
        for _ in range(num_insertions):
            # Choose a random word and get its synonym
            word = random.choice(replaceable_words)
            synonyms = self.get_synonyms(word)
            if synonyms:
                synonym = random.choice(synonyms)
                # Insert at random position
                insert_pos = random.randint(0, len(augmented_words))
                augmented_words.insert(insert_pos, synonym)
        
        return ' '.join(augmented_words)
    
    def random_deletion(self, text: str, deletion_prob: float = 0.1) -> str:
        """
        Randomly delete words from the text.
        
        Args:
            text: Input text
            deletion_prob: Probability of deleting each word
            
        Returns:
            Augmented text
        """
        words = word_tokenize(text.lower())
        
        # Don't delete if text is too short
        if len(words) <= 3:
            return text
        
        # Keep at least one word
        augmented_words = []
        for word in words:
            if random.random() > deletion_prob:
                augmented_words.append(word)
        
        # Ensure we have at least one word
        if not augmented_words:
            augmented_words = [random.choice(words)]
        
        return ' '.join(augmented_words)
    
    def random_swap(self, text: str, num_swaps: int = 1) -> str:
        """
        Randomly swap positions of words.
        
        Args:
            text: Input text
            num_swaps: Number of swaps to perform
            
        Returns:
            Augmented text
        """
        words = word_tokenize(text.lower())
        
        if len(words) < 2:
            return text
        
        augmented_words = words.copy()
        
        for _ in range(num_swaps):
            # Choose two random positions
            idx1, idx2 = random.sample(range(len(augmented_words)), 2)
            # Swap words
            augmented_words[idx1], augmented_words[idx2] = augmented_words[idx2], augmented_words[idx1]
        
        return ' '.join(augmented_words)
    
    def paraphrase_simple(self, text: str) -> str:
        """
        Simple paraphrasing using multiple augmentation techniques.
        
        Args:
            text: Input text
            
        Returns:
            Paraphrased text
        """
        # Apply a combination of techniques
        augmented = text
        
        # Random choice of techniques
        techniques = [
            lambda x: self.synonym_replacement(x, num_replacements=2),
            lambda x: self.random_insertion(x, num_insertions=1),
            lambda x: self.random_swap(x, num_swaps=1)
        ]
        
        # Apply 1-2 techniques
        selected_techniques = random.sample(techniques, random.randint(1, 2))
        
        for technique in selected_techniques:
            augmented = technique(augmented)
        
        return augmented
    
    def augment_text(self, text: str, method: str) -> str:
        """
        Apply specific augmentation method to text.
        
        Args:
            text: Input text
            method: Augmentation method name
            
        Returns:
            Augmented text
        """
        if method == 'synonym_replacement':
            return self.synonym_replacement(text, num_replacements=random.randint(1, 3))
        elif method == 'paraphrase':
            return self.paraphrase_simple(text)
        elif method == 'random_insertion':
            return self.random_insertion(text, num_insertions=random.randint(1, 2))
        elif method == 'random_deletion':
            return self.random_deletion(text, deletion_prob=0.1)
        elif method == 'random_swap':
            return self.random_swap(text, num_swaps=random.randint(1, 2))
        else:
            print(f"⚠️  Unknown augmentation method: {method}")
            return text
    
    def identify_rare_classes(self, labels: List[List[str]]) -> Dict[str, int]:
        """
        Identify rare classes based on the threshold.
        
        Args:
            labels: List of label lists
            
        Returns:
            Dictionary mapping class names to their counts
        """
        # Count occurrences of each label
        label_counts = Counter()
        for label_list in labels:
            for label in label_list:
                label_counts[label] += 1
        
        # Calculate threshold
        total_samples = len(labels)
        threshold = total_samples * self.config.data_augmentation.rare_threshold
        
        # Identify rare classes
        rare_classes = {label: count for label, count in label_counts.items() 
                       if count < threshold}
        
        print(f"📊 Identified {len(rare_classes)} rare classes (threshold: {threshold:.1f}):")
        for label, count in rare_classes.items():
            print(f"  {label}: {count} samples")
        
        return rare_classes
    
    def augment_dataset(self, texts: List[str], labels: List[List[str]]) -> Tuple[List[str], List[List[str]]]:
        """
        Augment the dataset by generating additional samples for rare classes.
        
        Args:
            texts: List of text samples
            labels: List of label lists
            
        Returns:
            Tuple of (augmented_texts, augmented_labels)
        """
        if not self.config.data_augmentation.enable:
            print("🔄 Data augmentation disabled, returning original dataset")
            return texts, labels
        
        print("\n" + "="*60)
        print("🔄 DATA AUGMENTATION")
        print("="*60)
        
        # Identify rare classes
        rare_classes = self.identify_rare_classes(labels)
        
        if not rare_classes:
            print("ℹ️  No rare classes found, returning original dataset")
            return texts, labels
        
        # Create augmented samples
        augmented_texts = texts.copy()
        augmented_labels = labels.copy()
        
        # For each rare class, find samples and augment them
        for rare_class, count in rare_classes.items():
            print(f"\n🎯 Augmenting samples for class: {rare_class}")
            
            # Find samples containing this rare class
            rare_samples = []
            for i, label_list in enumerate(labels):
                if rare_class in label_list:
                    rare_samples.append((i, texts[i], label_list))
            
            if not rare_samples:
                continue
            
            # Calculate how many augmented samples to generate
            target_samples = min(
                count * self.config.data_augmentation.augment_factor,
                len(rare_samples) * 5  # Don't exceed 5x the original samples
            )
            
            samples_to_generate = max(0, target_samples - count)
            
            print(f"  Original samples: {len(rare_samples)}")
            print(f"  Target samples: {target_samples}")
            print(f"  Generating: {samples_to_generate} new samples")
            
            # Generate augmented samples
            for _ in range(samples_to_generate):
                # Randomly select a sample to augment
                _, original_text, original_labels = random.choice(rare_samples)
                
                # Randomly select augmentation method
                method = random.choice(self.config.data_augmentation.methods)
                
                # Apply augmentation
                augmented_text = self.augment_text(original_text, method)
                
                # Add to augmented dataset
                augmented_texts.append(augmented_text)
                augmented_labels.append(original_labels.copy())
        
        print("\n✅ Data augmentation completed!")
        print(f"  Original dataset size: {len(texts)}")
        print(f"  Augmented dataset size: {len(augmented_texts)}")
        print(f"  Added {len(augmented_texts) - len(texts)} samples")
        
        return augmented_texts, augmented_labels
    
    def augment_dataframe(self, df: pd.DataFrame, text_column: str, label_columns: List[str]) -> pd.DataFrame:
        """
        Augment a pandas DataFrame containing text and labels.
        
        Args:
            df: Input DataFrame
            text_column: Name of the text column
            label_columns: List of label column names
            
        Returns:
            Augmented DataFrame
        """
        # Extract texts and labels
        texts = df[text_column].tolist()
        
        # Process labels - combine multiple annotator columns
        labels = []
        for _, row in df.iterrows():
            sample_labels = []
            for col in label_columns:
                if not pd.isna(row[col]) and row[col] != "":
                    # Split by comma and clean
                    col_labels = [label.strip() for label in str(row[col]).split(',')]
                    sample_labels.extend(col_labels)
            
            # Remove duplicates while preserving order
            unique_labels = []
            for label in sample_labels:
                if label and label not in unique_labels:
                    unique_labels.append(label)
            
            labels.append(unique_labels)
        
        # Augment the dataset
        augmented_texts, augmented_labels = self.augment_dataset(texts, labels)
        
        # Create augmented DataFrame
        augmented_rows = []
        
        # Add original rows
        for i, (text, label_list) in enumerate(zip(texts, labels)):
            row = df.iloc[i].copy()
            augmented_rows.append(row)
        
        # Add augmented rows
        for i in range(len(texts), len(augmented_texts)):
            text = augmented_texts[i]
            label_list = augmented_labels[i]
            
            # Create new row based on the original structure
            new_row = df.iloc[0].copy()  # Copy structure from first row
            new_row[text_column] = text
            
            # Distribute labels across annotator columns
            # For simplicity, put all labels in the first annotator column
            if label_columns:
                new_row[label_columns[0]] = ', '.join(label_list)
                for col in label_columns[1:]:
                    new_row[col] = ""
            
            augmented_rows.append(new_row)
        
        # Create augmented DataFrame
        augmented_df = pd.DataFrame(augmented_rows)
        augmented_df.reset_index(drop=True, inplace=True)
        
        return augmented_df


def augment_data_if_enabled(config: Config, texts: List[str], labels: List[List[str]]) -> Tuple[List[str], List[List[str]]]:
    """
    Convenience function to augment data if augmentation is enabled.
    
    Args:
        config: Configuration object
        texts: List of text samples
        labels: List of label lists
        
    Returns:
        Tuple of (texts, labels) - augmented if enabled, original otherwise
    """
    if config.data_augmentation.enable:
        augmenter = DataAugmenter(config)
        return augmenter.augment_dataset(texts, labels)
    else:
        return texts, labels
