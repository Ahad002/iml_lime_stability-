"""
Data Loading Module
===================
Handles loading and preprocessing of SST-2 dataset.
"""

import pandas as pd
from datasets import load_dataset


class DataLoader:
    """Load and manage SST-2 sentiment dataset"""
    
    def __init__(self, dataset_name="glue", config_name="sst2"):
        """
        Initialize data loader
        
        Parameters:
        -----------
        dataset_name : str
            Dataset identifier on Hugging Face
        config_name : str
            Specific configuration/subset
        """
        print(f"📥 Loading {config_name} from {dataset_name}...")
        
        dataset = load_dataset(dataset_name, config_name)
        
        self.train_df = pd.DataFrame(dataset['train'])
        self.test_df = pd.DataFrame(dataset['validation'])
        
        # Add sentence length
        self.test_df['length'] = self.test_df['sentence'].str.split().str.len()
        
        print(f"✅ Loaded {len(self.train_df)} train, {len(self.test_df)} test samples")
    
    def get_samples(self, n_samples=100, seed=42):
        """Get random sample of sentences"""
        return self.test_df.sample(n=n_samples, random_state=seed)
    
    def get_by_length(self, min_len, max_len):
        """Filter sentences by word count"""
        return self.test_df[(self.test_df['length'] >= min_len) & 
                           (self.test_df['length'] <= max_len)]
    
    def get_length_groups(self):
        """Return short, medium, long sentence groups"""
        short = self.test_df[self.test_df['length'] <= 7]
        medium = self.test_df[(self.test_df['length'] > 7) & 
                             (self.test_df['length'] <= 15)]
        long = self.test_df[self.test_df['length'] > 15]
        
        print(f"📏 Short (≤7): {len(short)} | Medium (8-15): {len(medium)} | Long (>15): {len(long)}")
        
        return short, medium, long
