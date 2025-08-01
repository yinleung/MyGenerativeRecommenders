#reco_dataset.py
import os
import ast
from typing import Dict, List, Optional, Any, Union

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from generative_recommenders_pl.utils.logger import RankedLogger

log = RankedLogger(__name__)


class FeatureLookupTable:
    """Dynamic feature lookup table"""
    
    def __init__(self, csv_path: str, id_column: str = None):
        self.csv_path = csv_path
        self.features = {}
        self.feature_names = []
        self.id_column = id_column
        
        if os.path.exists(csv_path):
            self._load_features()
        else:
            log.warning(f"Feature file not found: {csv_path}")
    
    def _load_features(self):
        """Load all features dynamically from CSV"""
        df = pd.read_csv(self.csv_path)
        
        # Auto-detect ID column if not specified
        if self.id_column is None:
            # Look for common ID patterns
            for col in df.columns:
                if col.lower().endswith('_id') or col.lower().endswith('id'):
                    self.id_column = col
                    break
            else:
                self.id_column = df.columns[0]  # Fallback to first column
        
        # Get feature columns (all except ID)
        self.feature_names = [col for col in df.columns if col != self.id_column]
        log.info(f"Loading {self.csv_path}: ID={self.id_column}, features={self.feature_names}")
        
        # Create lookup dictionaries
        for feature_name in self.feature_names:
            self.features[feature_name] = {}
            for _, row in df.iterrows():
                item_id = row[self.id_column]
                feature_value = row[feature_name]
                
                # Parse string lists like "[1,2,3]"
                if isinstance(feature_value, str):
                    try:
                        feature_value = ast.literal_eval(feature_value)
                    except (ValueError, SyntaxError):
                        pass  # Keep as string
                
                self.features[feature_name][item_id] = feature_value
    
    def get_feature(self, feature_name: str, item_id: int, default_value: Any = None):
        """Get feature value for specific item"""
        return self.features.get(feature_name, {}).get(item_id, default_value)


class RecoDataset(Dataset):
    """Fully dynamic sequential recommendation dataset"""
    
    def __init__(
        self,
        main_data_file: Union[str, pd.DataFrame],
        item_features_path: str,
        user_features_path: Optional[str] = None,
        max_sequence_length: int = 200,
        ignore_last_n: int = 0,
        shift_id_by: int = 0,
        chronological: bool = False,
        sample_ratio: float = 1.0,
        additional_columns: Optional[List[str]] = None,
        sequence_prefix: str = "sequence_",
        user_id_column: str = None,
    ):
        super().__init__()
        
        # Load main data
        self.main_df = pd.read_csv(main_data_file) if isinstance(main_data_file, str) else main_data_file
        
        # Store parameters
        self.max_sequence_length = max_sequence_length
        self.ignore_last_n = ignore_last_n
        self.shift_id_by = shift_id_by
        self.chronological = chronological
        self.sample_ratio = sample_ratio
        self.additional_columns = additional_columns or []
        self.sequence_prefix = sequence_prefix
        
        # Auto-detect user ID column
        self.user_id_column = self._detect_user_id_column(user_id_column)
        
        # Load feature tables
        self.item_features = FeatureLookupTable(item_features_path)
        # Note: user features are in main_data_file as additional_columns, no separate user_features_path needed
        
        # Analyze dataset structure
        self.sequence_columns = [col for col in self.main_df.columns if col.startswith(sequence_prefix)]
        self.static_columns = [col for col in self.main_df.columns 
                             if not col.startswith(sequence_prefix) 
                             and col not in [self.user_id_column, 'index', 'idx']]
        
        log.info(f"Dataset: user_id={self.user_id_column}, sequences={self.sequence_columns}, static={self.static_columns}")
        
        self._cache = {}
    
    def _detect_user_id_column(self, user_id_column: Optional[str]) -> str:
        """Auto-detect user ID column"""
        if user_id_column and user_id_column in self.main_df.columns:
            return user_id_column
        
        # Auto-detect patterns
        for col in self.main_df.columns:
            if col.lower() in ['user_id', 'userid', 'user', 'uid']:
                return col
        
        raise ValueError("Cannot detect user ID column")
    
    def __len__(self) -> int:
        return len(self.main_df)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if idx in self._cache:
            return self._cache[idx]
        
        sample = self._load_item(idx)
        self._cache[idx] = sample
        return sample
    
    def _load_item(self, idx: int) -> Dict[str, torch.Tensor]:
        """Load and process a single item"""
        row = self.main_df.iloc[idx]
        user_id = row[self.user_id_column]
        result = {'user_id': torch.tensor(user_id, dtype=torch.long)}
        
        # Process sequence columns
        for seq_col in self.sequence_columns:
            tensor_data = self._process_sequence(row[seq_col], seq_col)
            output_name = seq_col.replace(self.sequence_prefix, '', 1)
            
            # Special handling for main item sequence
            if 'item' in output_name.lower() and 'id' in output_name.lower():
                result['item_ids'] = tensor_data['tensor']
                result['length'] = tensor_data['length']
            else:
                result[output_name] = tensor_data['tensor']
        
        # Process static columns
        for col in self.static_columns:
            result[col] = torch.tensor(row[col], dtype=torch.long)
        
        # Process additional columns (user features from main data)
        for col in self.additional_columns:
            if col in self.main_df.columns:
                result[col] = torch.tensor(row[col], dtype=torch.long)
        
        return result
    
    def _process_sequence(self, data, column_name: str) -> Dict[str, torch.Tensor]:
        """Process sequence data with padding"""
        # Parse sequence
        if isinstance(data, str):
            try:
                sequence = ast.literal_eval(data)
            except:
                sequence = [data]
        else:
            sequence = list(data) if hasattr(data, '__iter__') else [data]
        
        # Apply transformations
        if self.ignore_last_n > 0:
            sequence = sequence[:-self.ignore_last_n]
        
        if self.sample_ratio < 1.0:
            keep_mask = torch.rand(len(sequence)) < self.sample_ratio
            sequence = [item for item, keep in zip(sequence, keep_mask) if keep]
        
        if not self.chronological:
            sequence = sequence[::-1]
        
        original_length = len(sequence)
        
        # Apply ID shifting for item sequences
        if ('item' in column_name.lower() or 'id' in column_name.lower()) and self.shift_id_by > 0:
            sequence = [item + self.shift_id_by for item in sequence]
        
        # Determine data type
        dtype = torch.float32 if any(kw in column_name.lower() for kw in ['rating', 'timestamp']) else torch.long
        pad_value = 0.0 if dtype == torch.float32 else 0
        
        # Convert to tensor and pad
        tensor = torch.tensor(sequence, dtype=dtype)
        if len(tensor) > self.max_sequence_length:
            tensor = tensor[:self.max_sequence_length]
            original_length = min(original_length, self.max_sequence_length)
        else:
            tensor = torch.nn.functional.pad(tensor, (0, self.max_sequence_length - len(tensor)), value=pad_value)
        
        return {'tensor': tensor, 'length': torch.tensor(original_length, dtype=torch.long)}
    
    def get_item_feature(self, feature_name: str, item_id: int):
        """Get item feature by name and ID"""
        return self.item_features.get_feature(feature_name, item_id)
    
    def get_user_feature(self, feature_name: str, user_id: int):
        """Get user feature by name and ID from main dataset"""
        # Find the row for this user_id
        user_row = self.main_df[self.main_df[self.user_id_column] == user_id]
        if user_row.empty:
            return None
        
        # Return the feature value for this user
        if feature_name in user_row.columns:
            return user_row.iloc[0][feature_name]
        else:
            return None


# Example usage
if __name__ == "__main__":
    dataset = RecoDataset(
        main_data_file="../generative-recommenders-pl/tmp/ml-1m/sasrec_format_by_user_train.csv",
        item_features_path="../generative-recommenders-pl/tmp/processed/ml-1m/movies_encoded.csv",
        additional_columns=['sex', 'age_group', 'occupation', 'zip_code']
    )
    
    print(f"Dataset size: {len(dataset)}")
    sample = dataset[0]
    print(f"Sample keys: {sample.keys()}")
    
    #print(f"User 1 sex feature: {dataset.get_user_feature('sex', 1)}")
    #print(f"User 1 age_group feature: {dataset.get_user_feature('age_group', 1)}")
    print(f"User 1 sex: {dataset.get_user_feature('sex', 1)}")           # 输出: 0 或 1
    print(f"User 1 age_group: {dataset.get_user_feature('age_group', 1)}") # 输出: 0-6
    print(f"User 1 occupation: {dataset.get_user_feature('occupation', 1)}") # 输出: 0-20
    print(f"User 1 zip_code: {dataset.get_user_feature('zip_code', 1)}")   # 输出: 0-3438
    
    # 获取电影特征（从item_features_path）
    print(f"Movie 1 year: {dataset.get_item_feature('year', 1)}")        # 输出: 1995    
    print(f"Movie 1 genres: {dataset.get_item_feature('genres', 1)}")  