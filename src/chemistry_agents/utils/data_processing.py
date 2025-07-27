import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Any, Tuple, Optional, Union
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import logging
import os
import pickle

class DataProcessor:
    """
    Comprehensive data processing utilities for molecular datasets
    """
    
    def __init__(self, 
                 scaler_type: str = "standard",
                 remove_duplicates: bool = True,
                 handle_missing: str = "drop"):
        self.scaler_type = scaler_type
        self.remove_duplicates = remove_duplicates
        self.handle_missing = handle_missing
        self.scaler = None
        self.logger = logging.getLogger(__name__)
        
        # Initialize scaler
        if scaler_type == "standard":
            self.scaler = StandardScaler()
        elif scaler_type == "minmax":
            self.scaler = MinMaxScaler()
        else:
            raise ValueError(f"Unsupported scaler type: {scaler_type}")
    
    def load_dataset(self, 
                    filepath: str, 
                    smiles_column: str = "smiles",
                    target_column: str = "target") -> pd.DataFrame:
        """Load dataset from various file formats"""
        try:
            if filepath.endswith('.csv'):
                df = pd.read_csv(filepath)
            elif filepath.endswith('.xlsx') or filepath.endswith('.xls'):
                df = pd.read_excel(filepath)
            elif filepath.endswith('.json'):
                df = pd.read_json(filepath)
            elif filepath.endswith('.tsv'):
                df = pd.read_csv(filepath, sep='\t')
            else:
                raise ValueError(f"Unsupported file format: {filepath}")
            
            # Validate required columns
            if smiles_column not in df.columns:
                raise ValueError(f"SMILES column '{smiles_column}' not found")
            if target_column not in df.columns:
                raise ValueError(f"Target column '{target_column}' not found")
            
            self.logger.info(f"Loaded dataset with {len(df)} samples")
            return df
            
        except Exception as e:
            self.logger.error(f"Failed to load dataset: {e}")
            raise
    
    def clean_dataset(self, 
                     df: pd.DataFrame,
                     smiles_column: str = "smiles",
                     target_column: str = "target") -> pd.DataFrame:
        """Clean and preprocess dataset"""
        initial_size = len(df)
        
        # Handle missing values
        if self.handle_missing == "drop":
            df = df.dropna(subset=[smiles_column, target_column])
        elif self.handle_missing == "fill":
            df[smiles_column] = df[smiles_column].fillna("")
            df[target_column] = df[target_column].fillna(df[target_column].mean())
        
        # Remove duplicates
        if self.remove_duplicates:
            df = df.drop_duplicates(subset=[smiles_column])
        
        # Clean SMILES strings
        df = self._clean_smiles_strings(df, smiles_column)
        
        # Remove invalid SMILES
        df = self._filter_valid_smiles(df, smiles_column)
        
        final_size = len(df)
        self.logger.info(f"Dataset cleaned: {initial_size} -> {final_size} samples")
        
        return df
    
    def _clean_smiles_strings(self, df: pd.DataFrame, smiles_column: str) -> pd.DataFrame:
        """Clean SMILES strings"""
        def clean_smiles(smiles):
            if pd.isna(smiles):
                return None
            
            # Remove whitespace
            smiles = str(smiles).strip()
            
            # Remove common artifacts
            smiles = smiles.replace(' ', '')
            
            # Handle salts (keep largest fragment)
            if '.' in smiles:
                fragments = smiles.split('.')
                smiles = max(fragments, key=len)
            
            return smiles
        
        df[smiles_column] = df[smiles_column].apply(clean_smiles)
        return df.dropna(subset=[smiles_column])
    
    def _filter_valid_smiles(self, df: pd.DataFrame, smiles_column: str) -> pd.DataFrame:
        """Filter out invalid SMILES strings"""
        def is_valid_smiles(smiles):
            try:
                from rdkit import Chem
                mol = Chem.MolFromSmiles(smiles)
                return mol is not None
            except:
                return False
        
        valid_mask = df[smiles_column].apply(is_valid_smiles)
        valid_df = df[valid_mask].copy()
        
        invalid_count = len(df) - len(valid_df)
        if invalid_count > 0:
            self.logger.warning(f"Removed {invalid_count} invalid SMILES")
        
        return valid_df
    
    def split_dataset(self, 
                     df: pd.DataFrame,
                     test_size: float = 0.2,
                     val_size: float = 0.1,
                     random_state: int = 42,
                     stratify_column: Optional[str] = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split dataset into train, validation, and test sets"""
        
        # First split: train+val vs test
        if stratify_column and stratify_column in df.columns:
            stratify_data = df[stratify_column]
        else:
            stratify_data = None
        
        train_val, test = train_test_split(
            df, 
            test_size=test_size, 
            random_state=random_state,
            stratify=stratify_data
        )
        
        # Second split: train vs val
        if val_size > 0:
            val_size_adjusted = val_size / (1 - test_size)
            
            if stratify_column and stratify_column in train_val.columns:
                stratify_data = train_val[stratify_column]
            else:
                stratify_data = None
            
            train, val = train_test_split(
                train_val,
                test_size=val_size_adjusted,
                random_state=random_state,
                stratify=stratify_data
            )
        else:
            train = train_val
            val = pd.DataFrame()
        
        self.logger.info(f"Dataset split - Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")
        
        return train, val, test
    
    def create_molecular_descriptors(self, 
                                   df: pd.DataFrame,
                                   smiles_column: str = "smiles") -> pd.DataFrame:
        """Create molecular descriptors for all molecules"""
        from ..models.molecular_predictor import MolecularFeatureExtractor
        
        extractor = MolecularFeatureExtractor()
        
        descriptors_list = []
        valid_indices = []
        
        for idx, smiles in enumerate(df[smiles_column]):
            descriptors = extractor.extract_features(smiles)
            if descriptors is not None:
                descriptors_list.append(descriptors)
                valid_indices.append(idx)
        
        if descriptors_list:
            # Create descriptor DataFrame
            descriptor_array = np.vstack(descriptors_list)
            descriptor_columns = [f"descriptor_{i}" for i in range(descriptor_array.shape[1])]
            descriptor_df = pd.DataFrame(descriptor_array, columns=descriptor_columns)
            
            # Combine with original data
            result_df = df.iloc[valid_indices].reset_index(drop=True)
            result_df = pd.concat([result_df, descriptor_df], axis=1)
            
            self.logger.info(f"Created {len(descriptor_columns)} molecular descriptors")
            return result_df
        else:
            self.logger.error("No valid molecular descriptors could be created")
            return df
    
    def normalize_features(self, 
                          train_features: np.ndarray,
                          val_features: Optional[np.ndarray] = None,
                          test_features: Optional[np.ndarray] = None) -> Tuple[np.ndarray, ...]:
        """Normalize features using fitted scaler"""
        
        # Fit scaler on training data
        train_normalized = self.scaler.fit_transform(train_features)
        
        results = [train_normalized]
        
        # Transform validation and test data
        if val_features is not None:
            val_normalized = self.scaler.transform(val_features)
            results.append(val_normalized)
        
        if test_features is not None:
            test_normalized = self.scaler.transform(test_features)
            results.append(test_normalized)
        
        return tuple(results)
    
    def save_scaler(self, filepath: str) -> None:
        """Save fitted scaler"""
        if self.scaler is None:
            raise ValueError("Scaler not fitted yet")
        
        with open(filepath, 'wb') as f:
            pickle.dump(self.scaler, f)
        
        self.logger.info(f"Scaler saved to {filepath}")
    
    def load_scaler(self, filepath: str) -> None:
        """Load fitted scaler"""
        with open(filepath, 'rb') as f:
            self.scaler = pickle.load(f)
        
        self.logger.info(f"Scaler loaded from {filepath}")
    
    def create_data_loaders(self,
                           train_dataset: 'MolecularDataset',
                           val_dataset: Optional['MolecularDataset'] = None,
                           test_dataset: Optional['MolecularDataset'] = None,
                           batch_size: int = 32,
                           num_workers: int = 0) -> Dict[str, DataLoader]:
        """Create PyTorch data loaders"""
        
        data_loaders = {
            'train': DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers
            )
        }
        
        if val_dataset is not None:
            data_loaders['val'] = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers
            )
        
        if test_dataset is not None:
            data_loaders['test'] = DataLoader(
                test_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers
            )
        
        return data_loaders
    
    def analyze_dataset(self, df: pd.DataFrame, target_column: str = "target") -> Dict[str, Any]:
        """Analyze dataset statistics"""
        analysis = {
            "dataset_size": len(df),
            "target_statistics": {},
            "missing_values": df.isnull().sum().to_dict(),
            "data_types": df.dtypes.to_dict()
        }
        
        if target_column in df.columns:
            target_data = df[target_column]
            analysis["target_statistics"] = {
                "mean": float(target_data.mean()),
                "std": float(target_data.std()),
                "min": float(target_data.min()),
                "max": float(target_data.max()),
                "median": float(target_data.median()),
                "quartiles": {
                    "Q1": float(target_data.quantile(0.25)),
                    "Q3": float(target_data.quantile(0.75))
                }
            }
        
        return analysis

class MolecularDataset(Dataset):
    """
    PyTorch Dataset class for molecular data
    """
    
    def __init__(self,
                 smiles_list: List[str],
                 targets: np.ndarray,
                 feature_extractor = None,
                 transform=None):
        self.smiles_list = smiles_list
        self.targets = torch.FloatTensor(targets)
        self.feature_extractor = feature_extractor
        self.transform = transform
        
        # Pre-extract features if extractor provided
        if feature_extractor is not None:
            self.features = self._extract_all_features()
        else:
            self.features = None
    
    def _extract_all_features(self):
        """Pre-extract all molecular features"""
        features_list = []
        valid_indices = []
        
        for idx, smiles in enumerate(self.smiles_list):
            features = self.feature_extractor.extract_features(smiles)
            if features is not None:
                features_list.append(features)
                valid_indices.append(idx)
        
        if features_list:
            # Filter invalid samples
            self.smiles_list = [self.smiles_list[i] for i in valid_indices]
            self.targets = self.targets[valid_indices]
            return torch.FloatTensor(np.vstack(features_list))
        else:
            return None
    
    def __len__(self):
        return len(self.smiles_list)
    
    def __getitem__(self, idx):
        smiles = self.smiles_list[idx]
        target = self.targets[idx]
        
        if self.features is not None:
            features = self.features[idx]
        else:
            # Extract features on-the-fly (slower but more memory efficient)
            if self.feature_extractor is not None:
                features = self.feature_extractor.extract_features(smiles)
                features = torch.FloatTensor(features) if features is not None else torch.zeros(1)
            else:
                features = torch.zeros(1)  # Placeholder
        
        if self.transform:
            features = self.transform(features)
        
        return {
            'features': features,
            'target': target,
            'smiles': smiles
        }

class DataAugmentation:
    """
    Data augmentation techniques for molecular datasets
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def augment_smiles(self, smiles: str, num_augmentations: int = 5) -> List[str]:
        """Generate augmented SMILES using random enumeration"""
        try:
            from rdkit import Chem
            
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return [smiles]
            
            augmented = [smiles]  # Include original
            
            for _ in range(num_augmentations):
                try:
                    # Generate random SMILES
                    random_smiles = Chem.MolToSmiles(mol, doRandom=True)
                    if random_smiles and random_smiles not in augmented:
                        augmented.append(random_smiles)
                except:
                    continue
            
            return augmented
            
        except Exception as e:
            self.logger.error(f"SMILES augmentation failed: {e}")
            return [smiles]
    
    def augment_dataset(self, 
                       df: pd.DataFrame,
                       smiles_column: str = "smiles",
                       target_column: str = "target",
                       augmentation_factor: int = 3) -> pd.DataFrame:
        """Augment entire dataset"""
        
        augmented_data = []
        
        for _, row in df.iterrows():
            smiles = row[smiles_column]
            target = row[target_column]
            
            # Generate augmented SMILES
            augmented_smiles = self.augment_smiles(smiles, augmentation_factor)
            
            # Add all augmentations
            for aug_smiles in augmented_smiles:
                augmented_row = row.copy()
                augmented_row[smiles_column] = aug_smiles
                augmented_data.append(augmented_row)
        
        augmented_df = pd.DataFrame(augmented_data)
        
        self.logger.info(f"Dataset augmented: {len(df)} -> {len(augmented_df)} samples")
        
        return augmented_df