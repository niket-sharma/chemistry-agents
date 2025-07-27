import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
import numpy as np
from typing import List, Dict, Optional, Tuple
import re

class MolecularTransformer(nn.Module):
    """
    Transformer-based model for molecular property prediction using SMILES
    """
    
    def __init__(self,
                 model_name: str = "DeepChem/ChemBERTa-77M-MLM",
                 max_length: int = 512,
                 hidden_dim: int = 256,
                 num_layers: int = 2,
                 output_dim: int = 1,
                 dropout_rate: float = 0.1,
                 freeze_transformer: bool = False):
        super(MolecularTransformer, self).__init__()
        
        self.model_name = model_name
        self.max_length = max_length
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim
        self.dropout_rate = dropout_rate
        
        # Load pre-trained transformer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.transformer = AutoModel.from_pretrained(model_name)
        
        # Freeze transformer weights if specified
        if freeze_transformer:
            for param in self.transformer.parameters():
                param.requires_grad = False
        
        # Get transformer hidden size
        transformer_hidden_size = self.transformer.config.hidden_size
        
        # Prediction head
        layers = []
        prev_dim = transformer_hidden_size
        
        for _ in range(num_layers):
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.prediction_head = nn.Sequential(*layers)
    
    def forward(self, input_ids, attention_mask):
        """Forward pass through the transformer model"""
        # Get transformer outputs
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        
        # Use CLS token representation or mean pooling
        if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
            pooled_output = outputs.pooler_output
        else:
            # Mean pooling over sequence length
            last_hidden_state = outputs.last_hidden_state
            pooled_output = torch.mean(last_hidden_state, dim=1)
        
        # Prediction
        predictions = self.prediction_head(pooled_output)
        return predictions
    
    def predict(self, smiles_list: List[str]) -> np.ndarray:
        """Make predictions on SMILES strings"""
        self.eval()
        
        # Tokenize SMILES
        inputs = self.tokenizer(
            smiles_list,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        with torch.no_grad():
            predictions = self.forward(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask']
            )
        
        return predictions.numpy()

class SMILESProcessor:
    """Process and augment SMILES strings for better model performance"""
    
    def __init__(self):
        self.canonical = True
        self.augment = True
    
    def canonicalize_smiles(self, smiles: str) -> Optional[str]:
        """Convert SMILES to canonical form"""
        try:
            from rdkit import Chem
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
            return Chem.MolToSmiles(mol, canonical=True)
        except:
            return None
    
    def augment_smiles(self, smiles: str, num_augmentations: int = 5) -> List[str]:
        """Generate augmented SMILES using random permutations"""
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
        except:
            return [smiles]
    
    def clean_smiles(self, smiles: str) -> str:
        """Clean and standardize SMILES string"""
        # Remove common artifacts
        smiles = re.sub(r'\s+', '', smiles)  # Remove whitespace
        smiles = smiles.strip()
        
        # Remove salts (simple approach)
        if '.' in smiles:
            parts = smiles.split('.')
            # Keep the longest part (usually the main molecule)
            smiles = max(parts, key=len)
        
        return smiles
    
    def process_smiles_batch(self, 
                           smiles_list: List[str], 
                           canonicalize: bool = True,
                           clean: bool = True) -> Tuple[List[str], List[bool]]:
        """Process a batch of SMILES strings"""
        processed_smiles = []
        valid_mask = []
        
        for smiles in smiles_list:
            try:
                if clean:
                    smiles = self.clean_smiles(smiles)
                
                if canonicalize:
                    canonical_smiles = self.canonicalize_smiles(smiles)
                    if canonical_smiles is not None:
                        processed_smiles.append(canonical_smiles)
                        valid_mask.append(True)
                    else:
                        valid_mask.append(False)
                else:
                    processed_smiles.append(smiles)
                    valid_mask.append(True)
            except:
                valid_mask.append(False)
        
        return processed_smiles, valid_mask
    
    def validate_smiles(self, smiles: str) -> bool:
        """Validate if SMILES string is chemically valid"""
        try:
            from rdkit import Chem
            mol = Chem.MolFromSmiles(smiles)
            return mol is not None
        except:
            return False

class MolecularPropertyDataset(torch.utils.data.Dataset):
    """Dataset class for molecular property prediction"""
    
    def __init__(self, 
                 smiles_list: List[str], 
                 properties: np.ndarray,
                 tokenizer,
                 max_length: int = 512,
                 processor: Optional[SMILESProcessor] = None):
        self.smiles_list = smiles_list
        self.properties = properties
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.processor = processor or SMILESProcessor()
        
        # Process SMILES
        self.processed_smiles, self.valid_mask = self.processor.process_smiles_batch(smiles_list)
        
        # Filter valid samples
        self.valid_indices = [i for i, valid in enumerate(self.valid_mask) if valid]
        
    def __len__(self):
        return len(self.valid_indices)
    
    def __getitem__(self, idx):
        actual_idx = self.valid_indices[idx]
        smiles = self.processed_smiles[actual_idx]
        property_value = self.properties[actual_idx]
        
        # Tokenize SMILES
        encoding = self.tokenizer(
            smiles,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(property_value, dtype=torch.float)
        }