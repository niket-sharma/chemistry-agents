import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
import joblib
import os

class MolecularPropertyPredictor(nn.Module):
    """
    Neural network model for predicting molecular properties from chemical descriptors
    """
    
    def __init__(self, 
                 input_dim: int = 200, 
                 hidden_dims: List[int] = [512, 256, 128], 
                 output_dim: int = 1,
                 dropout_rate: float = 0.3,
                 activation: str = 'relu'):
        super(MolecularPropertyPredictor, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.dropout_rate = dropout_rate
        
        # Build network layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                self._get_activation(activation),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self._initialize_weights()
    
    def _get_activation(self, activation: str) -> nn.Module:
        """Get activation function"""
        activations = {
            'relu': nn.ReLU(),
            'leaky_relu': nn.LeakyReLU(),
            'elu': nn.ELU(),
            'gelu': nn.GELU(),
            'tanh': nn.Tanh()
        }
        return activations.get(activation, nn.ReLU())
    
    def _initialize_weights(self):
        """Initialize network weights using Xavier initialization"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network"""
        return self.network(x)
    
    def predict(self, molecular_features: np.ndarray) -> np.ndarray:
        """Make predictions on molecular features"""
        self.eval()
        with torch.no_grad():
            if isinstance(molecular_features, np.ndarray):
                molecular_features = torch.FloatTensor(molecular_features)
            
            if len(molecular_features.shape) == 1:
                molecular_features = molecular_features.unsqueeze(0)
            
            predictions = self.forward(molecular_features)
            return predictions.numpy()
    
    def save_model(self, filepath: str):
        """Save model state dict"""
        torch.save({
            'model_state_dict': self.state_dict(),
            'input_dim': self.input_dim,
            'hidden_dims': self.hidden_dims,
            'output_dim': self.output_dim,
            'dropout_rate': self.dropout_rate
        }, filepath)
    
    @classmethod
    def load_model(cls, filepath: str) -> 'MolecularPropertyPredictor':
        """Load model from file"""
        checkpoint = torch.load(filepath, map_location='cpu')
        model = cls(
            input_dim=checkpoint['input_dim'],
            hidden_dims=checkpoint['hidden_dims'],
            output_dim=checkpoint['output_dim'],
            dropout_rate=checkpoint['dropout_rate']
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        return model

class MolecularFeatureExtractor:
    """Extract molecular descriptors from SMILES strings"""
    
    def __init__(self):
        self.descriptor_functions = [
            Descriptors.MolWt,
            Descriptors.MolLogP,
            Descriptors.NumHDonors,
            Descriptors.NumHAcceptors,
            Descriptors.TPSA,
            Descriptors.NumRotatableBonds,
            Descriptors.NumAromaticRings,
            Descriptors.FractionCsp3,
            Descriptors.BertzCT,
            rdMolDescriptors.CalcNumLipinskiHBD,
            rdMolDescriptors.CalcNumLipinskiHBA,
            rdMolDescriptors.CalcNumRings,
            rdMolDescriptors.CalcNumHeteroatoms,
            rdMolDescriptors.CalcNumSaturatedRings,
            rdMolDescriptors.CalcNumAliphaticRings
        ]
    
    def smiles_to_mol(self, smiles: str) -> Optional[Chem.Mol]:
        """Convert SMILES to RDKit molecule object"""
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
            return Chem.AddHs(mol)
        except:
            return None
    
    def extract_features(self, smiles: str) -> Optional[np.ndarray]:
        """Extract molecular descriptors from SMILES"""
        mol = self.smiles_to_mol(smiles)
        if mol is None:
            return None
        
        features = []
        for descriptor_func in self.descriptor_functions:
            try:
                value = descriptor_func(mol)
                features.append(value if value is not None else 0.0)
            except:
                features.append(0.0)
        
        # Add fingerprint features
        fingerprint = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
        features.extend(list(fingerprint.ToBitString()))
        
        return np.array(features, dtype=np.float32)
    
    def extract_batch_features(self, smiles_list: List[str]) -> Tuple[np.ndarray, List[bool]]:
        """Extract features for a batch of SMILES"""
        features_list = []
        valid_mask = []
        
        for smiles in smiles_list:
            features = self.extract_features(smiles)
            if features is not None:
                features_list.append(features)
                valid_mask.append(True)
            else:
                valid_mask.append(False)
        
        if features_list:
            return np.vstack(features_list), valid_mask
        else:
            return np.array([]), valid_mask