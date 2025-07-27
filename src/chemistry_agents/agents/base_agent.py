from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union
import logging
import numpy as np
from dataclasses import dataclass
import json

@dataclass
class PredictionResult:
    """Container for prediction results"""
    smiles: str
    prediction: float
    confidence: Optional[float] = None
    additional_info: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'smiles': self.smiles,
            'prediction': self.prediction,
            'confidence': self.confidence,
            'additional_info': self.additional_info
        }

@dataclass
class AgentConfig:
    """Configuration for chemistry agents"""
    model_path: Optional[str] = None
    batch_size: int = 8  # Reduced default for CPU compatibility
    confidence_threshold: float = 0.5
    log_level: str = "INFO"
    cache_predictions: bool = True
    device: str = "cpu"  # Default to CPU
    use_api: bool = False
    api_provider: str = "huggingface"
    api_key: Optional[str] = None
    model_name: Optional[str] = None
    cpu_optimization: bool = True
    
class BaseChemistryAgent(ABC):
    """
    Base class for all chemistry prediction agents
    """
    
    def __init__(self, config: Optional[AgentConfig] = None):
        self.config = config or AgentConfig()
        self.logger = self._setup_logger()
        self.model = None
        self.is_loaded = False
        self.prediction_cache = {} if self.config.cache_predictions else None
        
    def _setup_logger(self) -> logging.Logger:
        """Setup logger for the agent"""
        logger = logging.getLogger(f"{self.__class__.__name__}")
        logger.setLevel(getattr(logging, self.config.log_level))
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    @abstractmethod
    def load_model(self, model_path: Optional[str] = None) -> None:
        """Load the prediction model"""
        pass
    
    @abstractmethod
    def predict_single(self, smiles: str) -> PredictionResult:
        """Make prediction for a single molecule"""
        pass
    
    def predict_batch(self, smiles_list: List[str]) -> List[PredictionResult]:
        """Make predictions for a batch of molecules"""
        results = []
        
        # Check cache first
        if self.prediction_cache:
            cached_results = []
            uncached_smiles = []
            uncached_indices = []
            
            for i, smiles in enumerate(smiles_list):
                if smiles in self.prediction_cache:
                    cached_results.append((i, self.prediction_cache[smiles]))
                else:
                    uncached_smiles.append(smiles)
                    uncached_indices.append(i)
        else:
            uncached_smiles = smiles_list
            uncached_indices = list(range(len(smiles_list)))
            cached_results = []
        
        # Process uncached molecules
        if uncached_smiles:
            new_results = self._predict_batch_impl(uncached_smiles)
            
            # Cache new results
            if self.prediction_cache:
                for smiles, result in zip(uncached_smiles, new_results):
                    self.prediction_cache[smiles] = result
        else:
            new_results = []
        
        # Combine cached and new results
        all_results = [None] * len(smiles_list)
        
        # Place cached results
        for i, result in cached_results:
            all_results[i] = result
        
        # Place new results
        for i, result in zip(uncached_indices, new_results):
            all_results[i] = result
        
        return all_results
    
    def _predict_batch_impl(self, smiles_list: List[str]) -> List[PredictionResult]:
        """Default batch prediction implementation using single predictions"""
        return [self.predict_single(smiles) for smiles in smiles_list]
    
    def validate_smiles(self, smiles: str) -> bool:
        """Validate SMILES string"""
        try:
            from rdkit import Chem
            mol = Chem.MolFromSmiles(smiles)
            return mol is not None
        except:
            return False
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model"""
        return {
            'model_type': self.__class__.__name__,
            'is_loaded': self.is_loaded,
            'config': self.config.__dict__
        }
    
    def save_predictions(self, results: List[PredictionResult], filepath: str) -> None:
        """Save prediction results to file"""
        data = [result.to_dict() for result in results]
        
        if filepath.endswith('.json'):
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
        elif filepath.endswith('.csv'):
            import pandas as pd
            df = pd.DataFrame(data)
            df.to_csv(filepath, index=False)
        else:
            raise ValueError("Unsupported file format. Use .json or .csv")
        
        self.logger.info(f"Saved {len(results)} predictions to {filepath}")
    
    def load_predictions(self, filepath: str) -> List[PredictionResult]:
        """Load prediction results from file"""
        if filepath.endswith('.json'):
            with open(filepath, 'r') as f:
                data = json.load(f)
        elif filepath.endswith('.csv'):
            import pandas as pd
            df = pd.read_csv(filepath)
            data = df.to_dict('records')
        else:
            raise ValueError("Unsupported file format. Use .json or .csv")
        
        results = []
        for item in data:
            result = PredictionResult(
                smiles=item['smiles'],
                prediction=item['prediction'],
                confidence=item.get('confidence'),
                additional_info=item.get('additional_info')
            )
            results.append(result)
        
        self.logger.info(f"Loaded {len(results)} predictions from {filepath}")
        return results
    
    def clear_cache(self) -> None:
        """Clear prediction cache"""
        if self.prediction_cache:
            self.prediction_cache.clear()
            self.logger.info("Prediction cache cleared")
    
    def get_cache_size(self) -> int:
        """Get number of cached predictions"""
        return len(self.prediction_cache) if self.prediction_cache else 0