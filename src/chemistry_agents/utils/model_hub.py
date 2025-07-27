"""
Utilities for working with Hugging Face Model Hub and pre-trained models
"""

import os
import json
from typing import Dict, List, Optional, Tuple
import logging
from transformers import AutoConfig, AutoTokenizer, AutoModel

logger = logging.getLogger(__name__)

class ModelHub:
    """
    Utility class for managing Hugging Face models and local model cache
    """
    
    def __init__(self, cache_dir: Optional[str] = None):
        self.cache_dir = cache_dir or os.path.expanduser("~/.cache/chemistry_agents/models")
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Registry of known chemistry models
        self.chemistry_models = {
            "chemberta-77m": "DeepChem/ChemBERTa-77M-MLM",
            "chemberta-10m": "DeepChem/ChemBERTa-10M-MLM", 
            "chemberta-5m": "DeepChem/ChemBERTa-5M-MLM",
            "molt5-small": "laituan245/molt5-small",
            "molt5-base": "laituan245/molt5-base",
            "scibert": "allenai/scibert_scivocab_uncased",
            "pubmedbert": "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract"
        }
        
        # Property-specific model recommendations
        self.recommended_models = {
            "solubility": ["chemberta-77m", "chemberta-10m"],
            "toxicity": ["chemberta-77m", "scibert"],
            "bioactivity": ["chemberta-77m", "pubmedbert"],
            "logp": ["chemberta-77m", "chemberta-10m"],
            "molecular_weight": ["chemberta-5m", "chemberta-10m"],
            "general": ["chemberta-77m"]
        }
    
    def get_model_name(self, model_key: str) -> str:
        """
        Get full Hugging Face model name from short key
        
        Args:
            model_key: Short model key (e.g., "chemberta-77m") or full HF name
            
        Returns:
            Full Hugging Face model name
        """
        if model_key in self.chemistry_models:
            return self.chemistry_models[model_key]
        
        # If it's already a full name or path, return as-is
        return model_key
    
    def list_available_models(self) -> Dict[str, str]:
        """List all available chemistry models"""
        return self.chemistry_models.copy()
    
    def get_recommended_models(self, property_name: str) -> List[str]:
        """
        Get recommended models for a specific property
        
        Args:
            property_name: Name of the molecular property
            
        Returns:
            List of recommended model keys
        """
        return self.recommended_models.get(property_name.lower(), 
                                         self.recommended_models["general"])
    
    def get_model_info(self, model_name: str) -> Dict[str, any]:
        """
        Get information about a model
        
        Args:
            model_name: Model name or key
            
        Returns:
            Dictionary with model information
        """
        full_name = self.get_model_name(model_name)
        
        try:
            config = AutoConfig.from_pretrained(full_name)
            
            info = {
                "model_name": full_name,
                "model_type": config.model_type,
                "hidden_size": getattr(config, 'hidden_size', None),
                "num_layers": getattr(config, 'num_hidden_layers', None),
                "vocab_size": getattr(config, 'vocab_size', None),
                "max_position_embeddings": getattr(config, 'max_position_embeddings', None),
            }
            
            # Estimate parameters (rough)
            if info["hidden_size"] and info["num_layers"]:
                # Very rough estimation
                params = info["hidden_size"] * info["num_layers"] * 1000
                if params < 1e6:
                    info["estimated_params"] = f"{params/1e3:.0f}K"
                else:
                    info["estimated_params"] = f"{params/1e6:.0f}M"
            
            return info
            
        except Exception as e:
            logger.warning(f"Could not get info for {full_name}: {e}")
            return {
                "model_name": full_name,
                "error": str(e)
            }
    
    def verify_model_access(self, model_name: str) -> bool:
        """
        Verify that a model can be accessed from Hugging Face Hub
        
        Args:
            model_name: Model name or key
            
        Returns:
            True if model is accessible
        """
        full_name = self.get_model_name(model_name)
        
        try:
            # Try to load config (lightweight check)
            AutoConfig.from_pretrained(full_name)
            return True
        except Exception as e:
            logger.warning(f"Model {full_name} not accessible: {e}")
            return False
    
    def download_model(self, model_name: str, local_path: Optional[str] = None) -> str:
        """
        Download and cache a model locally
        
        Args:
            model_name: Model name or key
            local_path: Optional local path to save model
            
        Returns:
            Path to downloaded model
        """
        full_name = self.get_model_name(model_name)
        
        if local_path is None:
            # Create local path in cache
            safe_name = full_name.replace("/", "_")
            local_path = os.path.join(self.cache_dir, safe_name)
        
        try:
            logger.info(f"Downloading {full_name} to {local_path}")
            
            # Download tokenizer and model
            tokenizer = AutoTokenizer.from_pretrained(full_name)
            model = AutoModel.from_pretrained(full_name)
            
            # Save locally
            os.makedirs(local_path, exist_ok=True)
            tokenizer.save_pretrained(local_path)
            model.save_pretrained(local_path)
            
            # Save metadata
            metadata = {
                "original_name": full_name,
                "download_date": str(pd.Timestamp.now()),
                "local_path": local_path
            }
            
            with open(os.path.join(local_path, "metadata.json"), "w") as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"Model downloaded successfully to {local_path}")
            return local_path
            
        except Exception as e:
            logger.error(f"Failed to download {full_name}: {e}")
            raise
    
    def list_local_models(self) -> List[Dict[str, str]]:
        """List locally cached models"""
        local_models = []
        
        if not os.path.exists(self.cache_dir):
            return local_models
        
        for item in os.listdir(self.cache_dir):
            item_path = os.path.join(self.cache_dir, item)
            
            if os.path.isdir(item_path):
                metadata_path = os.path.join(item_path, "metadata.json")
                
                if os.path.exists(metadata_path):
                    try:
                        with open(metadata_path, "r") as f:
                            metadata = json.load(f)
                        
                        local_models.append({
                            "local_name": item,
                            "original_name": metadata.get("original_name", "unknown"),
                            "local_path": item_path,
                            "download_date": metadata.get("download_date", "unknown")
                        })
                    except:
                        # Metadata file corrupted, skip
                        continue
        
        return local_models
    
    def suggest_model(self, 
                     property_name: str, 
                     dataset_size: Optional[int] = None,
                     speed_priority: bool = False) -> str:
        """
        Suggest the best model for a given task
        
        Args:
            property_name: Target property name
            dataset_size: Size of training dataset (if known)
            speed_priority: Whether to prioritize inference speed
            
        Returns:
            Recommended model name
        """
        candidates = self.get_recommended_models(property_name)
        
        if speed_priority:
            # Prefer smaller models for speed
            speed_preference = ["chemberta-5m", "chemberta-10m", "chemberta-77m"]
            for model in speed_preference:
                if model in candidates:
                    return self.get_model_name(model)
        
        if dataset_size is not None:
            if dataset_size < 1000:
                # Small dataset - prefer smaller pre-trained model
                small_models = ["chemberta-10m", "chemberta-5m"]
                for model in small_models:
                    if model in candidates:
                        return self.get_model_name(model)
            elif dataset_size > 10000:
                # Large dataset - can use larger model
                large_models = ["chemberta-77m"]
                for model in large_models:
                    if model in candidates:
                        return self.get_model_name(model)
        
        # Default recommendation
        if candidates:
            return self.get_model_name(candidates[0])
        else:
            return self.get_model_name("chemberta-77m")  # Fallback
    
    def create_model_comparison(self, model_names: List[str]) -> Dict[str, Dict]:
        """
        Create a comparison of different models
        
        Args:
            model_names: List of model names/keys to compare
            
        Returns:
            Dictionary with comparison data
        """
        comparison = {}
        
        for model_name in model_names:
            info = self.get_model_info(model_name)
            accessible = self.verify_model_access(model_name)
            
            comparison[model_name] = {
                **info,
                "accessible": accessible,
                "recommended_for": []
            }
            
            # Find what properties this model is recommended for
            for prop, recommended in self.recommended_models.items():
                if model_name in recommended or self.get_model_name(model_name) in [self.get_model_name(r) for r in recommended]:
                    comparison[model_name]["recommended_for"].append(prop)
        
        return comparison

# Convenience functions
def get_chemistry_models() -> Dict[str, str]:
    """Get available chemistry models"""
    hub = ModelHub()
    return hub.list_available_models()

def suggest_model_for_property(property_name: str, **kwargs) -> str:
    """Suggest best model for a property"""
    hub = ModelHub()
    return hub.suggest_model(property_name, **kwargs)

def get_model_info(model_name: str) -> Dict[str, any]:
    """Get information about a model"""
    hub = ModelHub()
    return hub.get_model_info(model_name)

# Global model hub instance
_model_hub = None

def get_model_hub() -> ModelHub:
    """Get global model hub instance"""
    global _model_hub
    if _model_hub is None:
        _model_hub = ModelHub()
    return _model_hub