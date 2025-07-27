import torch
import numpy as np
from typing import List, Optional, Dict, Any
from .base_agent import BaseChemistryAgent, PredictionResult, AgentConfig
from ..models.molecular_predictor import MolecularPropertyPredictor, MolecularFeatureExtractor
from ..models.transformer_model import MolecularTransformer, SMILESProcessor

class PropertyPredictionAgent(BaseChemistryAgent):
    """
    General-purpose molecular property prediction agent
    Supports both traditional ML and transformer-based models
    """
    
    def __init__(self, 
                 config: Optional[AgentConfig] = None,
                 property_name: str = "molecular_property",
                 model_type: str = "neural_network",
                 transformer_model: str = "DeepChem/ChemBERTa-77M-MLM"):
        super().__init__(config)
        self.property_name = property_name
        self.model_type = model_type
        self.transformer_model = transformer_model
        
        # Initialize processors
        if model_type == "neural_network":
            self.feature_extractor = MolecularFeatureExtractor()
        elif model_type == "transformer":
            self.smiles_processor = SMILESProcessor()
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
    
    def load_model(self, model_path: Optional[str] = None) -> None:
        """Load the prediction model or setup API connection"""
        
        # Check if using API
        if self.config.use_api:
            self._setup_api_model()
            return
        
        model_path = model_path or self.config.model_path
        
        if not model_path:
            self.logger.warning("No model path provided, using default pre-trained model")
            self._load_default_model()
            return
        
        try:
            device = torch.device(self.config.device)
            
            if self.model_type == "neural_network":
                self.model = MolecularPropertyPredictor.load_model(model_path)
            elif self.model_type == "transformer":
                self.model = MolecularTransformer(model_name=self.transformer_model)
                # Load fine-tuned weights if available
                if model_path.endswith('.pt') or model_path.endswith('.pth'):
                    checkpoint = torch.load(model_path, map_location=device)
                    self.model.load_state_dict(checkpoint)
            
            self.model = self.model.to(device)
            self.model.eval()
            self.is_loaded = True
            
            if device.type == 'cpu' and self.config.cpu_optimization:
                self.logger.info("🔧 CPU optimization enabled for faster inference")
                # Enable CPU-specific optimizations
                torch.set_num_threads(4)  # Limit thread usage
            
            self.logger.info(f"Model loaded successfully from {model_path} on {device}")
            
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            if self.config.device == 'cuda':
                self.logger.info("💡 Try using CPU: set device='cpu' in config")
            raise
    
    def _setup_api_model(self) -> None:
        """Setup API-based model"""
        try:
            from ..utils.api_integration import APIConfig, APIModelWrapper
            
            api_config = APIConfig(
                provider=self.config.api_provider,
                api_key=self.config.api_key,
                model_name=self.config.model_name or self.transformer_model
            )
            
            self.api_model = APIModelWrapper(api_config)
            
            if self.api_model.is_available():
                self.is_loaded = True
                self.logger.info(f"✅ Connected to {self.config.api_provider} API")
                self.logger.info("📝 Note: API predictions may have different accuracy than local models")
            else:
                raise RuntimeError(f"API model not available: {api_config.model_name}")
                
        except ImportError:
            self.logger.error("API integration not available. Install requests package.")
            raise
        except Exception as e:
            self.logger.error(f"Failed to setup API model: {e}")
            self.logger.info("💡 Try local model or check API credentials")
            raise
    
    def _load_default_model(self) -> None:
        """Load default pre-trained model"""
        if self.model_type == "transformer":
            device = torch.device(self.config.device)
            self.model = MolecularTransformer(model_name=self.transformer_model)
            self.model = self.model.to(device)
            self.is_loaded = True
            self.logger.info(f"Loaded default transformer model on {device}")
        else:
            raise ValueError("Default model only available for transformer type")
    
    def predict_single(self, smiles: str) -> PredictionResult:
        """Make prediction for a single molecule"""
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        if not self.validate_smiles(smiles):
            return PredictionResult(
                smiles=smiles,
                prediction=0.0,
                confidence=0.0,
                additional_info={"error": "Invalid SMILES"}
            )
        
        try:
            # Use API if configured
            if self.config.use_api and hasattr(self, 'api_model'):
                predictions = self.api_model.predict([smiles])
                prediction = predictions[0] if predictions else 0.0
                confidence = 0.8  # Default confidence for API predictions
                model_info = f"{self.config.api_provider}_api"
            else:
                # Use local model
                if self.model_type == "neural_network":
                    prediction = self._predict_neural_network(smiles)
                elif self.model_type == "transformer":
                    prediction = self._predict_transformer(smiles)
                else:
                    raise ValueError(f"Unsupported model type: {self.model_type}")
                
                # Calculate confidence (simplified approach)
                confidence = min(1.0, 1.0 / (1.0 + abs(prediction)))
                model_info = self.model_type
            
            return PredictionResult(
                smiles=smiles,
                prediction=float(prediction),
                confidence=float(confidence),
                additional_info={
                    "property_name": self.property_name,
                    "model_type": model_info,
                    "device": self.config.device if not self.config.use_api else "api"
                }
            )
            
        except Exception as e:
            self.logger.error(f"Prediction failed for {smiles}: {e}")
            if self.config.use_api:
                self.logger.info("💡 API prediction failed. Try local model or check connection.")
            return PredictionResult(
                smiles=smiles,
                prediction=0.0,
                confidence=0.0,
                additional_info={"error": str(e)}
            )
    
    def _predict_neural_network(self, smiles: str) -> float:
        """Make prediction using neural network model"""
        features = self.feature_extractor.extract_features(smiles)
        if features is None:
            raise ValueError("Failed to extract molecular features")
        
        features = torch.FloatTensor(features).unsqueeze(0)
        
        with torch.no_grad():
            prediction = self.model(features)
            return prediction.item()
    
    def _predict_transformer(self, smiles: str) -> float:
        """Make prediction using transformer model"""
        processed_smiles, valid_mask = self.smiles_processor.process_smiles_batch([smiles])
        
        if not valid_mask[0]:
            raise ValueError("Failed to process SMILES")
        
        predictions = self.model.predict(processed_smiles)
        return predictions[0].item() if predictions.size > 0 else 0.0
    
    def _predict_batch_impl(self, smiles_list: List[str]) -> List[PredictionResult]:
        """Efficient batch prediction implementation"""
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        results = []
        
        try:
            if self.model_type == "neural_network":
                predictions = self._predict_batch_neural_network(smiles_list)
            elif self.model_type == "transformer":
                predictions = self._predict_batch_transformer(smiles_list)
            else:
                raise ValueError(f"Unsupported model type: {self.model_type}")
            
            for smiles, pred in zip(smiles_list, predictions):
                if pred is not None:
                    confidence = min(1.0, 1.0 / (1.0 + abs(pred)))
                    results.append(PredictionResult(
                        smiles=smiles,
                        prediction=float(pred),
                        confidence=float(confidence),
                        additional_info={
                            "property_name": self.property_name,
                            "model_type": self.model_type
                        }
                    ))
                else:
                    results.append(PredictionResult(
                        smiles=smiles,
                        prediction=0.0,
                        confidence=0.0,
                        additional_info={"error": "Prediction failed"}
                    ))
        
        except Exception as e:
            self.logger.error(f"Batch prediction failed: {e}")
            # Return error results for all molecules
            for smiles in smiles_list:
                results.append(PredictionResult(
                    smiles=smiles,
                    prediction=0.0,
                    confidence=0.0,
                    additional_info={"error": str(e)}
                ))
        
        return results
    
    def _predict_batch_neural_network(self, smiles_list: List[str]) -> List[Optional[float]]:
        """Batch prediction for neural network model"""
        features_batch, valid_mask = self.feature_extractor.extract_batch_features(smiles_list)
        
        if len(features_batch) == 0:
            return [None] * len(smiles_list)
        
        features_tensor = torch.FloatTensor(features_batch)
        
        with torch.no_grad():
            predictions = self.model(features_tensor).numpy().flatten()
        
        # Map predictions back to original order
        result_predictions = []
        pred_idx = 0
        
        for valid in valid_mask:
            if valid:
                result_predictions.append(predictions[pred_idx])
                pred_idx += 1
            else:
                result_predictions.append(None)
        
        return result_predictions
    
    def _predict_batch_transformer(self, smiles_list: List[str]) -> List[Optional[float]]:
        """Batch prediction for transformer model"""
        processed_smiles, valid_mask = self.smiles_processor.process_smiles_batch(smiles_list)
        
        if not processed_smiles:
            return [None] * len(smiles_list)
        
        predictions = self.model.predict(processed_smiles).flatten()
        
        # Map predictions back to original order
        result_predictions = []
        pred_idx = 0
        
        for valid in valid_mask:
            if valid:
                result_predictions.append(predictions[pred_idx])
                pred_idx += 1
            else:
                result_predictions.append(None)
        
        return result_predictions
    
    def set_property_name(self, property_name: str) -> None:
        """Set the name of the property being predicted"""
        self.property_name = property_name
        self.logger.info(f"Property name set to: {property_name}")
    
    def get_supported_properties(self) -> List[str]:
        """Get list of supported molecular properties"""
        return [
            "logP", "solubility", "toxicity", "bioavailability",
            "molecular_weight", "pka", "permeability", "clearance",
            "half_life", "bioactivity", "selectivity", "potency"
        ]