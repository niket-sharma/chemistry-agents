from .data_processing import DataProcessor, MolecularDataset
from .evaluation import ModelEvaluator, MetricsCalculator
from .model_hub import ModelHub, get_chemistry_models, suggest_model_for_property, get_model_info

# Import feature extractors from models
try:
    from ..models.molecular_predictor import MolecularFeatureExtractor
    from ..models.transformer_model import SMILESProcessor
    _EXTRACTORS_AVAILABLE = True
except ImportError:
    _EXTRACTORS_AVAILABLE = False

__all__ = [
    "DataProcessor",
    "MolecularDataset", 
    "ModelEvaluator",
    "MetricsCalculator",
    "ModelHub",
    "get_chemistry_models",
    "suggest_model_for_property", 
    "get_model_info"
]

# Add extractors if available
if _EXTRACTORS_AVAILABLE:
    __all__.extend(["MolecularFeatureExtractor", "SMILESProcessor"])