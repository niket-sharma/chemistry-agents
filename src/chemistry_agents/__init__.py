"""
Chemistry Agents: AI-powered molecular property prediction and analysis
CPU-optimized with API integration support
"""

__version__ = "0.1.0"
__author__ = "Chemistry Agents Team"

from .models import MolecularPropertyPredictor
from .agents import PropertyPredictionAgent, SolubilityAgent, ToxicityAgent, DrugDiscoveryAgent, UnitOperationsAgent
from .agents.base_agent import AgentConfig
from .agents.unit_operations_agent import UnitOperationConfig
from .utils import MolecularFeatureExtractor, SMILESProcessor

# API Integration (optional)
try:
    from .utils.api_integration import (
        get_api_model, 
        setup_cloud_training, 
        show_free_alternatives,
        HuggingFaceInferenceAPI
    )
    _API_AVAILABLE = True
except ImportError:
    _API_AVAILABLE = False

__all__ = [
    "MolecularPropertyPredictor",
    "PropertyPredictionAgent", 
    "SolubilityAgent",
    "ToxicityAgent",
    "DrugDiscoveryAgent",
    "UnitOperationsAgent",
    "AgentConfig",
    "UnitOperationConfig",
    "MolecularFeatureExtractor",
    "SMILESProcessor"
]

# Add API utilities if available
if _API_AVAILABLE:
    __all__.extend([
        "get_api_model",
        "setup_cloud_training", 
        "show_free_alternatives",
        "HuggingFaceInferenceAPI"
    ])

# CPU optimization message
import sys
if 'torch' in sys.modules:
    import torch
    if not torch.cuda.is_available():
        print("🔧 Chemistry Agents: CPU mode enabled")
        print("💡 For faster performance, see CPU_OPTIMIZATION_GUIDE.md")