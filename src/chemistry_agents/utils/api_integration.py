"""
API Integration utilities for external inference and training services
"""

import os
import requests
import json
import time
from typing import List, Dict, Any, Optional, Union
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class APIConfig:
    """Configuration for API services"""
    provider: str = "huggingface"
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    model_name: Optional[str] = None
    timeout: int = 30
    max_retries: int = 3

class HuggingFaceInferenceAPI:
    """
    Hugging Face Inference API integration for model inference without local GPU
    """
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("HUGGINGFACE_API_KEY")
        self.base_url = "https://api-inference.huggingface.co/models"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}" if self.api_key else ""
        }
        
        if not self.api_key:
            logger.warning("⚠️  No Hugging Face API key provided. Some features may be limited.")
            logger.info("💡 Get a free API key at: https://huggingface.co/settings/tokens")
    
    def predict_text_classification(self, 
                                  texts: List[str], 
                                  model_name: str = "DeepChem/ChemBERTa-77M-MLM",
                                  wait_for_model: bool = True) -> List[Dict[str, Any]]:
        """
        Use HF Inference API for text classification/regression
        """
        url = f"{self.base_url}/{model_name}"
        
        payload = {
            "inputs": texts,
            "options": {"wait_for_model": wait_for_model}
        }
        
        try:
            response = requests.post(url, headers=self.headers, json=payload, timeout=30)
            
            if response.status_code == 503:
                logger.info("🔄 Model is loading, waiting...")
                time.sleep(10)
                return self.predict_text_classification(texts, model_name, wait_for_model)
            
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.RequestException as e:
            logger.error(f"❌ HF API request failed: {e}")
            return []
    
    def get_embeddings(self, 
                      texts: List[str], 
                      model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> Optional[List[List[float]]]:
        """
        Get embeddings from Hugging Face models
        """
        url = f"{self.base_url}/{model_name}"
        
        payload = {
            "inputs": texts,
            "options": {"wait_for_model": True}
        }
        
        try:
            response = requests.post(url, headers=self.headers, json=payload, timeout=30)
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.RequestException as e:
            logger.error(f"❌ Embedding request failed: {e}")
            return None
    
    def check_model_status(self, model_name: str) -> Dict[str, Any]:
        """
        Check if a model is available and loaded
        """
        url = f"{self.base_url}/{model_name}"
        
        try:
            response = requests.get(url, headers=self.headers, timeout=10)
            return {
                "available": response.status_code == 200,
                "status_code": response.status_code,
                "model_name": model_name
            }
        except:
            return {"available": False, "model_name": model_name}

class GoogleColabIntegration:
    """
    Utilities for Google Colab integration
    """
    
    @staticmethod
    def generate_colab_notebook(dataset_path: str, 
                               model_name: str = "DeepChem/ChemBERTa-77M-MLM",
                               target_property: str = "solubility") -> str:
        """
        Generate a Google Colab notebook for training
        """
        notebook_content = {
            "cells": [
                {
                    "cell_type": "markdown",
                    "source": [
                        "# Chemistry Agents - GPU Training on Colab\n",
                        f"Training {model_name} for {target_property} prediction\n",
                        "\n",
                        "**Free GPU provided by Google Colab**"
                    ]
                },
                {
                    "cell_type": "code",
                    "source": [
                        "# Install chemistry-agents\n",
                        "!pip install chemistry-agents torch transformers rdkit-pypi\n",
                        "!git clone https://github.com/yourusername/chemistry-agents.git\n",
                        "%cd chemistry-agents"
                    ]
                },
                {
                    "cell_type": "code",
                    "source": [
                        "# Upload your dataset\n",
                        "from google.colab import files\n",
                        "uploaded = files.upload()\n",
                        "dataset_file = list(uploaded.keys())[0]"
                    ]
                },
                {
                    "cell_type": "code",
                    "source": [
                        "# Run fine-tuning with GPU\n",
                        f"!python scripts/fine_tune_transformer.py \\\n",
                        f"    --data_path $dataset_file \\\n",
                        f"    --model_name {model_name} \\\n",
                        f"    --smiles_column smiles \\\n",
                        f"    --target_column {target_property} \\\n",
                        "    --device cuda \\\n",
                        "    --batch_size 16 \\\n",
                        "    --epochs 10 \\\n",
                        "    --output_dir ./models"
                    ]
                },
                {
                    "cell_type": "code",
                    "source": [
                        "# Download trained model\n",
                        "from google.colab import files\n",
                        "!zip -r trained_model.zip ./models\n",
                        "files.download('trained_model.zip')"
                    ]
                }
            ]
        }
        
        return json.dumps(notebook_content, indent=2)
    
    @staticmethod
    def is_colab_environment() -> bool:
        """Check if running in Google Colab"""
        try:
            import google.colab
            return True
        except:
            return False

class AzureMLIntegration:
    """
    Azure ML integration for cloud training
    """
    
    @staticmethod
    def generate_azure_config(dataset_path: str, model_name: str) -> Dict[str, Any]:
        """
        Generate Azure ML configuration
        """
        return {
            "script": "fine_tune_transformer.py",
            "arguments": [
                "--data_path", dataset_path,
                "--model_name", model_name,
                "--device", "cuda",
                "--batch_size", "32",
                "--epochs", "20"
            ],
            "environment": {
                "docker": "mcr.microsoft.com/azureml/pytorch-1.9-ubuntu18.04-py37-cuda11-gpu",
                "conda_dependencies": {
                    "channels": ["conda-forge"],
                    "dependencies": [
                        "python=3.8",
                        "pytorch",
                        "transformers",
                        "rdkit",
                        "scikit-learn",
                        "pandas",
                        "numpy"
                    ]
                }
            },
            "compute": {
                "target": "gpu-cluster",
                "instance_type": "STANDARD_NC6"
            }
        }

class APIModelWrapper:
    """
    Wrapper to use API models as if they were local PyTorch models
    """
    
    def __init__(self, config: APIConfig):
        self.config = config
        
        if config.provider == "huggingface":
            self.api = HuggingFaceInferenceAPI(config.api_key)
        else:
            raise ValueError(f"Unsupported API provider: {config.provider}")
    
    def predict(self, smiles_list: List[str]) -> List[float]:
        """
        Make predictions using external API
        """
        if self.config.provider == "huggingface":
            # For now, we'll use a classification approach
            # You can extend this to use fine-tuned models
            results = self.api.predict_text_classification(
                smiles_list, 
                self.config.model_name or "DeepChem/ChemBERTa-77M-MLM"
            )
            
            # Extract predictions (this is a simplified example)
            predictions = []
            for result in results:
                if isinstance(result, list) and result:
                    # Take the score of the first prediction
                    predictions.append(result[0].get('score', 0.0))
                else:
                    predictions.append(0.0)
            
            return predictions
        
        return []
    
    def is_available(self) -> bool:
        """Check if API is available"""
        if self.config.provider == "huggingface":
            status = self.api.check_model_status(
                self.config.model_name or "DeepChem/ChemBERTa-77M-MLM"
            )
            return status.get("available", False)
        return False

class CloudTrainingManager:
    """
    Manager for cloud training options
    """
    
    def __init__(self):
        self.platforms = {
            "colab": GoogleColabIntegration(),
            "azure": AzureMLIntegration()
        }
    
    def generate_cloud_setup(self, 
                           platform: str,
                           dataset_path: str,
                           model_name: str,
                           target_property: str) -> Union[str, Dict[str, Any]]:
        """
        Generate setup files for cloud platforms
        """
        if platform == "colab":
            return self.platforms["colab"].generate_colab_notebook(
                dataset_path, model_name, target_property
            )
        elif platform == "azure":
            return self.platforms["azure"].generate_azure_config(
                dataset_path, model_name
            )
        else:
            raise ValueError(f"Unsupported platform: {platform}")
    
    def get_free_options(self) -> List[Dict[str, str]]:
        """
        Get list of free cloud training options
        """
        return [
            {
                "name": "Google Colab",
                "url": "https://colab.research.google.com/",
                "gpu": "Tesla T4 (free tier)",
                "time_limit": "12 hours",
                "description": "Free GPU access with some limitations"
            },
            {
                "name": "Kaggle Notebooks",
                "url": "https://kaggle.com/notebooks",
                "gpu": "Tesla P100",
                "time_limit": "9 hours/week",
                "description": "Free GPU with weekly quota"
            },
            {
                "name": "Hugging Face Spaces",
                "url": "https://huggingface.co/spaces",
                "gpu": "Limited free GPU",
                "time_limit": "Varies",
                "description": "Host models and demos"
            }
        ]

# Convenience functions
def get_api_model(provider: str = "huggingface", 
                  api_key: Optional[str] = None,
                  model_name: Optional[str] = None) -> APIModelWrapper:
    """Get an API-based model wrapper"""
    config = APIConfig(
        provider=provider,
        api_key=api_key,
        model_name=model_name
    )
    return APIModelWrapper(config)

def setup_cloud_training(platform: str, dataset_path: str) -> None:
    """Setup cloud training for the given platform"""
    manager = CloudTrainingManager()
    
    if platform == "colab":
        notebook = manager.generate_cloud_setup(platform, dataset_path, 
                                               "DeepChem/ChemBERTa-77M-MLM", "solubility")
        
        # Save notebook
        with open("chemistry_agents_colab.ipynb", "w") as f:
            f.write(notebook)
        
        print("✅ Google Colab notebook generated: chemistry_agents_colab.ipynb")
        print("📤 Upload this to https://colab.research.google.com/")
    
    elif platform == "azure":
        config = manager.generate_cloud_setup(platform, dataset_path,
                                             "DeepChem/ChemBERTa-77M-MLM", "solubility")
        
        with open("azure_ml_config.json", "w") as f:
            json.dump(config, f, indent=2)
        
        print("✅ Azure ML configuration generated: azure_ml_config.json")

def show_free_alternatives():
    """Show free cloud alternatives for GPU training"""
    manager = CloudTrainingManager()
    options = manager.get_free_options()
    
    print("🆓 Free GPU Training Options:")
    print("=" * 50)
    
    for option in options:
        print(f"\n📍 {option['name']}")
        print(f"   URL: {option['url']}")
        print(f"   GPU: {option['gpu']}")
        print(f"   Time Limit: {option['time_limit']}")
        print(f"   Description: {option['description']}")
    
    print("\n💡 Recommendation: Start with Google Colab for free GPU training!")