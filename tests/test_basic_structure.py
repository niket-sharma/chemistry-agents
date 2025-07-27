"""
Basic structure tests that don't require heavy dependencies
"""

import pytest
import os
import sys
import json
from unittest.mock import Mock, patch

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


class TestBasicStructure:
    """Test basic code structure and imports"""
    
    def test_project_structure_exists(self):
        """Test that key project files exist"""
        base_dir = os.path.join(os.path.dirname(__file__), '..')
        
        required_files = [
            'src/chemistry_agents/__init__.py',
            'src/chemistry_agents/agents/__init__.py',
            'src/chemistry_agents/agents/base_agent.py',
            'src/chemistry_agents/agents/property_prediction_agent.py',
            'src/chemistry_agents/utils/__init__.py',
            'src/chemistry_agents/utils/api_integration.py',
            'configs/cpu_config.json',
            'examples/cpu_optimized_usage.py',
            'CPU_OPTIMIZATION_GUIDE.md'
        ]
        
        for file_path in required_files:
            full_path = os.path.join(base_dir, file_path)
            assert os.path.exists(full_path), f"Required file missing: {file_path}"
    
    def test_basic_imports(self):
        """Test that basic imports work without heavy dependencies"""
        # Test basic agent config
        from chemistry_agents.agents.base_agent import AgentConfig, PredictionResult
        
        config = AgentConfig()
        assert config.device == "cpu"
        assert config.batch_size <= 8
        assert config.cpu_optimization == True
        
        # Test prediction result
        result = PredictionResult(
            smiles="CCO",
            prediction=0.5,
            confidence=0.8,
            additional_info={"test": "data"}
        )
        
        assert result.smiles == "CCO"
        assert result.prediction == 0.5
        assert result.confidence == 0.8
        assert result.additional_info["test"] == "data"
    
    def test_api_integration_imports(self):
        """Test API integration imports work"""
        from chemistry_agents.utils.api_integration import APIConfig, CloudTrainingManager
        
        # Test API config
        config = APIConfig()
        assert config.provider == "huggingface"
        assert config.timeout == 30
        
        # Test cloud training manager
        manager = CloudTrainingManager()
        assert "colab" in manager.platforms
        assert "azure" in manager.platforms
    
    def test_cpu_config_file(self):
        """Test CPU configuration file is valid"""
        config_path = os.path.join(os.path.dirname(__file__), '..', 'configs', 'cpu_config.json')
        
        assert os.path.exists(config_path)
        
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        assert "agent_config" in config
        assert config["agent_config"]["device"] == "cpu"
        assert config["agent_config"]["cpu_optimization"] == True
        assert config["agent_config"]["batch_size"] <= 8


class TestCPUOptimizations:
    """Test CPU optimization features without requiring torch"""
    
    def test_cpu_config_defaults(self):
        """Test CPU configuration has correct defaults"""
        from chemistry_agents.agents.base_agent import AgentConfig
        
        config = AgentConfig()
        
        # Should default to CPU-friendly settings
        assert config.device == "cpu"
        assert config.batch_size <= 8
        assert config.cpu_optimization == True
        assert config.cache_predictions == True
    
    def test_api_configuration(self):
        """Test API configuration options"""
        from chemistry_agents.agents.base_agent import AgentConfig
        
        config = AgentConfig(
            use_api=True,
            api_provider="huggingface",
            api_key="test_key",
            model_name="test_model"
        )
        
        assert config.use_api == True
        assert config.api_provider == "huggingface"
        assert config.api_key == "test_key"
        assert config.model_name == "test_model"


class TestAPIIntegrationBasics:
    """Test API integration basics without external dependencies"""
    
    def test_hf_api_initialization(self):
        """Test Hugging Face API can be initialized"""
        from chemistry_agents.utils.api_integration import HuggingFaceInferenceAPI
        
        # Test with API key
        api = HuggingFaceInferenceAPI(api_key="test_key")
        assert api.api_key == "test_key"
        assert "Authorization" in api.headers
        
        # Test without API key
        api_no_key = HuggingFaceInferenceAPI()
        assert api_no_key.api_key is None or api_no_key.api_key == ""
    
    def test_cloud_training_manager(self):
        """Test cloud training manager basics"""
        from chemistry_agents.utils.api_integration import CloudTrainingManager
        
        manager = CloudTrainingManager()
        
        # Should have supported platforms
        assert hasattr(manager, 'platforms')
        assert "colab" in manager.platforms
        assert "azure" in manager.platforms
        
        # Should provide free options
        free_options = manager.get_free_options()
        assert isinstance(free_options, list)
        assert len(free_options) > 0
        
        # Each option should have required fields
        for option in free_options:
            assert "name" in option
            assert "url" in option
            assert "gpu" in option
    
    def test_google_colab_integration(self):
        """Test Google Colab integration features"""
        from chemistry_agents.utils.api_integration import GoogleColabIntegration
        
        # Test notebook generation
        notebook_json = GoogleColabIntegration.generate_colab_notebook(
            dataset_path="test.csv",
            model_name="test_model",
            target_property="test_prop"
        )
        
        # Should be valid JSON
        notebook = json.loads(notebook_json)
        assert "cells" in notebook
        assert isinstance(notebook["cells"], list)
        
        # Should contain key elements
        notebook_str = str(notebook)
        assert "test_model" in notebook_str
        assert "test_prop" in notebook_str


class TestUtilityFunctions:
    """Test utility functions work correctly"""
    
    def test_get_api_model_function(self):
        """Test get_api_model utility function"""
        from chemistry_agents.utils.api_integration import get_api_model, APIModelWrapper
        
        model = get_api_model(
            provider="huggingface",
            api_key="test_key",
            model_name="test_model"
        )
        
        assert isinstance(model, APIModelWrapper)
        assert model.config.provider == "huggingface"
        assert model.config.api_key == "test_key"
        assert model.config.model_name == "test_model"
    
    @patch('builtins.open', create=True)
    def test_setup_cloud_training_function(self, mock_open):
        """Test setup_cloud_training utility function"""
        from chemistry_agents.utils.api_integration import setup_cloud_training
        
        mock_file = Mock()
        mock_open.return_value.__enter__.return_value = mock_file
        
        # Should not raise an exception
        setup_cloud_training("colab", "test.csv")
        
        # Should have attempted to write a file
        mock_open.assert_called_once()


class TestScriptIntegration:
    """Test integration with training scripts"""
    
    def test_fine_tune_script_imports(self):
        """Test that fine-tune script can be imported"""
        # Add scripts to path
        scripts_path = os.path.join(os.path.dirname(__file__), '..', 'scripts')
        sys.path.insert(0, scripts_path)
        
        try:
            # Test that we can import the functions
            from fine_tune_transformer import parse_args, setup_device
            
            # Test setup_device function
            device = setup_device('cpu')
            # Would return torch.device('cpu') but we're testing without torch
            
        except ImportError as e:
            # Expected if torch is not available
            assert "torch" in str(e).lower() or "transformers" in str(e).lower()
    
    def test_cpu_optimization_guide_exists(self):
        """Test that CPU optimization guide exists and has content"""
        guide_path = os.path.join(os.path.dirname(__file__), '..', 'CPU_OPTIMIZATION_GUIDE.md')
        
        assert os.path.exists(guide_path)
        
        with open(guide_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Should contain key sections
        assert "CPU Optimization" in content
        assert "API Integration" in content
        assert "Google Colab" in content
        assert "Hugging Face" in content
        assert len(content) > 1000  # Should be substantial


if __name__ == "__main__":
    pytest.main([__file__, "-v"])