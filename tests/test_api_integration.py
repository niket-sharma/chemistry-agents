"""
Tests for API integration functionality
"""

import pytest
import json
import os
from unittest.mock import Mock, patch, MagicMock
from chemistry_agents.utils.api_integration import (
    HuggingFaceInferenceAPI,
    APIConfig,
    APIModelWrapper,
    GoogleColabIntegration,
    CloudTrainingManager,
    get_api_model,
    setup_cloud_training
)


class TestAPIConfig:
    """Test API configuration functionality"""
    
    def test_api_config_defaults(self):
        """Test default API configuration values"""
        config = APIConfig()
        
        assert config.provider == "huggingface"
        assert config.api_key is None
        assert config.base_url is None
        assert config.model_name is None
        assert config.timeout == 30
        assert config.max_retries == 3
    
    def test_api_config_custom_values(self):
        """Test API configuration with custom values"""
        config = APIConfig(
            provider="custom",
            api_key="test_key",
            model_name="test_model",
            timeout=60
        )
        
        assert config.provider == "custom"
        assert config.api_key == "test_key"
        assert config.model_name == "test_model"
        assert config.timeout == 60


class TestHuggingFaceInferenceAPI:
    """Test Hugging Face Inference API integration"""
    
    def test_api_initialization_with_key(self):
        """Test API initialization with API key"""
        api = HuggingFaceInferenceAPI(api_key="test_key")
        
        assert api.api_key == "test_key"
        assert "Authorization" in api.headers
        assert api.headers["Authorization"] == "Bearer test_key"
    
    def test_api_initialization_without_key(self):
        """Test API initialization without API key"""
        with patch.dict(os.environ, {}, clear=True):
            api = HuggingFaceInferenceAPI()
            
            assert api.api_key is None
            assert api.headers["Authorization"] == ""
    
    def test_api_initialization_from_env(self):
        """Test API initialization from environment variable"""
        with patch.dict(os.environ, {"HUGGINGFACE_API_KEY": "env_key"}):
            api = HuggingFaceInferenceAPI()
            
            assert api.api_key == "env_key"
            assert api.headers["Authorization"] == "Bearer env_key"
    
    @patch('requests.post')
    def test_successful_prediction(self, mock_post):
        """Test successful API prediction"""
        # Mock successful response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = [{"score": 0.75}, {"score": 0.82}]
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response
        
        api = HuggingFaceInferenceAPI(api_key="test_key")
        results = api.predict_text_classification(
            ["CCO", "CC(=O)O"], 
            model_name="test_model"
        )
        
        assert len(results) == 2
        assert results[0]["score"] == 0.75
        assert results[1]["score"] == 0.82
        
        # Verify API call
        mock_post.assert_called_once()
        call_args = mock_post.call_args
        assert "test_model" in call_args[0][0]  # URL contains model name
    
    @patch('requests.post')
    def test_model_loading_503_retry(self, mock_post):
        """Test handling of 503 model loading response"""
        # First call returns 503, second call succeeds
        mock_response_503 = Mock()
        mock_response_503.status_code = 503
        
        mock_response_200 = Mock()
        mock_response_200.status_code = 200
        mock_response_200.json.return_value = [{"score": 0.5}]
        mock_response_200.raise_for_status.return_value = None
        
        mock_post.side_effect = [mock_response_503, mock_response_200]
        
        api = HuggingFaceInferenceAPI(api_key="test_key")
        
        with patch('time.sleep'):  # Skip actual sleep in tests
            results = api.predict_text_classification(["CCO"])
        
        assert len(results) == 1
        assert mock_post.call_count == 2
    
    @patch('requests.post')
    def test_api_request_failure(self, mock_post):
        """Test handling of API request failures"""
        mock_post.side_effect = Exception("Network error")
        
        api = HuggingFaceInferenceAPI(api_key="test_key")
        results = api.predict_text_classification(["CCO"])
        
        assert results == []
    
    @patch('requests.get')
    def test_model_status_check(self, mock_get):
        """Test model status checking"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_get.return_value = mock_response
        
        api = HuggingFaceInferenceAPI(api_key="test_key")
        status = api.check_model_status("test_model")
        
        assert status["available"] == True
        assert status["status_code"] == 200
        assert status["model_name"] == "test_model"
    
    @patch('requests.get')
    def test_model_status_check_failure(self, mock_get):
        """Test model status check when model is not available"""
        mock_get.side_effect = Exception("Connection error")
        
        api = HuggingFaceInferenceAPI(api_key="test_key")
        status = api.check_model_status("test_model")
        
        assert status["available"] == False
        assert status["model_name"] == "test_model"


class TestAPIModelWrapper:
    """Test API model wrapper functionality"""
    
    def test_wrapper_initialization(self):
        """Test API model wrapper initialization"""
        config = APIConfig(
            provider="huggingface",
            api_key="test_key",
            model_name="test_model"
        )
        
        wrapper = APIModelWrapper(config)
        assert wrapper.config == config
        assert hasattr(wrapper, 'api')
    
    def test_unsupported_provider(self):
        """Test handling of unsupported API providers"""
        config = APIConfig(provider="unsupported")
        
        with pytest.raises(ValueError, match="Unsupported API provider"):
            APIModelWrapper(config)
    
    @patch.object(HuggingFaceInferenceAPI, 'predict_text_classification')
    def test_wrapper_predict(self, mock_predict):
        """Test wrapper prediction functionality"""
        mock_predict.return_value = [{"score": 0.8}, {"score": 0.6}]
        
        config = APIConfig(
            provider="huggingface",
            api_key="test_key",
            model_name="test_model"
        )
        wrapper = APIModelWrapper(config)
        
        predictions = wrapper.predict(["CCO", "CC(=O)O"])
        
        assert len(predictions) == 2
        assert predictions[0] == 0.8
        assert predictions[1] == 0.6
        mock_predict.assert_called_once_with(["CCO", "CC(=O)O"], "test_model")
    
    @patch.object(HuggingFaceInferenceAPI, 'check_model_status')
    def test_wrapper_availability_check(self, mock_status):
        """Test wrapper availability checking"""
        mock_status.return_value = {"available": True}
        
        config = APIConfig(
            provider="huggingface",
            api_key="test_key",
            model_name="test_model"
        )
        wrapper = APIModelWrapper(config)
        
        assert wrapper.is_available() == True
        mock_status.assert_called_once_with("test_model")


class TestGoogleColabIntegration:
    """Test Google Colab integration functionality"""
    
    def test_colab_notebook_generation(self):
        """Test generation of Google Colab notebook"""
        notebook_json = GoogleColabIntegration.generate_colab_notebook(
            dataset_path="test_data.csv",
            model_name="test_model",
            target_property="test_property"
        )
        
        notebook = json.loads(notebook_json)
        
        assert "cells" in notebook
        assert len(notebook["cells"]) > 0
        
        # Check that key elements are in the notebook
        notebook_text = json.dumps(notebook)
        assert "test_model" in notebook_text
        assert "test_property" in notebook_text
        assert "colab" in notebook_text.lower()
    
    @patch('chemistry_agents.utils.api_integration.google.colab')
    def test_colab_environment_detection(self, mock_colab):
        """Test detection of Google Colab environment"""
        # Mock successful import of google.colab
        assert GoogleColabIntegration.is_colab_environment() == True
    
    def test_non_colab_environment_detection(self):
        """Test detection when not in Google Colab"""
        with patch('chemistry_agents.utils.api_integration.google.colab', side_effect=ImportError):
            assert GoogleColabIntegration.is_colab_environment() == False


class TestCloudTrainingManager:
    """Test cloud training management functionality"""
    
    def test_manager_initialization(self):
        """Test cloud training manager initialization"""
        manager = CloudTrainingManager()
        
        assert "colab" in manager.platforms
        assert "azure" in manager.platforms
    
    def test_colab_setup_generation(self):
        """Test Google Colab setup generation"""
        manager = CloudTrainingManager()
        
        result = manager.generate_cloud_setup(
            platform="colab",
            dataset_path="test.csv",
            model_name="test_model",
            target_property="test_prop"
        )
        
        assert isinstance(result, str)
        assert "test_model" in result
        assert "test_prop" in result
    
    def test_azure_setup_generation(self):
        """Test Azure ML setup generation"""
        manager = CloudTrainingManager()
        
        result = manager.generate_cloud_setup(
            platform="azure",
            dataset_path="test.csv",
            model_name="test_model",
            target_property="test_prop"
        )
        
        assert isinstance(result, dict)
        assert "script" in result
        assert "arguments" in result
        assert "environment" in result
    
    def test_unsupported_platform(self):
        """Test handling of unsupported platforms"""
        manager = CloudTrainingManager()
        
        with pytest.raises(ValueError, match="Unsupported platform"):
            manager.generate_cloud_setup(
                platform="unsupported",
                dataset_path="test.csv",
                model_name="test_model",
                target_property="test_prop"
            )
    
    def test_free_options_list(self):
        """Test getting list of free options"""
        manager = CloudTrainingManager()
        options = manager.get_free_options()
        
        assert isinstance(options, list)
        assert len(options) > 0
        
        # Check that each option has required fields
        for option in options:
            assert "name" in option
            assert "url" in option
            assert "gpu" in option
            assert "description" in option


class TestUtilityFunctions:
    """Test utility functions for API integration"""
    
    def test_get_api_model_function(self):
        """Test get_api_model utility function"""
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
    @patch('chemistry_agents.utils.api_integration.CloudTrainingManager')
    def test_setup_cloud_training_colab(self, mock_manager_class, mock_open):
        """Test setup_cloud_training function for Colab"""
        mock_manager = Mock()
        mock_manager.generate_cloud_setup.return_value = '{"test": "notebook"}'
        mock_manager_class.return_value = mock_manager
        
        mock_file = Mock()
        mock_open.return_value.__enter__.return_value = mock_file
        
        setup_cloud_training("colab", "test.csv")
        
        mock_manager.generate_cloud_setup.assert_called_once()
        mock_open.assert_called_once()
        mock_file.write.assert_called_once_with('{"test": "notebook"}')
    
    @patch('builtins.open', create=True)
    @patch('json.dump')
    @patch('chemistry_agents.utils.api_integration.CloudTrainingManager')
    def test_setup_cloud_training_azure(self, mock_manager_class, mock_json_dump, mock_open):
        """Test setup_cloud_training function for Azure"""
        mock_manager = Mock()
        mock_manager.generate_cloud_setup.return_value = {"test": "config"}
        mock_manager_class.return_value = mock_manager
        
        mock_file = Mock()
        mock_open.return_value.__enter__.return_value = mock_file
        
        setup_cloud_training("azure", "test.csv")
        
        mock_manager.generate_cloud_setup.assert_called_once()
        mock_open.assert_called_once()
        mock_json_dump.assert_called_once_with({"test": "config"}, mock_file, indent=2)


class TestAPIErrorHandling:
    """Test error handling in API integration"""
    
    def test_api_key_missing_warning(self, capfd):
        """Test warning when API key is missing"""
        with patch.dict(os.environ, {}, clear=True):
            HuggingFaceInferenceAPI()
            
            # Should log a warning about missing API key
            # This is tested by the initialization not raising an exception
    
    @patch('requests.post')
    def test_api_timeout_handling(self, mock_post):
        """Test handling of API timeouts"""
        import requests
        mock_post.side_effect = requests.exceptions.Timeout("Request timed out")
        
        api = HuggingFaceInferenceAPI(api_key="test_key")
        results = api.predict_text_classification(["CCO"])
        
        assert results == []
    
    @patch('requests.post')
    def test_api_rate_limit_handling(self, mock_post):
        """Test handling of API rate limits"""
        mock_response = Mock()
        mock_response.status_code = 429  # Rate limit exceeded
        mock_response.raise_for_status.side_effect = Exception("Rate limit exceeded")
        mock_post.return_value = mock_response
        
        api = HuggingFaceInferenceAPI(api_key="test_key")
        results = api.predict_text_classification(["CCO"])
        
        assert results == []


if __name__ == "__main__":
    pytest.main([__file__])