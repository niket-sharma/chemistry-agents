"""
Tests for cloud training utilities and script functionality
"""

import pytest
import json
import os
import tempfile
from unittest.mock import Mock, patch, MagicMock, call
from chemistry_agents.utils.api_integration import (
    CloudTrainingManager,
    GoogleColabIntegration,
    AzureMLIntegration,
    setup_cloud_training,
    show_free_alternatives
)


class TestCloudTrainingUtilities:
    """Test cloud training utility functions"""
    
    def test_show_free_alternatives(self, capsys):
        """Test showing free alternatives function"""
        show_free_alternatives()
        
        captured = capsys.readouterr()
        output = captured.out
        
        # Should mention key free platforms
        assert "Google Colab" in output
        assert "Kaggle" in output
        assert "Hugging Face" in output
        assert "Free" in output or "free" in output
    
    @patch('builtins.open', create=True)
    @patch('chemistry_agents.utils.api_integration.CloudTrainingManager')
    def test_setup_cloud_training_creates_files(self, mock_manager_class, mock_open):
        """Test that setup_cloud_training creates appropriate files"""
        mock_manager = Mock()
        mock_manager.generate_cloud_setup.return_value = '{"test": "notebook"}'
        mock_manager_class.return_value = mock_manager
        
        mock_file = Mock()
        mock_open.return_value.__enter__.return_value = mock_file
        
        setup_cloud_training("colab", "test_data.csv")
        
        # Should have called file operations
        mock_open.assert_called_once()
        mock_file.write.assert_called_once()
        
        # Should have generated the setup
        mock_manager.generate_cloud_setup.assert_called_once_with(
            "colab", "test_data.csv", "DeepChem/ChemBERTa-77M-MLM", "solubility"
        )


class TestGoogleColabIntegration:
    """Test Google Colab specific integration"""
    
    def test_colab_notebook_structure(self):
        """Test that generated Colab notebook has correct structure"""
        notebook_json = GoogleColabIntegration.generate_colab_notebook(
            dataset_path="data.csv",
            model_name="test_model",
            target_property="solubility"
        )
        
        notebook = json.loads(notebook_json)
        
        # Should be valid notebook structure
        assert "cells" in notebook
        assert isinstance(notebook["cells"], list)
        assert len(notebook["cells"]) > 0
        
        # Each cell should have required fields
        for cell in notebook["cells"]:
            assert "cell_type" in cell
            assert "source" in cell
            assert cell["cell_type"] in ["markdown", "code"]
    
    def test_colab_notebook_content_includes_requirements(self):
        """Test that notebook includes installation requirements"""
        notebook_json = GoogleColabIntegration.generate_colab_notebook(
            dataset_path="data.csv",
            model_name="test_model",
            target_property="test_prop"
        )
        
        # Convert to string to search content
        notebook_content = notebook_json.lower()
        
        # Should include key installation steps
        assert "install" in notebook_content
        assert "chemistry-agents" in notebook_content or "chemistry_agents" in notebook_content
        assert "torch" in notebook_content
        assert "rdkit" in notebook_content
    
    def test_colab_notebook_includes_training_command(self):
        """Test that notebook includes proper training command"""
        model_name = "DeepChem/ChemBERTa-10M-MLM"
        target_prop = "toxicity"
        
        notebook_json = GoogleColabIntegration.generate_colab_notebook(
            dataset_path="data.csv",
            model_name=model_name,
            target_property=target_prop
        )
        
        # Should include the model name and property in training command
        assert model_name in notebook_json
        assert target_prop in notebook_json
        assert "fine_tune_transformer.py" in notebook_json
    
    @patch('chemistry_agents.utils.api_integration.google.colab')
    def test_colab_environment_detection_positive(self, mock_colab):
        """Test positive detection of Colab environment"""
        # Mock successful import
        mock_colab.return_value = Mock()
        
        result = GoogleColabIntegration.is_colab_environment()
        assert result == True
    
    def test_colab_environment_detection_negative(self):
        """Test negative detection of Colab environment"""
        # Test when google.colab is not available
        with patch('builtins.__import__', side_effect=ImportError):
            result = GoogleColabIntegration.is_colab_environment()
            assert result == False


class TestAzureMLIntegration:
    """Test Azure ML specific integration"""
    
    def test_azure_config_structure(self):
        """Test that Azure ML config has correct structure"""
        config = AzureMLIntegration.generate_azure_config(
            dataset_path="data.csv",
            model_name="test_model"
        )
        
        # Should have required Azure ML fields
        assert "script" in config
        assert "arguments" in config
        assert "environment" in config
        assert "compute" in config
        
        # Script should point to fine-tuning script
        assert config["script"] == "fine_tune_transformer.py"
        
        # Arguments should include dataset and model
        assert "data.csv" in str(config["arguments"])
        assert "test_model" in str(config["arguments"])
    
    def test_azure_config_environment_setup(self):
        """Test Azure ML environment configuration"""
        config = AzureMLIntegration.generate_azure_config(
            dataset_path="data.csv",
            model_name="test_model"
        )
        
        env = config["environment"]
        
        # Should have Docker configuration
        assert "docker" in env
        
        # Should have conda dependencies
        assert "conda_dependencies" in env
        conda_deps = env["conda_dependencies"]
        assert "dependencies" in conda_deps
        
        # Should include required packages
        deps = conda_deps["dependencies"]
        assert "pytorch" in deps or "torch" in deps
        assert "transformers" in deps
        assert "rdkit" in deps
    
    def test_azure_config_compute_setup(self):
        """Test Azure ML compute configuration"""
        config = AzureMLIntegration.generate_azure_config(
            dataset_path="data.csv",
            model_name="test_model"
        )
        
        compute = config["compute"]
        
        # Should have compute target and instance type
        assert "target" in compute
        assert "instance_type" in compute
        
        # Should specify GPU instance
        assert "gpu" in compute["instance_type"].lower() or "nc" in compute["instance_type"]


class TestCloudTrainingManager:
    """Test the main cloud training manager"""
    
    def test_manager_platform_support(self):
        """Test that manager supports required platforms"""
        manager = CloudTrainingManager()
        
        assert "colab" in manager.platforms
        assert "azure" in manager.platforms
        
        # Should be able to generate setup for supported platforms
        for platform in ["colab", "azure"]:
            result = manager.generate_cloud_setup(
                platform=platform,
                dataset_path="test.csv",
                model_name="test_model",
                target_property="test_prop"
            )
            assert result is not None
    
    def test_manager_unsupported_platform(self):
        """Test handling of unsupported platforms"""
        manager = CloudTrainingManager()
        
        with pytest.raises(ValueError, match="Unsupported platform"):
            manager.generate_cloud_setup(
                platform="unsupported_platform",
                dataset_path="test.csv",
                model_name="test_model",
                target_property="test_prop"
            )
    
    def test_manager_free_options_content(self):
        """Test that free options contain required information"""
        manager = CloudTrainingManager()
        options = manager.get_free_options()
        
        assert isinstance(options, list)
        assert len(options) > 0
        
        for option in options:
            # Each option should have required fields
            assert "name" in option
            assert "url" in option
            assert "gpu" in option
            assert "time_limit" in option
            assert "description" in option
            
            # Should contain reasonable values
            assert option["name"]  # Not empty
            assert option["url"].startswith("http")
            assert len(option["description"]) > 10


class TestFinetuneScriptIntegration:
    """Test integration with fine-tuning script functionality"""
    
    @patch('builtins.print')
    def test_cloud_training_flag_handling(self, mock_print):
        """Test handling of --cloud_training flag"""
        # Import the handler functions
        try:
            from chemistry_agents.scripts.fine_tune_transformer import handle_cloud_training
            
            # Mock args object
            mock_args = Mock()
            mock_args.data_path = "test.csv"
            
            with patch('chemistry_agents.utils.api_integration.setup_cloud_training'):
                with patch('chemistry_agents.utils.api_integration.show_free_alternatives'):
                    handle_cloud_training(mock_args)
                    
                    # Should have printed setup information
                    assert mock_print.called
                    
        except ImportError:
            # Skip if the script module is not importable in test environment
            pytest.skip("Fine-tune script not importable in test environment")
    
    @patch('builtins.print')
    @patch('os.path.exists')
    def test_api_flag_handling(self, mock_exists, mock_print):
        """Test handling of --use_api flag"""
        try:
            from chemistry_agents.scripts.fine_tune_transformer import handle_api_training
            
            mock_args = Mock()
            mock_args.api_key = None
            mock_args.data_path = "test.csv"
            
            # Test without API key
            handle_api_training(mock_args)
            
            # Should have printed alternatives
            assert mock_print.called
            
        except ImportError:
            pytest.skip("Fine-tune script not importable in test environment")


class TestCloudPlatformSpecifics:
    """Test platform-specific functionality"""
    
    def test_colab_specific_features(self):
        """Test Colab-specific features in generated notebooks"""
        notebook_json = GoogleColabIntegration.generate_colab_notebook(
            dataset_path="data.csv",
            model_name="test_model",
            target_property="test_prop"
        )
        
        # Should include Colab-specific features
        assert "google.colab" in notebook_json
        assert "files.upload" in notebook_json
        assert "files.download" in notebook_json
        
        # Should use Colab's file upload mechanism
        assert "uploaded" in notebook_json
    
    def test_azure_specific_features(self):
        """Test Azure ML specific configuration features"""
        config = AzureMLIntegration.generate_azure_config(
            dataset_path="data.csv", 
            model_name="test_model"
        )
        
        # Should include Azure-specific settings
        assert "azureml" in config["environment"]["docker"].lower()
        assert config["compute"]["target"] == "gpu-cluster"
        
        # Should specify CUDA-capable instance
        assert "NC" in config["compute"]["instance_type"]
    
    def test_platform_resource_specifications(self):
        """Test that platforms specify appropriate resources"""
        manager = CloudTrainingManager()
        free_options = manager.get_free_options()
        
        colab_option = next((opt for opt in free_options if "Colab" in opt["name"]), None)
        kaggle_option = next((opt for opt in free_options if "Kaggle" in opt["name"]), None)
        
        assert colab_option is not None
        assert kaggle_option is not None
        
        # Should specify GPU types
        assert "GPU" in colab_option["gpu"] or "Tesla" in colab_option["gpu"]
        assert "GPU" in kaggle_option["gpu"] or "Tesla" in kaggle_option["gpu"]
        
        # Should have time limits
        assert "hour" in colab_option["time_limit"]
        assert "hour" in kaggle_option["time_limit"] or "week" in kaggle_option["time_limit"]


class TestCloudTrainingWorkflow:
    """Test end-to-end cloud training workflow"""
    
    @patch('builtins.open', create=True)
    def test_colab_workflow_file_creation(self, mock_open):
        """Test complete Colab workflow creates correct files"""
        mock_file = Mock()
        mock_open.return_value.__enter__.return_value = mock_file
        
        setup_cloud_training("colab", "dataset.csv")
        
        # Should create the notebook file
        mock_open.assert_called_with("chemistry_agents_colab.ipynb", "w")
        mock_file.write.assert_called_once()
        
        # Verify the content is valid JSON
        written_content = mock_file.write.call_args[0][0]
        parsed_content = json.loads(written_content)
        assert "cells" in parsed_content
    
    @patch('builtins.open', create=True)
    @patch('json.dump')
    def test_azure_workflow_file_creation(self, mock_json_dump, mock_open):
        """Test complete Azure workflow creates correct files"""
        mock_file = Mock()
        mock_open.return_value.__enter__.return_value = mock_file
        
        setup_cloud_training("azure", "dataset.csv")
        
        # Should create the config file
        mock_open.assert_called_with("azure_ml_config.json", "w")
        mock_json_dump.assert_called_once()
        
        # Verify the config structure
        config_data = mock_json_dump.call_args[0][0]
        assert "script" in config_data
        assert "environment" in config_data
        assert "compute" in config_data


if __name__ == "__main__":
    pytest.main([__file__])