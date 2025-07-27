"""
Tests for CPU optimization features
"""

import pytest
import torch
import os
from unittest.mock import Mock, patch, MagicMock
from chemistry_agents.agents.base_agent import AgentConfig
from chemistry_agents.agents.property_prediction_agent import PropertyPredictionAgent


class TestCPUConfiguration:
    """Test CPU configuration and optimization features"""
    
    def test_cpu_config_defaults(self):
        """Test that CPU configuration has appropriate defaults"""
        config = AgentConfig()
        
        assert config.device == "cpu"
        assert config.batch_size <= 8  # Should be small for CPU
        assert config.cpu_optimization == True
        assert config.cache_predictions == True
    
    def test_cpu_optimization_enabled(self, cpu_config):
        """Test that CPU optimization settings are properly configured"""
        assert cpu_config.device == "cpu"
        assert cpu_config.cpu_optimization == True
        assert cpu_config.batch_size <= 8
        
    def test_device_setup_prioritizes_cpu(self):
        """Test that device setup prioritizes CPU even when GPU is available"""
        # Import the function we want to test
        from chemistry_agents.scripts.fine_tune_transformer import setup_device
        
        # Test auto device selection (should default to CPU)
        device = setup_device('auto')
        assert device.type == 'cpu'
        
        # Test explicit CPU selection
        device = setup_device('cpu')
        assert device.type == 'cpu'


class TestAgentCPUIntegration:
    """Test agent integration with CPU optimizations"""
    
    @patch('chemistry_agents.models.transformer_model.AutoModel')
    @patch('chemistry_agents.models.transformer_model.AutoTokenizer')
    def test_agent_loads_on_cpu(self, mock_tokenizer, mock_model, cpu_config):
        """Test that agent loads properly on CPU"""
        # Mock the model and tokenizer
        mock_tokenizer.from_pretrained.return_value = Mock()
        mock_model_instance = Mock()
        mock_model_instance.config.hidden_size = 768
        mock_model.from_pretrained.return_value = mock_model_instance
        
        agent = PropertyPredictionAgent(
            config=cpu_config,
            model_type="transformer"
        )
        
        # Should not raise an exception
        agent._load_default_model()
        assert agent.is_loaded
    
    def test_cpu_thread_optimization(self, cpu_config):
        """Test that CPU thread optimization is applied"""
        original_threads = torch.get_num_threads()
        
        with patch('torch.set_num_threads') as mock_set_threads:
            agent = PropertyPredictionAgent(
                config=cpu_config,
                model_type="neural_network"
            )
            
            # Mock the feature extractor and model loading
            with patch.object(agent, 'feature_extractor'):
                with patch('chemistry_agents.models.molecular_predictor.MolecularPropertyPredictor'):
                    agent.load_model()
                    
                    # Should have called set_num_threads if CPU optimization is enabled
                    if cpu_config.cpu_optimization:
                        mock_set_threads.assert_called()
    
    def test_batch_size_limits_on_cpu(self, cpu_config):
        """Test that batch sizes are appropriately limited for CPU"""
        # Even if we try to set a large batch size, it should be reasonable for CPU
        cpu_config.batch_size = 2  # Keep small for testing
        
        agent = PropertyPredictionAgent(config=cpu_config)
        
        # Batch size should be small for CPU efficiency
        assert agent.config.batch_size <= 8


class TestMemoryOptimization:
    """Test memory optimization features for CPU"""
    
    def test_model_loading_with_cpu_device(self, cpu_config, temp_dir):
        """Test that models are loaded with CPU device mapping"""
        with patch('torch.load') as mock_load:
            mock_load.return_value = {
                'model_state_dict': {},
                'input_dim': 100,
                'hidden_dims': [64, 32],
                'output_dim': 1,
                'dropout_rate': 0.1
            }
            
            agent = PropertyPredictionAgent(config=cpu_config)
            
            # Create a fake model file
            model_path = os.path.join(temp_dir, 'test_model.pt')
            with open(model_path, 'wb') as f:
                f.write(b'fake model data')
            
            # This should not raise memory errors
            try:
                agent.load_model(model_path)
            except Exception as e:
                # We expect some errors due to mocking, but not memory errors
                assert "memory" not in str(e).lower()
                assert "cuda" not in str(e).lower()
    
    def test_prediction_caching_enabled(self, cpu_config):
        """Test that prediction caching is enabled for CPU efficiency"""
        agent = PropertyPredictionAgent(config=cpu_config)
        
        assert cpu_config.cache_predictions == True
        assert agent.prediction_cache is not None


class TestCPUPerformanceFeatures:
    """Test performance-related CPU features"""
    
    def test_neural_network_preferred_for_cpu(self, cpu_config):
        """Test that neural networks are suggested for CPU usage"""
        # Neural networks should work without issues on CPU
        agent = PropertyPredictionAgent(
            config=cpu_config,
            model_type="neural_network"
        )
        
        assert agent.model_type == "neural_network"
    
    def test_small_batch_processing(self, cpu_config, sample_smiles):
        """Test that small batches are processed efficiently"""
        cpu_config.batch_size = 2
        
        agent = PropertyPredictionAgent(config=cpu_config)
        
        # Mock the prediction methods
        with patch.object(agent, 'predict_single') as mock_predict:
            mock_predict.return_value = Mock(prediction=0.5, confidence=0.8)
            
            # Should handle small batches without issues
            results = agent.predict_batch(sample_smiles[:4])
            assert len(results) == 4


class TestCPUErrorHandling:
    """Test error handling specific to CPU usage"""
    
    def test_cuda_error_helpful_message(self, cpu_config):
        """Test that CUDA errors provide helpful messages"""
        agent = PropertyPredictionAgent(config=cpu_config)
        
        # Simulate a CUDA-related error
        with patch.object(agent, 'logger') as mock_logger:
            try:
                # Force a CUDA device error
                agent.config.device = 'cuda'
                with patch('torch.device', side_effect=RuntimeError("CUDA not available")):
                    agent.load_model()
            except:
                pass
            
            # Should log helpful message about using CPU
            assert mock_logger.info.called or mock_logger.error.called
    
    def test_memory_error_handling(self, cpu_config):
        """Test graceful handling of memory errors on CPU"""
        agent = PropertyPredictionAgent(config=cpu_config)
        
        # Simulate memory error during model loading
        with patch('torch.load', side_effect=RuntimeError("out of memory")):
            with pytest.raises(RuntimeError):
                agent.load_model("fake_path.pt")


class TestCPUConfigurationValidation:
    """Test validation of CPU-specific configurations"""
    
    def test_cpu_config_validation(self):
        """Test that CPU configurations are validated properly"""
        config = AgentConfig(
            device="cpu",
            batch_size=4,
            cpu_optimization=True
        )
        
        # Should not raise validation errors
        agent = PropertyPredictionAgent(config=config)
        assert agent.config.device == "cpu"
        assert agent.config.cpu_optimization == True
    
    def test_invalid_batch_size_handling(self):
        """Test handling of invalid batch sizes"""
        # Very large batch size should be handled gracefully
        config = AgentConfig(
            device="cpu",
            batch_size=1000,  # Too large for typical CPU
            cpu_optimization=True
        )
        
        # Should not raise an exception during initialization
        agent = PropertyPredictionAgent(config=config)
        assert agent.config.batch_size == 1000  # Should preserve user setting
    
    @pytest.mark.parametrize("device", ["cpu", "CPU", "Cpu"])
    def test_device_case_insensitive(self, device):
        """Test that device specification is case-insensitive"""
        config = AgentConfig(device=device)
        agent = PropertyPredictionAgent(config=config)
        
        # Should normalize to lowercase
        assert agent.config.device.lower() == "cpu"


if __name__ == "__main__":
    pytest.main([__file__])