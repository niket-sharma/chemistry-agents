"""
Integration tests for agent functionality with CPU optimization and API integration
"""

import pytest
import torch
import tempfile
import os
from unittest.mock import Mock, patch, MagicMock
from chemistry_agents.agents.base_agent import AgentConfig, PredictionResult
from chemistry_agents.agents.property_prediction_agent import PropertyPredictionAgent
from chemistry_agents.agents.solubility_agent import SolubilityAgent
from chemistry_agents.agents.toxicity_agent import ToxicityAgent
from chemistry_agents.agents.drug_discovery_agent import DrugDiscoveryAgent


class TestAgentCPUIntegration:
    """Test agent integration with CPU optimization"""
    
    @patch('chemistry_agents.models.molecular_predictor.MolecularPropertyPredictor')
    def test_property_prediction_agent_cpu(self, mock_predictor, cpu_config, sample_smiles):
        """Test PropertyPredictionAgent works on CPU"""
        # Mock the predictor
        mock_model = Mock()
        mock_model.predict.return_value = [0.5, 0.3, 0.8]
        mock_predictor.return_value = mock_model
        mock_predictor.load_model.return_value = mock_model
        
        agent = PropertyPredictionAgent(
            config=cpu_config,
            property_name="test_property",
            model_type="neural_network"
        )
        
        # Mock feature extractor
        with patch.object(agent, 'feature_extractor') as mock_extractor:
            mock_extractor.extract_features.return_value = [0.1, 0.2, 0.3]
            
            agent.load_model()
            assert agent.is_loaded
            
            # Test single prediction
            result = agent.predict_single(sample_smiles[0])
            assert isinstance(result, PredictionResult)
            assert result.smiles == sample_smiles[0]
            assert isinstance(result.prediction, float)
    
    @patch('chemistry_agents.models.molecular_predictor.MolecularPropertyPredictor')
    def test_batch_prediction_cpu(self, mock_predictor, cpu_config, sample_smiles):
        """Test batch prediction on CPU"""
        # Mock the predictor
        mock_model = Mock()
        mock_model.predict.return_value = [[0.5], [0.3], [0.8]]
        mock_predictor.return_value = mock_model
        mock_predictor.load_model.return_value = mock_model
        
        agent = PropertyPredictionAgent(
            config=cpu_config,
            model_type="neural_network"
        )
        
        # Mock feature extractor
        with patch.object(agent, 'feature_extractor') as mock_extractor:
            mock_extractor.extract_batch_features.return_value = (
                [[0.1, 0.2], [0.2, 0.3], [0.3, 0.4]], 
                [True, True, True]
            )
            
            agent.load_model()
            
            # Test batch prediction
            results = agent.predict_batch(sample_smiles[:3])
            assert len(results) == 3
            assert all(isinstance(r, PredictionResult) for r in results)
    
    def test_solubility_agent_cpu(self, cpu_config, sample_smiles):
        """Test SolubilityAgent works on CPU"""
        agent = SolubilityAgent(
            config=cpu_config,
            model_type="neural_network"
        )
        
        # Mock the underlying prediction
        with patch.object(agent, '_predict_neural_network', return_value=-1.5):
            with patch.object(agent, 'feature_extractor'):
                agent.load_model()
                
                result = agent.predict_single(sample_smiles[0])
                
                assert isinstance(result, PredictionResult)
                assert "solubility_class" in result.additional_info
                assert "lipinski_compliant" in result.additional_info
                assert "interpretation" in result.additional_info
    
    def test_toxicity_agent_cpu(self, cpu_config, sample_smiles):
        """Test ToxicityAgent works on CPU"""
        agent = ToxicityAgent(
            config=cpu_config,
            toxicity_endpoint="acute_toxicity",
            model_type="neural_network"
        )
        
        # Mock the underlying prediction
        with patch.object(agent, '_predict_neural_network', return_value=50.0):
            with patch.object(agent, 'feature_extractor'):
                agent.load_model()
                
                result = agent.predict_single(sample_smiles[0])
                
                assert isinstance(result, PredictionResult)
                assert "toxicity_class" in result.additional_info
                assert "structural_alerts" in result.additional_info
                assert "safety_assessment" in result.additional_info
    
    def test_drug_discovery_agent_cpu(self, cpu_config, sample_smiles):
        """Test DrugDiscoveryAgent works on CPU"""
        agent = DrugDiscoveryAgent(config=cpu_config)
        
        # Mock all sub-agents
        with patch.object(agent.solubility_agent, 'predict_single') as mock_sol:
            with patch.object(agent.toxicity_agent, 'predict_single') as mock_tox:
                with patch.object(agent.bioactivity_agent, 'predict_single') as mock_bio:
                    
                    # Setup mock responses
                    mock_sol.return_value = PredictionResult(
                        smiles=sample_smiles[0], 
                        prediction=-2.0,
                        additional_info={"solubility_class": "soluble"}
                    )
                    mock_tox.return_value = PredictionResult(
                        smiles=sample_smiles[0], 
                        prediction=10.0,
                        additional_info={
                            "toxicity_class": "low_toxicity",
                            "safety_assessment": {"safety_score": 80}
                        }
                    )
                    mock_bio.return_value = PredictionResult(
                        smiles=sample_smiles[0], 
                        prediction=0.7
                    )
                    
                    agent.load_model()
                    
                    result = agent.predict_single(sample_smiles[0])
                    
                    assert isinstance(result, PredictionResult)
                    assert "discovery_score" in result.additional_info
                    assert "drug_likeness" in result.additional_info
                    assert "development_recommendation" in result.additional_info


class TestAgentAPIIntegration:
    """Test agent integration with API functionality"""
    
    @patch('chemistry_agents.utils.api_integration.APIModelWrapper')
    def test_agent_api_mode(self, mock_wrapper_class, api_config, sample_smiles):
        """Test agent works in API mode"""
        # Mock API wrapper
        mock_wrapper = Mock()
        mock_wrapper.is_available.return_value = True
        mock_wrapper.predict.return_value = [0.75, 0.82]
        mock_wrapper_class.return_value = mock_wrapper
        
        agent = PropertyPredictionAgent(config=api_config)
        agent.load_model()
        
        assert agent.is_loaded
        assert hasattr(agent, 'api_model')
        
        # Test prediction using API
        result = agent.predict_single(sample_smiles[0])
        
        assert isinstance(result, PredictionResult)
        assert result.additional_info["model_type"] == f"{api_config.api_provider}_api"
        assert result.additional_info["device"] == "api"
    
    @patch('chemistry_agents.utils.api_integration.APIModelWrapper')
    def test_agent_api_unavailable(self, mock_wrapper_class, api_config):
        """Test agent handling when API is unavailable"""
        # Mock unavailable API
        mock_wrapper = Mock()
        mock_wrapper.is_available.return_value = False
        mock_wrapper_class.return_value = mock_wrapper
        
        agent = PropertyPredictionAgent(config=api_config)
        
        with pytest.raises(RuntimeError, match="API model not available"):
            agent.load_model()
    
    @patch('chemistry_agents.utils.api_integration.APIModelWrapper')
    def test_agent_api_prediction_failure(self, mock_wrapper_class, api_config, sample_smiles):
        """Test agent handling of API prediction failures"""
        # Mock API wrapper with prediction failure
        mock_wrapper = Mock()
        mock_wrapper.is_available.return_value = True
        mock_wrapper.predict.side_effect = Exception("API Error")
        mock_wrapper_class.return_value = mock_wrapper
        
        agent = PropertyPredictionAgent(config=api_config)
        agent.load_model()
        
        result = agent.predict_single(sample_smiles[0])
        
        assert isinstance(result, PredictionResult)
        assert "error" in result.additional_info
        assert result.prediction == 0.0
        assert result.confidence == 0.0


class TestAgentConfigurationIntegration:
    """Test agent configuration integration"""
    
    def test_agent_config_inheritance(self, cpu_config):
        """Test that agents properly inherit configuration"""
        agent = PropertyPredictionAgent(config=cpu_config)
        
        assert agent.config.device == cpu_config.device
        assert agent.config.batch_size == cpu_config.batch_size
        assert agent.config.cpu_optimization == cpu_config.cpu_optimization
    
    def test_agent_config_override(self, cpu_config):
        """Test that configuration can be overridden"""
        # Modify config
        cpu_config.batch_size = 1
        cpu_config.cache_predictions = False
        
        agent = PropertyPredictionAgent(config=cpu_config)
        
        assert agent.config.batch_size == 1
        assert agent.config.cache_predictions == False
    
    def test_agent_default_config(self):
        """Test agent with default configuration"""
        agent = PropertyPredictionAgent()
        
        # Should use default CPU-friendly config
        assert agent.config.device == "cpu"
        assert agent.config.batch_size <= 8
        assert agent.config.cpu_optimization == True


class TestAgentCaching:
    """Test agent caching functionality"""
    
    @patch('chemistry_agents.models.molecular_predictor.MolecularPropertyPredictor')
    def test_prediction_caching(self, mock_predictor, cpu_config, sample_smiles):
        """Test that predictions are properly cached"""
        # Mock predictor
        mock_model = Mock()
        mock_model.predict.return_value = [0.5]
        mock_predictor.return_value = mock_model
        mock_predictor.load_model.return_value = mock_model
        
        agent = PropertyPredictionAgent(
            config=cpu_config,
            model_type="neural_network"
        )
        
        with patch.object(agent, 'feature_extractor') as mock_extractor:
            mock_extractor.extract_features.return_value = [0.1, 0.2, 0.3]
            
            agent.load_model()
            
            # First prediction
            result1 = agent.predict_single(sample_smiles[0])
            
            # Second prediction of same molecule (should use cache)
            result2 = agent.predict_single(sample_smiles[0])
            
            assert result1.smiles == result2.smiles
            assert result1.prediction == result2.prediction
            
            # Cache should contain the prediction
            assert agent.get_cache_size() > 0
    
    def test_cache_clearing(self, cpu_config):
        """Test cache clearing functionality"""
        agent = PropertyPredictionAgent(config=cpu_config)
        
        # Add something to cache manually
        if agent.prediction_cache is not None:
            agent.prediction_cache["test"] = Mock()
            assert agent.get_cache_size() > 0
            
            # Clear cache
            agent.clear_cache()
            assert agent.get_cache_size() == 0


class TestAgentErrorHandling:
    """Test agent error handling"""
    
    def test_invalid_smiles_handling(self, cpu_config):
        """Test handling of invalid SMILES strings"""
        agent = PropertyPredictionAgent(config=cpu_config)
        
        # Test with invalid SMILES
        invalid_smiles = "INVALID_SMILES_123"
        
        with patch.object(agent, 'validate_smiles', return_value=False):
            result = agent.predict_single(invalid_smiles)
            
            assert isinstance(result, PredictionResult)
            assert result.smiles == invalid_smiles
            assert "error" in result.additional_info
            assert result.additional_info["error"] == "Invalid SMILES"
    
    def test_model_not_loaded_error(self, cpu_config, sample_smiles):
        """Test error when model is not loaded"""
        agent = PropertyPredictionAgent(config=cpu_config)
        
        # Don't load model
        assert not agent.is_loaded
        
        with pytest.raises(RuntimeError, match="Model not loaded"):
            agent.predict_single(sample_smiles[0])
    
    @patch('chemistry_agents.models.molecular_predictor.MolecularPropertyPredictor')
    def test_prediction_error_handling(self, mock_predictor, cpu_config, sample_smiles):
        """Test handling of prediction errors"""
        # Mock predictor that raises an error
        mock_model = Mock()
        mock_model.predict.side_effect = Exception("Prediction failed")
        mock_predictor.return_value = mock_model
        mock_predictor.load_model.return_value = mock_model
        
        agent = PropertyPredictionAgent(
            config=cpu_config,
            model_type="neural_network"
        )
        
        with patch.object(agent, 'feature_extractor') as mock_extractor:
            mock_extractor.extract_features.return_value = [0.1, 0.2, 0.3]
            
            agent.load_model()
            
            result = agent.predict_single(sample_smiles[0])
            
            assert isinstance(result, PredictionResult)
            assert "error" in result.additional_info
            assert result.prediction == 0.0
            assert result.confidence == 0.0


class TestAgentPerformance:
    """Test agent performance characteristics"""
    
    @patch('chemistry_agents.models.molecular_predictor.MolecularPropertyPredictor')
    def test_small_batch_performance(self, mock_predictor, cpu_config, sample_smiles):
        """Test performance with small batches (CPU-optimized)"""
        # Mock predictor
        mock_model = Mock()
        mock_model.predict.return_value = [[0.5], [0.3], [0.8], [0.6]]
        mock_predictor.return_value = mock_model
        mock_predictor.load_model.return_value = mock_model
        
        # Use very small batch size for CPU
        cpu_config.batch_size = 2
        
        agent = PropertyPredictionAgent(
            config=cpu_config,
            model_type="neural_network"
        )
        
        with patch.object(agent, 'feature_extractor') as mock_extractor:
            mock_extractor.extract_batch_features.return_value = (
                [[0.1, 0.2], [0.2, 0.3], [0.3, 0.4], [0.4, 0.5]], 
                [True, True, True, True]
            )
            
            agent.load_model()
            
            # Process batch
            results = agent.predict_batch(sample_smiles[:4])
            
            assert len(results) == 4
            assert all(isinstance(r, PredictionResult) for r in results)
    
    def test_memory_efficient_processing(self, cpu_config):
        """Test memory-efficient processing for CPU"""
        agent = PropertyPredictionAgent(config=cpu_config)
        
        # Should have small batch size for memory efficiency
        assert agent.config.batch_size <= 8
        assert agent.config.cpu_optimization == True


if __name__ == "__main__":
    pytest.main([__file__])