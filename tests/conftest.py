"""
Pytest configuration and fixtures for chemistry agents tests
"""

import pytest
import torch
import tempfile
import os
import pandas as pd
from unittest.mock import Mock, patch
from chemistry_agents.agents.base_agent import AgentConfig

@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests"""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir

@pytest.fixture
def sample_smiles():
    """Sample SMILES strings for testing"""
    return [
        "CCO",           # Ethanol
        "CC(=O)O",       # Acetic acid
        "c1ccccc1",      # Benzene
        "CCN(CC)CC",     # Triethylamine
        "CC(C)C",        # Isobutane
        "O",             # Water
        "CO",            # Methanol
        "C"              # Methane
    ]

@pytest.fixture
def sample_targets():
    """Sample target values for testing"""
    return [
        -0.77, -0.13, -2.1, 0.5, -1.8, -1.5, -0.92, -1.2
    ]

@pytest.fixture
def sample_dataset(sample_smiles, sample_targets, temp_dir):
    """Create a sample CSV dataset for testing"""
    df = pd.DataFrame({
        'smiles': sample_smiles,
        'target': sample_targets,
        'property': ['solubility'] * len(sample_smiles)
    })
    
    csv_path = os.path.join(temp_dir, 'test_data.csv')
    df.to_csv(csv_path, index=False)
    return csv_path

@pytest.fixture
def cpu_config():
    """CPU-optimized configuration for testing"""
    return AgentConfig(
        device="cpu",
        batch_size=2,  # Very small for testing
        cpu_optimization=True,
        cache_predictions=True,
        log_level="WARNING"  # Reduce log noise in tests
    )

@pytest.fixture
def api_config():
    """API configuration for testing"""
    return AgentConfig(
        use_api=True,
        api_provider="huggingface",
        api_key="test_key",
        model_name="test_model",
        batch_size=2
    )

@pytest.fixture
def mock_torch_device():
    """Mock torch device to always return CPU"""
    with patch('torch.device') as mock_device:
        mock_device.return_value = torch.device('cpu')
        yield mock_device

@pytest.fixture
def mock_rdkit():
    """Mock RDKit for tests that don't need real chemistry"""
    with patch('chemistry_agents.agents.base_agent.Chem') as mock_chem:
        # Mock successful molecule creation
        mock_mol = Mock()
        mock_chem.MolFromSmiles.return_value = mock_mol
        yield mock_chem

@pytest.fixture
def mock_api_response():
    """Mock API response for testing"""
    return [
        {"score": 0.75},
        {"score": 0.82}
    ]

@pytest.fixture
def mock_requests():
    """Mock requests for API testing"""
    with patch('requests.post') as mock_post, patch('requests.get') as mock_get:
        # Setup successful response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = [{"score": 0.75}, {"score": 0.82}]
        mock_response.raise_for_status.return_value = None
        
        mock_post.return_value = mock_response
        mock_get.return_value = mock_response
        
        yield mock_post, mock_get

@pytest.fixture(autouse=True)
def setup_test_environment():
    """Setup test environment variables"""
    # Ensure we're in CPU mode for tests
    os.environ['CHEMISTRY_AGENTS_DEVICE'] = 'cpu'
    os.environ['CHEMISTRY_AGENTS_LOG_LEVEL'] = 'WARNING'
    
    # Mock API keys for testing
    original_hf_key = os.environ.get('HUGGINGFACE_API_KEY')
    os.environ['HUGGINGFACE_API_KEY'] = 'test_key_123'
    
    yield
    
    # Cleanup
    if original_hf_key:
        os.environ['HUGGINGFACE_API_KEY'] = original_hf_key
    else:
        os.environ.pop('HUGGINGFACE_API_KEY', None)

@pytest.fixture
def mock_model_loading():
    """Mock model loading to avoid downloading large models during tests"""
    with patch('chemistry_agents.models.transformer_model.AutoModel') as mock_auto_model, \
         patch('chemistry_agents.models.transformer_model.AutoTokenizer') as mock_auto_tokenizer:
        
        # Mock tokenizer
        mock_tokenizer = Mock()
        mock_tokenizer.return_value = {
            'input_ids': torch.tensor([[1, 2, 3, 4]]),
            'attention_mask': torch.tensor([[1, 1, 1, 1]])
        }
        mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer
        
        # Mock model
        mock_model = Mock()
        mock_model.config.hidden_size = 768
        mock_model.return_value.last_hidden_state = torch.randn(1, 4, 768)
        mock_auto_model.from_pretrained.return_value = mock_model
        
        yield mock_auto_model, mock_auto_tokenizer