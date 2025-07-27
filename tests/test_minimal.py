"""
Minimal tests that work without heavy dependencies
"""

import pytest
import os
import json
import sys
from pathlib import Path

# Add src to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))


class TestProjectStructure:
    """Test basic project structure"""
    
    def test_project_files_exist(self):
        """Test that key project files exist"""
        base_dir = project_root
        
        required_files = [
            'src/chemistry_agents/__init__.py',
            'src/chemistry_agents/agents/__init__.py',
            'src/chemistry_agents/agents/base_agent.py',
            'src/chemistry_agents/utils/__init__.py',
            'src/chemistry_agents/utils/api_integration.py',
            'configs/cpu_config.json',
            'CPU_OPTIMIZATION_GUIDE.md'
        ]
        
        for file_path in required_files:
            full_path = base_dir / file_path
            assert full_path.exists(), f"Required file missing: {file_path}"
    
    def test_cpu_config_file_structure(self):
        """Test CPU configuration file is valid JSON"""
        config_path = project_root / 'configs' / 'cpu_config.json'
        
        assert config_path.exists()
        
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        assert "agent_config" in config
        assert config["agent_config"]["device"] == "cpu"
        assert config["agent_config"]["cpu_optimization"] == True
        assert config["agent_config"]["batch_size"] <= 8
    
    def test_optimization_guide_exists(self):
        """Test that optimization guide exists and has content"""
        guide_path = project_root / 'CPU_OPTIMIZATION_GUIDE.md'
        
        assert guide_path.exists()
        
        with open(guide_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Should contain key sections
        assert "CPU Optimization" in content
        assert "API Integration" in content
        assert "Google Colab" in content
        assert "Hugging Face" in content
        assert len(content) > 1000  # Should be substantial


class TestCodeStructure:
    """Test basic code structure without imports"""
    
    def test_agent_config_class_exists(self):
        """Test AgentConfig class can be read from file"""
        base_agent_path = project_root / 'src' / 'chemistry_agents' / 'agents' / 'base_agent.py'
        
        with open(base_agent_path, 'r') as f:
            content = f.read()
        
        # Should contain AgentConfig class definition
        assert "class AgentConfig" in content
        assert "device: str = \"cpu\"" in content
        assert "cpu_optimization: bool = True" in content
        assert "use_api: bool = False" in content
    
    def test_api_integration_classes_exist(self):
        """Test API integration classes can be read from file"""
        api_path = project_root / 'src' / 'chemistry_agents' / 'utils' / 'api_integration.py'
        
        with open(api_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Should contain key classes
        assert "class HuggingFaceInferenceAPI" in content
        assert "class APIModelWrapper" in content
        assert "class CloudTrainingManager" in content
        assert "class GoogleColabIntegration" in content
    
    def test_fine_tune_script_modifications(self):
        """Test fine-tune script has CPU optimizations"""
        script_path = project_root / 'scripts' / 'fine_tune_transformer.py'
        
        with open(script_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Should contain CPU optimization code
        assert "def setup_device" in content
        assert "device('cpu')" in content
        assert "use_api" in content or "api_key" in content


class TestConfiguration:
    """Test configuration files and settings"""
    
    def test_pipfile_exists_and_valid(self):
        """Test Pipfile exists and contains required packages"""
        pipfile_path = project_root / 'Pipfile'
        
        assert pipfile_path.exists()
        
        with open(pipfile_path, 'r') as f:
            content = f.read()
        
        # Should contain essential packages
        assert "torch" in content
        assert "pytest" in content
        assert "transformers" in content
        assert "rdkit-pypi" in content
    
    def test_examples_directory_exists(self):
        """Test examples directory has CPU usage example"""
        examples_path = project_root / 'examples'
        
        assert examples_path.exists()
        
        cpu_example = examples_path / 'cpu_optimized_usage.py'
        assert cpu_example.exists()
        
        with open(cpu_example, 'r', encoding='utf-8') as f:
            content = f.read()
        
        assert "cpu" in content.lower()
        assert "AgentConfig" in content


class TestTestInfrastructure:
    """Test that test infrastructure is set up correctly"""
    
    def test_test_files_exist(self):
        """Test all test files exist"""
        tests_dir = project_root / 'tests'
        
        expected_test_files = [
            'test_cpu_optimization.py',
            'test_api_integration.py',
            'test_agent_integration.py',
            'test_cloud_training.py',
            'conftest.py.bak',
            'test_runner.py'
        ]
        
        for test_file in expected_test_files:
            test_path = tests_dir / test_file
            assert test_path.exists(), f"Test file missing: {test_file}"
    
    def test_test_runner_script(self):
        """Test runner script exists and has correct structure"""
        runner_path = project_root / 'tests' / 'test_runner.py'
        
        with open(runner_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        assert "def run_all_tests" in content
        assert "def run_quick_smoke_test" in content
        assert "def check_dependencies" in content


if __name__ == "__main__":
    # Run tests directly
    import subprocess
    result = subprocess.run([sys.executable, "-m", "pytest", __file__, "-v"], 
                          capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)
    exit(result.returncode)