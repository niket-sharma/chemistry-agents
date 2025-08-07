# Chemistry Agents - Package Distribution Guide

This guide explains how to build, distribute, and publish the `chemistry-agents` Python package.

## 📦 Package Overview

The `chemistry-agents` package provides:
- **Molecular Property Prediction**: Using transformers and neural networks
- **Chemical Engineering Unit Operations**: Distillation, heat exchangers, reactors, separation processes
- **CPU Optimization**: Full support for CPU-only systems
- **API Integration**: External model access via Hugging Face and cloud platforms

## 🔨 Building the Package

### Prerequisites

```bash
pip install build twine
```

### Quick Build

Use the provided build script:

```bash
# Full build pipeline (recommended)
python build_package.py all

# Individual steps
python build_package.py clean    # Clean artifacts
python build_package.py build    # Build package
python build_package.py check    # Check package
python build_package.py install  # Install locally
```

### Manual Build

```bash
# Clean previous builds
rm -rf dist/ build/ src/chemistry_agents.egg-info/

# Build both wheel and source distribution
python -m build

# Check package
twine check dist/*
```

## 📁 Package Structure

```
chemistry-agents-0.1.0/
├── src/chemistry_agents/           # Main package
│   ├── agents/                     # Agent classes
│   │   ├── base_agent.py          # Base agent class
│   │   ├── property_prediction_agent.py
│   │   ├── unit_operations_agent.py    # NEW: Unit operations
│   │   ├── solubility_agent.py
│   │   ├── toxicity_agent.py
│   │   └── drug_discovery_agent.py
│   ├── models/                     # Model architectures
│   │   ├── molecular_predictor.py
│   │   ├── transformer_model.py
│   │   └── graph_neural_network.py
│   └── utils/                      # Utilities
│       ├── api_integration.py      # API integration
│       ├── data_processing.py
│       ├── evaluation.py
│       └── model_hub.py
├── examples/                       # Usage examples
├── tests/                          # Test suite
├── configs/                        # Configuration files
└── scripts/                        # Training scripts
```

## 🧪 Testing the Package

### Local Installation Testing

```bash
# Install in development mode
pip install -e .

# Install from built package
pip install dist/chemistry_agents-0.1.0-py3-none-any.whl

# Test basic functionality
python -c "import chemistry_agents; print('Package imported successfully')"
```

### Run Tests

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test categories
python -m pytest tests/test_unit_operations_agent.py -v
python -m pytest tests/test_cpu_optimization.py -v
```

## 🚀 Publishing to PyPI

### Test PyPI (Recommended First)

```bash
# Upload to Test PyPI
twine upload --repository testpypi dist/*

# Install from Test PyPI to verify
pip install -i https://test.pypi.org/simple/ chemistry-agents
```

### Production PyPI

```bash
# Upload to PyPI
twine upload dist/*

# Install from PyPI
pip install chemistry-agents
```

### PyPI Configuration

Create `~/.pypirc`:

```ini
[distutils]
index-servers =
    pypi
    testpypi

[pypi]
username = __token__
password = pypi-your-production-token

[testpypi]
repository = https://test.pypi.org/legacy/
username = __token__
password = pypi-your-test-token
```

## 📊 Package Information

### Installation Options

```bash
# Minimal installation (core features only)
pip install chemistry-agents

# With GPU support
pip install chemistry-agents[gpu]

# With visualization tools
pip install chemistry-agents[visualization]

# Development installation
pip install chemistry-agents[dev]

# Everything included
pip install chemistry-agents[all]
```

### Key Features by Installation

| Installation | Features | Use Case |
|-------------|----------|----------|
| **Core** | Molecular prediction, Unit operations, CPU optimization | Basic usage, CPU systems |
| **[gpu]** | + GPU acceleration, CUDA support | High-performance computing |
| **[visualization]** | + Molecular visualization, 3D plots | Interactive analysis |
| **[notebooks]** | + Jupyter support, interactive widgets | Research, education |
| **[dev]** | + Testing tools, linting, formatting | Development |
| **[all]** | All features included | Complete installation |

## 🔧 Package Configuration

### Dependencies

**Core Dependencies (always installed):**
- `torch>=1.9.0` - PyTorch for neural networks
- `numpy>=1.21.0` - Numerical computing
- `pandas>=1.3.0` - Data manipulation
- `scikit-learn>=1.0.0` - Machine learning utilities
- `rdkit-pypi>=2022.3.0` - Chemistry toolkit
- `transformers>=4.20.0` - Hugging Face transformers
- `requests>=2.28.0` - API integration

**Optional Dependencies:**
- GPU acceleration, visualization, development tools, cloud integration

### Entry Points

The package provides command-line tools:

```bash
# Fine-tune transformer models
chemistry-agents-finetune --data_path data.csv --model_name ChemBERTa

# Train neural network models  
chemistry-agents-train --data_path data.csv --model_type neural_network
```

## 📈 Usage Examples

### Basic Molecular Property Prediction

```python
from chemistry_agents import PropertyPredictionAgent

agent = PropertyPredictionAgent()
agent.load_model()

# Predict properties
results = agent.predict_batch(["CCO", "CC(=O)O", "c1ccccc1"])
for result in results:
    print(f"SMILES: {result.smiles}, Prediction: {result.prediction:.3f}")
```

### Unit Operations Engineering

```python
from chemistry_agents import UnitOperationsAgent, UnitOperationConfig

agent = UnitOperationsAgent()
agent.load_model()

# Distillation column design
config = UnitOperationConfig(
    operation_type="distillation",
    temperature=351.15,  # K
    operation_params={
        'alpha': 2.37,      # Relative volatility
        'xd': 0.89,         # Distillate composition
        'xw': 0.02,         # Bottoms composition
        'xf': 0.40          # Feed composition
    }
)

result = agent.predict_single(config)
print(f"Performance: {result.prediction:.3f}")
```

### CPU-Optimized Usage

```python
from chemistry_agents import AgentConfig, PropertyPredictionAgent

# CPU-optimized configuration
config = AgentConfig(
    device="cpu",
    batch_size=4,  # Reduced for CPU
    cpu_optimization=True
)

agent = PropertyPredictionAgent(config=config)
agent.load_model()
```

## 🐛 Troubleshooting

### Common Installation Issues

**RDKit Installation Problems:**
```bash
# Try alternative RDKit installation
conda install -c conda-forge rdkit
# Or use pre-compiled wheel
pip install rdkit-pypi
```

**PyTorch CPU Installation:**
```bash
# Install CPU-only PyTorch
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

**Build Issues:**
```bash
# Update build tools
pip install --upgrade build setuptools wheel

# Clean and rebuild
python build_package.py clean
python build_package.py build
```

### Package Testing Issues

**Import Errors:**
- Ensure all dependencies are installed
- Check Python version compatibility (>=3.8)
- Verify package installation: `pip show chemistry-agents`

**Performance Issues:**
- Use CPU-optimized configuration
- Reduce batch sizes for memory-constrained systems
- Consider API integration for heavy computations

## 📚 Additional Resources

- **GitHub Repository**: https://github.com/niket-sharma/CHEMISTRY-AGENTS
- **CPU Optimization Guide**: `CPU_OPTIMIZATION_GUIDE.md`
- **API Integration Guide**: `src/chemistry_agents/utils/api_integration.py`
- **Examples**: `examples/` directory
- **Tests**: `tests/` directory

## 🔄 Version Management

### Updating Version

1. Update version in `pyproject.toml` and `setup.py`
2. Update `src/chemistry_agents/__init__.py`
3. Create git tag: `git tag v0.1.1`
4. Rebuild and republish package

### Release Checklist

- [ ] All tests pass
- [ ] Documentation updated
- [ ] Version bumped
- [ ] Git tag created
- [ ] Package built successfully
- [ ] Tested on Test PyPI
- [ ] Published to PyPI
- [ ] GitHub release created

---

## 🎉 Success!

Your `chemistry-agents` package is now ready for distribution! 

The package provides a comprehensive toolkit for both molecular property prediction and chemical engineering unit operations, with full CPU optimization and API integration support.

**Package Size**: ~50MB (wheel), ~2MB (source)  
**Python Compatibility**: >=3.8  
**Platforms**: Windows, macOS, Linux  
**License**: MIT