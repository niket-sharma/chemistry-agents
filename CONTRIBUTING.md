# Contributing to Chemistry Agents

We welcome contributions to Chemistry Agents! This document provides guidelines for contributing to the project.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [How to Contribute](#how-to-contribute)
- [Coding Standards](#coding-standards)
- [Testing](#testing)
- [Documentation](#documentation)
- [Pull Request Process](#pull-request-process)
- [Issue Reporting](#issue-reporting)
- [Community Guidelines](#community-guidelines)

## Code of Conduct

This project adheres to a Code of Conduct that we expect all contributors to follow. Please read and follow these guidelines to help us maintain a welcoming community.

### Our Pledge

We pledge to make participation in our project a harassment-free experience for everyone, regardless of age, body size, disability, ethnicity, sex characteristics, gender identity and expression, level of experience, education, socio-economic status, nationality, personal appearance, race, religion, or sexual identity and orientation.

### Our Standards

Examples of behavior that contributes to creating a positive environment include:

- Using welcoming and inclusive language
- Being respectful of differing viewpoints and experiences
- Gracefully accepting constructive criticism
- Focusing on what is best for the community
- Showing empathy towards other community members

## Getting Started

### Prerequisites

- Python 3.8 or higher
- Git
- Basic knowledge of chemistry and machine learning
- Familiarity with PyTorch and RDKit

### Development Setup

1. **Fork the repository**
   ```bash
   # Fork the repo on GitHub, then clone your fork
   git clone https://github.com/YOUR_USERNAME/chemistry-agents.git
   cd chemistry-agents
   ```

2. **Create a development environment**
   ```bash
   # Using conda (recommended)
   conda create -n chemistry-agents-dev python=3.9
   conda activate chemistry-agents-dev
   
   # Or using venv
   python -m venv chemistry-agents-dev
   source chemistry-agents-dev/bin/activate  # On Windows: chemistry-agents-dev\Scripts\activate
   ```

3. **Install development dependencies**
   ```bash
   # Install in development mode
   pip install -e ".[dev,all]"
   
   # Install pre-commit hooks
   pre-commit install
   ```

4. **Verify installation**
   ```bash
   # Run tests to ensure everything is working
   pytest tests/ -v
   
   # Run basic examples
   python examples/basic_usage.py
   ```

## How to Contribute

### Types of Contributions

We welcome several types of contributions:

1. **Bug Reports**: Help us identify and fix issues
2. **Feature Requests**: Suggest new functionality
3. **Code Contributions**: Implement new features or fix bugs
4. **Documentation**: Improve or expand documentation
5. **Examples**: Add usage examples or tutorials
6. **Performance Improvements**: Optimize existing code
7. **Testing**: Add or improve test coverage

### Contribution Areas

#### High Priority Areas

- **New Model Architectures**: Implement novel neural network designs
- **Additional Properties**: Add support for new molecular properties
- **Performance Optimization**: Memory and speed improvements
- **Better Error Handling**: Robust error handling and recovery
- **Advanced Evaluation Metrics**: Chemistry-specific evaluation methods

#### Feature Ideas

- **New Agents**: Specialized agents for specific tasks
- **Model Interpretability**: SHAP, LIME, attention visualization
- **API Endpoints**: REST API for model serving
- **Cloud Integration**: AWS/GCP/Azure deployment tools
- **Visualization Tools**: Molecular and results visualization
- **Benchmark Datasets**: Standard dataset integration

## Coding Standards

### Python Style Guide

We follow PEP 8 with some modifications:

- **Line Length**: 100 characters (instead of 79)
- **String Quotes**: Use double quotes for strings
- **Import Organization**: Use isort for import sorting

### Code Formatting

We use several tools to maintain code quality:

```bash
# Format code with black
black src/ tests/ examples/

# Sort imports with isort
isort src/ tests/ examples/

# Lint code with flake8
flake8 src/ tests/ examples/

# Type checking with mypy
mypy src/
```

### Pre-commit Hooks

Pre-commit hooks automatically run these tools before each commit:

```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/psf/black
    rev: 22.3.0
    hooks:
      - id: black
  - repo: https://github.com/pycqa/isort
    rev: 5.10.1
    hooks:
      - id: isort
  - repo: https://github.com/pycqa/flake8
    rev: 4.0.1
    hooks:
      - id: flake8
```

### Documentation Standards

- Use Google-style docstrings
- Include type hints for all function parameters and return values
- Document complex algorithms and chemical concepts
- Provide usage examples in docstrings

Example:
```python
def predict_solubility(self, smiles: str, model_type: str = "transformer") -> PredictionResult:
    """
    Predict aqueous solubility for a molecule.
    
    Args:
        smiles: SMILES string representation of the molecule
        model_type: Type of model to use ("transformer" or "neural_network")
    
    Returns:
        PredictionResult containing prediction, confidence, and additional info
    
    Raises:
        ValueError: If SMILES string is invalid
        RuntimeError: If model is not loaded
    
    Example:
        >>> agent = SolubilityAgent()
        >>> agent.load_model()
        >>> result = agent.predict_solubility("CCO")  # Ethanol
        >>> print(f"Log S: {result.prediction:.2f}")
    """
```

## Testing

### Test Structure

```
tests/
├── unit/                 # Unit tests
│   ├── test_agents.py
│   ├── test_models.py
│   └── test_utils.py
├── integration/          # Integration tests
│   ├── test_pipelines.py
│   └── test_workflows.py
├── data/                 # Test data
│   ├── test_molecules.csv
│   └── test_datasets/
└── conftest.py          # Pytest configuration
```

### Writing Tests

- Write unit tests for all new functions and classes
- Include edge cases and error conditions
- Use descriptive test names
- Mock external dependencies (file I/O, network calls)

Example test:
```python
import pytest
from chemistry_agents import SolubilityAgent

class TestSolubilityAgent:
    @pytest.fixture
    def agent(self):
        agent = SolubilityAgent()
        agent.is_loaded = True  # Mock loaded state
        return agent
    
    def test_predict_single_valid_smiles(self, agent):
        """Test prediction with valid SMILES."""
        result = agent.predict_single("CCO")
        
        assert result.smiles == "CCO"
        assert isinstance(result.prediction, float)
        assert result.confidence is not None
        assert result.confidence >= 0.0
        assert result.confidence <= 1.0
    
    def test_predict_single_invalid_smiles(self, agent):
        """Test prediction with invalid SMILES."""
        result = agent.predict_single("invalid_smiles")
        
        assert result.additional_info is not None
        assert "error" in result.additional_info
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=chemistry_agents --cov-report=html

# Run specific test file
pytest tests/unit/test_agents.py

# Run with verbose output
pytest -v

# Run tests in parallel
pytest -n auto
```

## Documentation

### Documentation Types

1. **API Documentation**: Auto-generated from docstrings
2. **User Guides**: High-level usage documentation
3. **Tutorials**: Step-by-step examples
4. **Developer Docs**: Architecture and contribution guides

### Building Documentation

```bash
# Install documentation dependencies
pip install -e ".[docs]"

# Build documentation
cd docs/
make html

# Serve documentation locally
python -m http.server 8000 -d _build/html/
```

### Documentation Guidelines

- Keep documentation up-to-date with code changes
- Include practical examples
- Explain chemistry concepts for non-experts
- Use clear, concise language
- Add diagrams and visualizations where helpful

## Pull Request Process

### Before Submitting

1. **Create an Issue**: For significant changes, create an issue first to discuss the approach
2. **Create a Branch**: Use a descriptive branch name
   ```bash
   git checkout -b feature/new-toxicity-agent
   git checkout -b fix/solubility-prediction-bug
   git checkout -b docs/improve-api-reference
   ```

3. **Make Changes**: Follow coding standards and write tests
4. **Test Thoroughly**: Ensure all tests pass
5. **Update Documentation**: Update relevant documentation

### Submitting a Pull Request

1. **Push to Your Fork**
   ```bash
   git push origin feature/new-toxicity-agent
   ```

2. **Create Pull Request**: Use the GitHub interface to create a PR

3. **PR Template**: Fill out the PR template completely

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] Added unit tests
- [ ] Added integration tests
- [ ] All tests pass
- [ ] Manual testing completed

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] No breaking changes (or justified)
```

### Review Process

1. **Automated Checks**: Ensure all CI checks pass
2. **Code Review**: At least one maintainer will review
3. **Testing**: Reviewers may test functionality
4. **Feedback**: Address review comments promptly
5. **Approval**: PR approved by maintainer
6. **Merge**: Maintainer will merge the PR

## Issue Reporting

### Bug Reports

Use the bug report template:

```markdown
**Bug Description**
Clear description of the bug

**To Reproduce**
Steps to reproduce the behavior

**Expected Behavior**
What you expected to happen

**Environment**
- OS: [e.g., Ubuntu 20.04]
- Python version: [e.g., 3.9.7]
- Chemistry Agents version: [e.g., 0.1.0]
- Dependencies: [relevant versions]

**Additional Context**
Any other context about the problem
```

### Feature Requests

Use the feature request template:

```markdown
**Feature Description**
Clear description of the proposed feature

**Motivation**
Why is this feature needed?

**Proposed Solution**
How should this feature work?

**Alternatives**
Alternative approaches considered

**Additional Context**
Any other context or examples
```

## Community Guidelines

### Communication Channels

- **GitHub Issues**: Bug reports and feature requests
- **GitHub Discussions**: General questions and community discussion
- **Email**: For security issues or private communications

### Getting Help

- Check existing documentation and examples first
- Search existing issues before creating new ones
- Provide minimal reproducible examples
- Be patient and respectful when asking for help

### Best Practices for Contributors

1. **Start Small**: Begin with small contributions to understand the codebase
2. **Ask Questions**: Don't hesitate to ask for clarification
3. **Be Patient**: Reviews may take time, especially for large changes
4. **Stay Updated**: Keep your fork updated with the main repository
5. **Follow Up**: Respond to review comments promptly

### Recognition

Contributors are recognized in several ways:

- **Contributors File**: Listed in CONTRIBUTORS.md
- **Release Notes**: Significant contributions mentioned in releases
- **GitHub**: Automatic recognition through GitHub's contribution graphs

## Development Workflow

### Git Workflow

We use a simplified Git flow:

1. **Main Branch**: Contains stable, released code
2. **Feature Branches**: For new features and bug fixes
3. **Pull Requests**: All changes go through PR review

### Release Process

1. **Version Bump**: Update version numbers
2. **Changelog**: Update CHANGELOG.md
3. **Testing**: Comprehensive testing on multiple platforms
4. **Documentation**: Ensure documentation is up-to-date
5. **Release**: Create GitHub release and publish to PyPI

## Advanced Contributing

### Custom Agents

When contributing new agents:

1. **Inherit from BaseChemistryAgent**
2. **Implement required methods**: `load_model()`, `predict_single()`
3. **Add comprehensive tests**
4. **Include usage examples**
5. **Document chemical background**

### New Model Architectures

For new model architectures:

1. **Follow existing patterns** in the models/ directory
2. **Include training scripts**
3. **Provide pre-trained weights** (if possible)
4. **Benchmark against existing models**
5. **Document architecture choices**

### Performance Optimization

For performance improvements:

1. **Profile before optimizing**
2. **Benchmark improvements**
3. **Consider memory usage**
4. **Test on different hardware**
5. **Document performance gains**

## Questions?

If you have questions about contributing, please:

1. Check this document first
2. Search existing GitHub issues and discussions
3. Create a new discussion for general questions
4. Create an issue for specific bugs or feature requests

Thank you for contributing to Chemistry Agents! 🧪⚗️