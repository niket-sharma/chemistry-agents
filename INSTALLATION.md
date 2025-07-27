# Installation Guide

This guide will help you set up Chemistry Agents on your system.

## 🚀 Quick Start (Recommended)

### Option 1: Automated Setup

1. **Navigate to the project directory:**
   ```bash
   cd chemistry-agents
   ```

2. **Run the setup script:**
   ```bash
   python setup_environment.py
   ```

3. **Activate the environment:**
   ```bash
   # Windows
   chemistry_agents_env\Scripts\activate
   
   # Linux/Mac
   source chemistry_agents_env/bin/activate
   ```

4. **Test the installation:**
   ```bash
   python test_installation.py
   ```

### Option 2: Manual Setup

## 📋 Manual Installation Steps

### Step 1: Check Python Version

```bash
python --version
# Should be Python 3.8 or higher
```

If you don't have Python 3.8+, download from [python.org](https://www.python.org/downloads/).

### Step 2: Create Virtual Environment

```bash
# Create virtual environment
python -m venv chemistry_agents_env

# Activate it
# Windows:
chemistry_agents_env\Scripts\activate

# Linux/Mac:
source chemistry_agents_env/bin/activate
```

### Step 3: Install Core Dependencies

```bash
# Upgrade pip first
pip install --upgrade pip

# Install core ML libraries
pip install torch>=1.9.0
pip install numpy>=1.21.0
pip install pandas>=1.3.0
pip install scikit-learn>=1.0.0
pip install matplotlib>=3.5.0
pip install seaborn>=0.11.0
pip install tqdm>=4.64.0
```

### Step 4: Install Chemistry Libraries

```bash
# Install RDKit (chemistry toolkit)
pip install rdkit-pypi

# Install transformers for AI models
pip install transformers>=4.20.0
pip install tokenizers>=0.13.0
pip install datasets>=2.0.0
```

### Step 5: Install Chemistry Agents

```bash
# Install in development mode
pip install -e .

# Or install from requirements
pip install -r requirements.txt
```

## 🧪 Testing the Installation

### Quick Test

```bash
python test_installation.py
```

This will test:
- ✅ Basic imports (NumPy, Pandas, PyTorch)
- ✅ Chemistry libraries (RDKit, Transformers)
- ✅ Chemistry Agents functionality

### Run Examples

```bash
# Basic usage examples
python examples/basic_usage.py

# Hugging Face integration
python examples/quick_start_hf.py

# Advanced workflows
python examples/advanced_usage.py
```

## 🔧 Platform-Specific Instructions

### Windows

```cmd
# Create environment
python -m venv chemistry_agents_env

# Activate
chemistry_agents_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install project
pip install -e .
```

### Linux/Ubuntu

```bash
# Install system dependencies
sudo apt-get update
sudo apt-get install python3-dev python3-pip

# Create environment
python3 -m venv chemistry_agents_env

# Activate
source chemistry_agents_env/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install project
pip install -e .
```

### macOS

```bash
# Install using Homebrew (recommended)
brew install python

# Create environment
python3 -m venv chemistry_agents_env

# Activate
source chemistry_agents_env/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install project
pip install -e .
```

## 🐳 Docker Installation (Optional)

### Build Docker Image

```bash
# Create Dockerfile
cat > Dockerfile << EOF
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Install in development mode
RUN pip install -e .

# Run tests
RUN python test_installation.py

CMD ["python", "examples/basic_usage.py"]
EOF

# Build image
docker build -t chemistry-agents .

# Run container
docker run -it chemistry-agents
```

## ☁️ Cloud Setup (Google Colab, Jupyter, etc.)

### Google Colab

```python
# In a Colab cell
!git clone https://github.com/yourusername/chemistry-agents.git
%cd chemistry-agents

# Install dependencies
!pip install rdkit-pypi
!pip install transformers
!pip install -e .

# Test installation
!python test_installation.py
```

### Jupyter Notebook

```bash
# Install Jupyter
pip install jupyter

# Start notebook server
jupyter notebook

# Or use JupyterLab
pip install jupyterlab
jupyter lab
```

## 🚨 Troubleshooting

### Common Issues

#### 1. RDKit Installation Failed

```bash
# Try conda instead of pip
conda install -c conda-forge rdkit

# Or use alternative
pip install rdkit-pypi --no-cache-dir
```

#### 2. PyTorch Issues

```bash
# For CPU-only PyTorch
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# For CUDA (if you have GPU)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

#### 3. Transformers/Tokenizers Issues

```bash
# Install specific versions
pip install transformers==4.21.0
pip install tokenizers==0.13.3
```

#### 4. Memory Issues

```bash
# Install with no-cache for large packages
pip install --no-cache-dir torch
pip install --no-cache-dir transformers
```

#### 5. Permission Issues (Linux/Mac)

```bash
# Use user directory
pip install --user -r requirements.txt

# Or fix permissions
sudo chown -R $USER:$USER chemistry_agents_env/
```

### Getting Help

If you encounter issues:

1. **Check Python version**: Must be 3.8+
2. **Update pip**: `pip install --upgrade pip`
3. **Clear cache**: `pip cache purge`
4. **Reinstall**: Delete `chemistry_agents_env/` and start over
5. **Check logs**: Look for specific error messages

## 📊 Verification Checklist

After installation, verify these work:

- [ ] Python imports: `import numpy, pandas, torch`
- [ ] RDKit: `from rdkit import Chem; Chem.MolFromSmiles("CCO")`
- [ ] Transformers: `from transformers import AutoTokenizer`
- [ ] Chemistry Agents: `from chemistry_agents import PropertyPredictionAgent`
- [ ] Example runs: `python examples/basic_usage.py`

## 🔄 Updating

To update Chemistry Agents:

```bash
# Activate environment
source chemistry_agents_env/bin/activate  # or Windows equivalent

# Pull latest changes
git pull origin main

# Update dependencies
pip install --upgrade -r requirements.txt

# Reinstall project
pip install -e .

# Test
python test_installation.py
```

## 🎯 Next Steps

Once installed successfully:

1. **Read the README**: Understanding the framework
2. **Run Examples**: Start with `examples/basic_usage.py`
3. **Try Tutorials**: Check `examples/quick_start_hf.py`
4. **Train Models**: Use `scripts/train_model.py`
5. **Build Applications**: Create your own agents

## 💡 Performance Tips

### For Better Performance

```bash
# Install optimized libraries
pip install intel-scipy  # Intel-optimized SciPy
pip install mkl  # Intel Math Kernel Library

# For GPU support
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### For Development

```bash
# Install development tools
pip install black isort flake8 mypy
pip install pytest pytest-cov
pip install pre-commit

# Setup pre-commit hooks
pre-commit install
```

---

## ✅ Installation Complete!

You're now ready to use Chemistry Agents for molecular property prediction! 🧪✨