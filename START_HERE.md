# 🚀 Start Here - Chemistry Agents Setup Guide

**Welcome to Chemistry Agents!** This guide will get you up and running in 5 minutes.

## ⚡ Quick Setup (5 minutes)

### Step 1: Prerequisites Check ✅

```bash
# Check Python version (need 3.8+)
python --version
```

If you don't have Python 3.8+, download from [python.org](https://www.python.org/downloads/).

### Step 2: Clone/Navigate to Directory 📁

```bash
# If you haven't already, navigate to the chemistry-agents directory
cd chemistry-agents
```

### Step 3: Automated Setup 🤖

**Option A: Automated (Recommended)**
```bash
# Run the setup script
python setup_environment.py

# Follow the prompts - this will:
# - Create virtual environment
# - Install all dependencies  
# - Test the installation
```

**Option B: Manual Setup**
```bash
# Create virtual environment
python -m venv chemistry_agents_env

# Activate it
# Windows:
chemistry_agents_env\Scripts\activate
# Mac/Linux:
source chemistry_agents_env/bin/activate

# Install dependencies
pip install torch numpy pandas scikit-learn matplotlib rdkit-pypi transformers
pip install -e .
```

### Step 4: Test Everything Works 🧪

```bash
# Quick test (30 seconds)
python quick_test.py

# Should see:
# ✅ Basic imports: PASS
# ✅ Chemistry libs: PASS  
# ✅ Chemistry Agents: PASS
```

### Step 5: See It In Action 🎬

```bash
# Run the demo (2 minutes)
python get_started.py

# This shows:
# - Basic property prediction
# - Specialized agents
# - Batch processing
# - Hugging Face integration
```

## 🎯 What You Can Do Now

### Immediate Usage (No Training Needed)

```python
# 1. Basic prediction with mock model
from chemistry_agents import PropertyPredictionAgent

agent = PropertyPredictionAgent(property_name="logP")
agent.is_loaded = True  # Mock for demo
result = agent.predict_single("CCO")  # Ethanol
print(f"LogP: {result.prediction:.2f}")
```

```python
# 2. Use Hugging Face models
agent = PropertyPredictionAgent(
    property_name="solubility",
    model_type="transformer",
    transformer_model="DeepChem/ChemBERTa-77M-MLM"
)
# agent.load_model()  # Downloads from HF Hub
```

```python
# 3. Specialized agents
from chemistry_agents import SolubilityAgent, ToxicityAgent

sol_agent = SolubilityAgent()
tox_agent = ToxicityAgent(toxicity_endpoint="acute_toxicity")
```

### With Your Own Data

```bash
# Train your own model
python scripts/train_model.py \
    --data_path your_data.csv \
    --model_type transformer \
    --epochs 10

# Fine-tune existing model  
python scripts/fine_tune_transformer.py \
    --data_path your_data.csv \
    --model_name DeepChem/ChemBERTa-77M-MLM
```

## 📚 Examples to Try

```bash
# Start with basics
python examples/basic_usage.py

# Hugging Face integration
python examples/quick_start_hf.py

# Advanced workflows
python examples/advanced_usage.py

# Complete HF workflow
python examples/complete_hf_workflow.py
```

## 🆘 Troubleshooting

### Common Issues & Fixes

**❌ "RDKit not found"**
```bash
pip install rdkit-pypi
# or
conda install -c conda-forge rdkit
```

**❌ "torch not found"**
```bash
pip install torch
# For CPU-only: 
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

**❌ "chemistry_agents not found"**
```bash
# Make sure you're in the right directory and environment is activated
pip install -e .
```

**❌ Permission errors (Mac/Linux)**
```bash
pip install --user -e .
```

### Still Having Issues?

1. **Check Python version**: `python --version` (need 3.8+)
2. **Update pip**: `pip install --upgrade pip`
3. **Clear cache**: `pip cache purge`
4. **Start fresh**: Delete `chemistry_agents_env/` and restart
5. **Check detailed guide**: Read `INSTALLATION.md`

## 🏆 Success Checklist

After setup, you should be able to:

- [ ] ✅ Import chemistry_agents: `from chemistry_agents import PropertyPredictionAgent`
- [ ] ✅ Create agents: `agent = PropertyPredictionAgent()`
- [ ] ✅ Parse molecules: `from rdkit import Chem; Chem.MolFromSmiles("CCO")`
- [ ] ✅ Run examples: `python examples/basic_usage.py`
- [ ] ✅ Use HF models: `transformer_model="chemberta-77m"`

## 🎉 You're Ready!

Once you see ✅ for the checklist above, you're ready to use Chemistry Agents!

### Next Steps:

1. **📖 Read**: `README.md` for full documentation
2. **🧪 Explore**: `examples/` directory for more use cases  
3. **🏗️ Build**: Create your own agents and models
4. **🚀 Deploy**: Use in your chemistry/drug discovery projects

---

**Need help?** Check the `INSTALLATION.md` for detailed instructions or create an issue on GitHub.

**Ready to build?** Jump to `examples/quick_start_hf.py` to see Hugging Face integration!