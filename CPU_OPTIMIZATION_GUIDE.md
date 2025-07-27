# CPU Optimization & API Integration Guide

This guide explains how to use Chemistry Agents efficiently on CPU-only systems and with external APIs when GPU resources are not available.

## 🔧 Quick Start for CPU Users

### 1. Installation (CPU-Only)
```bash
# Install CPU-only version
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

# Skip GPU-specific packages if needed
pip install chemistry-agents --no-deps
```

### 2. Basic CPU Usage
```python
from chemistry_agents import PropertyPredictionAgent
from chemistry_agents.agents.base_agent import AgentConfig

# CPU-optimized configuration
config = AgentConfig(
    device="cpu",
    batch_size=4,  # Reduced for CPU
    cpu_optimization=True
)

# Use faster neural networks on CPU
agent = PropertyPredictionAgent(
    config=config,
    model_type="neural_network"  # Faster than transformers on CPU
)

agent.load_model()
results = agent.predict_batch(["CCO", "CC(=O)O", "c1ccccc1"])
```

## 🌐 External API Integration

### Hugging Face Inference API (Recommended)
```python
import os
from chemistry_agents.agents.base_agent import AgentConfig

# Set up API configuration
config = AgentConfig(
    use_api=True,
    api_provider="huggingface",
    api_key=os.getenv("HUGGINGFACE_API_KEY"),  # Get from https://huggingface.co/settings/tokens
    model_name="DeepChem/ChemBERTa-77M-MLM"
)

agent = PropertyPredictionAgent(config=config)
agent.load_model()  # Connects to API instead of loading local model
```

### API Benefits
- ✅ No local GPU/CPU intensive computation
- ✅ Access to large pre-trained models
- ✅ Always up-to-date models
- ✅ No local storage requirements
- ❌ Requires internet connection
- ❌ API rate limits
- ❌ Potential latency

## ☁️ Free Cloud Training Options

### 1. Google Colab (Recommended)
```bash
# Generate Colab notebook
python scripts/fine_tune_transformer.py --cloud_training --data_path your_data.csv
```

**Benefits:**
- 🆓 Free Tesla T4 GPU
- ⏰ 12-hour sessions
- 🔄 Easy to restart
- 📤 Download trained models

### 2. Kaggle Notebooks
- 🆓 Free Tesla P100 GPU  
- ⏰ 30 GPU hours/week
- 📊 Built-in datasets

### 3. Hugging Face Spaces
- 🆓 Limited free GPU
- 🌐 Host models publicly
- 🔗 Share with others

## ⚡ CPU Performance Optimization

### 1. Model Selection
```python
# ✅ CPU-Friendly Models
models = [
    "DeepChem/ChemBERTa-5M-MLM",   # Smallest, fastest
    "DeepChem/ChemBERTa-10M-MLM",  # Good balance
]

# ❌ Avoid on CPU
avoid = [
    "DeepChem/ChemBERTa-77M-MLM",  # Too large for efficient CPU use
]
```

### 2. Optimal Settings
```python
# CPU-optimized training
python scripts/fine_tune_transformer.py \
    --data_path data.csv \
    --device cpu \
    --batch_size 4 \
    --gradient_accumulation_steps 4 \
    --epochs 5 \
    --model_name "DeepChem/ChemBERTa-10M-MLM"
```

### 3. Performance Tips
```python
import torch

# Limit CPU threads to avoid oversubscription
torch.set_num_threads(4)

# Enable CPU optimizations
config = AgentConfig(
    device="cpu",
    batch_size=4,           # Small batches
    cpu_optimization=True,  # Enable optimizations
    cache_predictions=True  # Cache to avoid recomputation
)

# Use neural networks for fastest CPU inference
agent = PropertyPredictionAgent(
    config=config,
    model_type="neural_network"  # 10-100x faster than transformers on CPU
)
```

## 📊 Performance Comparison

| Model Type | CPU Speed | GPU Speed | Accuracy | Memory |
|------------|-----------|-----------|----------|---------|
| Neural Network | ✅ Fast | ✅ Very Fast | ⭐⭐⭐ Good | ✅ Low |
| Transformer (Small) | ⚠️ Moderate | ✅ Fast | ⭐⭐⭐⭐ Very Good | ⚠️ Medium |
| Transformer (Large) | ❌ Slow | ✅ Fast | ⭐⭐⭐⭐⭐ Excellent | ❌ High |
| Graph Neural Network | ⚠️ Moderate | ✅ Fast | ⭐⭐⭐⭐ Very Good | ⚠️ Medium |

## 🛠 Troubleshooting

### Common CPU Issues

**1. Out of Memory**
```python
# Reduce batch size
config.batch_size = 2  # or even 1
config.gradient_accumulation_steps = 8  # Maintain effective batch size
```

**2. Slow Training**
```bash
# Use cloud training instead
python scripts/fine_tune_transformer.py --cloud_training
```

**3. Model Loading Errors**
```python
# Ensure CPU-compatible model loading
checkpoint = torch.load(model_path, map_location='cpu')
```

### API Issues

**1. API Key Not Working**
```bash
# Set environment variable
export HUGGINGFACE_API_KEY="your_key_here"

# Or in Python
import os
os.environ["HUGGINGFACE_API_KEY"] = "your_key_here"
```

**2. Rate Limiting**
```python
# Add delays between requests
import time
time.sleep(1)  # Wait 1 second between API calls
```

**3. Model Not Available**
```python
# Check model status
from chemistry_agents.utils.api_integration import HuggingFaceInferenceAPI

api = HuggingFaceInferenceAPI(api_key)
status = api.check_model_status("DeepChem/ChemBERTa-77M-MLM")
print(status)
```

## 💰 Cost Comparison

| Option | Cost | Speed | Setup | GPU Access |
|--------|------|-------|-------|------------|
| **Local CPU** | $0 | Slow | Easy | ❌ |
| **HF API** | $0.50-2/1M tokens | Fast | Easy | ✅ |
| **Google Colab** | $0 (Free tier) | Very Fast | Moderate | ✅ |
| **Colab Pro** | $10/month | Very Fast | Easy | ✅ |
| **AWS SageMaker** | $0.05-0.20/hour | Very Fast | Hard | ✅ |

## 📋 Complete CPU Example

```python
#!/usr/bin/env python3
"""Complete CPU-optimized example"""

from chemistry_agents import SolubilityAgent
from chemistry_agents.agents.base_agent import AgentConfig
import json

def main():
    # Load CPU configuration
    config = AgentConfig(
        device="cpu",
        batch_size=4,
        cpu_optimization=True,
        cache_predictions=True
    )
    
    # Initialize with CPU-friendly model
    agent = SolubilityAgent(
        config=config,
        model_type="neural_network"  # Fast on CPU
    )
    
    print("Loading model...")
    agent.load_model()
    
    # Test molecules
    molecules = [
        "CCO",                    # Ethanol
        "CC(=O)O",               # Acetic acid  
        "c1ccccc1",              # Benzene
        "CC(C)C1=CC=C(C=C1)C(C)C(=O)O"  # Ibuprofen
    ]
    
    print("Making predictions...")
    results = agent.predict_batch(molecules)
    
    # Display results
    for result in results:
        if result.additional_info and "error" not in result.additional_info:
            solubility_class = result.additional_info.get("solubility_class")
            print(f"{result.smiles}: {result.prediction:.2f} log S ({solubility_class})")
    
    print("✅ CPU prediction complete!")

if __name__ == "__main__":
    main()
```

## 🚀 Next Steps

1. **For occasional use**: Use API integration
2. **For development**: Local CPU with neural networks  
3. **For training**: Google Colab or cloud platforms
4. **For production**: Consider cloud GPU instances

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/yourusername/chemistry-agents/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/chemistry-agents/discussions)
- **API Docs**: [Hugging Face Docs](https://huggingface.co/docs/api-inference/)

---
**💡 Remember**: CPU training is possible but slow. For best results, use cloud training and CPU for inference only.