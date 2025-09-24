# Chemistry Agents: Intelligent ChemBERT-Powered Molecular Analysis

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An intelligent agentic framework for molecular property prediction using ChemBERT with automatic task detection and routing. Features conversational AI that understands chemistry queries and routes them to specialized models for superior accuracy over general-purpose LLMs.

## 🚀 Key Features

### 🧠 Intelligent ChemBERT Agent
- **Automatic Task Detection**: Understands natural language queries about toxicity, solubility, bioactivity
- **Intelligent Routing**: Routes queries to specialized fine-tuned ChemBERT models
- **Conversational Interface**: Chat with the agent using natural language chemistry questions
- **Reasoning Traces**: Shows step-by-step decision making process
- **Superior Chemistry Understanding**: Trained on 77M molecules vs text-only LLMs

### 🎯 Specialized Models
- **Toxicity Prediction**: Dedicated ChemBERT model for toxicity assessment
- **Solubility Analysis**: Specialized aqueous solubility prediction
- **Bioactivity Screening**: Pharmacological activity evaluation
- **General Chemistry**: Molecular similarity and clustering

### 🚀 Traditional Agents
- **Multiple Model Architectures**: Neural networks, Graph Neural Networks (GNNs), and Transformer models
- **Unit Operations Agent**: Distillation, heat exchangers, reactors, separation processes, and fluid mechanics
- **CPU Optimization**: Full support for CPU-only systems with optimized performance
- **Pre-trained Models**: Built on ChemBERTa and other chemistry-specific transformers

## 📋 Table of Contents

- [Installation](#installation)
- [🧠 Intelligent ChemBERT Agent](#-intelligent-chemberta-agent)
- [🚀 Quick Start](#-quick-start-intelligent-agent)
- [🎯 Specialized Training](#-training-specialized-models)
- [Traditional Agents](#traditional-agents)
- [Model Architectures](#model-architectures)
- [API Reference](#api-reference)
- [Examples](#examples)
- [Contributing](#contributing)

## 🛠 Installation

### Requirements

- Python 3.8+
- PyTorch 1.9+
- RDKit
- Transformers
- scikit-learn
- pandas
- numpy

### Install from Source

```bash
git clone https://github.com/yourusername/chemistry-agents.git
cd chemistry-agents
pip install -r requirements.txt
```

## 🧠 Intelligent ChemBERT Agent

The core innovation of this framework is the **Intelligent ChemBERT Agent** that automatically understands chemistry questions and routes them to specialized models.

### How It Works

1. **Task Detection**: Analyzes your natural language query to detect task type
2. **Molecule Extraction**: Finds chemical names and SMILES in your question
3. **Intelligent Routing**: Routes to specialized ChemBERT models or general analysis
4. **Conversational Response**: Provides detailed molecular intelligence

### Example Workflow

```python
from chemistry_agents.agents.intelligent_chemberta_agent import create_intelligent_chemberta_agent
from chemistry_agents.agents.base_agent import AgentConfig

# Create intelligent agent
config = AgentConfig(device="cpu", log_level="WARNING")
agent = create_intelligent_chemberta_agent(config)

# Ask natural language chemistry questions
response = agent.chat("Is benzene toxic to humans?")
print(response)

# The agent automatically:
# 1. Detects this is a toxicity query (confidence: 0.33)
# 2. Extracts "benzene" and converts to SMILES "c1ccccc1"
# 3. Routes to toxicity-specialized ChemBERT model (if trained)
# 4. Provides molecular intelligence analysis
```

## 🚀 Quick Start: Intelligent Agent

### 1. Interactive Demo
Test the intelligent agent with various chemistry questions:

```bash
python demo_intelligent_chemberta.py
```

### 2. Web UI Interface
Launch a user-friendly web interface:

```bash
python ui_intelligent_agent.py
```
Then open http://127.0.0.1:7860 in your browser

### 3. Interactive Testing
Chat directly with the agent in terminal:

```bash
python test_intelligent_agent_interactive.py
```

Try these example queries:
- `"Is benzene toxic?"`
- `"How soluble is aspirin in water?"`
- `"What's the bioactivity of caffeine?"`
- `"Find molecules similar to ethanol"`

### 3. Programmatic Usage

```python
from chemistry_agents.agents.intelligent_chemberta_agent import create_intelligent_chemberta_agent

# Initialize agent
agent = create_intelligent_chemberta_agent()

# Chat interface
response = agent.chat("How toxic is benzene?")

# Get reasoning trace
trace = agent.get_reasoning_trace()
for step in trace:
    print(f"{step.step_number}. {step.description}")
```

## 🎯 Training Specialized Models

To enable the full intelligent routing capabilities, train specialized ChemBERT models:

```bash
# Train all specialized models (toxicity, solubility, bioactivity)
python train_task_specific_chemberta.py
```

This will:
1. Download task-specific datasets
2. Fine-tune ChemBERT for each task
3. Save specialized models to `models/task_specific_chemberta/`
4. Enable automatic routing in the intelligent agent

### Individual Model Training

```bash
# Train only toxicity model
python train_task_specific_chemberta.py --task toxicity

# Train only solubility model
python train_task_specific_chemberta.py --task solubility

# Train only bioactivity model
python train_task_specific_chemberta.py --task bioactivity
```

### What Happens After Training

Once specialized models are trained, the intelligent agent will:

```python
# Before training: Uses general ChemBERT
response = agent.chat("Is benzene toxic?")
# → "Using general ChemBERT analysis"

# After training: Uses specialized toxicity model
response = agent.chat("Is benzene toxic?")
# → "Using specialized toxicity ChemBERT model"
# → More accurate toxicity-specific predictions
```

## Other Available Agents

The framework also includes traditional specialized agents for specific chemistry tasks:

- **PropertyPredictionAgent**: General molecular property prediction
- **SolubilityAgent**: Aqueous solubility prediction
- **ToxicityAgent**: Multi-endpoint toxicity assessment
- **UnitOperationsAgent**: Chemical engineering unit operations

## 🏗 Model Architectures

### Intelligent ChemBERT Models
- **Task-Specific ChemBERT**: Fine-tuned on toxicity, solubility, bioactivity datasets
- **Base ChemBERT**: General molecular understanding (77M parameters)
- **Intelligent Routing**: Automatic task detection and model selection

### Traditional Models
- **Neural Networks**: Fast predictions with molecular descriptors
- **Graph Neural Networks**: Leveraging molecular topology
- **Transformer Models**: SMILES-based sequence modeling

## 🎯 Additional Training

For traditional neural network models:

```bash
python scripts/train_model.py \
    --data_path data/your_dataset.csv \
    --target_column target \
    --epochs 100
```

## 📚 API Reference

### Intelligent ChemBERT Agent

#### IntelligentChemBERTaAgent
Main class for intelligent chemistry conversations:

```python
from chemistry_agents.agents.intelligent_chemberta_agent import create_intelligent_chemberta_agent

# Create agent
agent = create_intelligent_chemberta_agent()

# Chat interface
response = agent.chat("Is benzene toxic?")

# Task detection
task_type, confidence = agent.detect_task_type("How soluble is aspirin?")

# Reasoning trace
trace = agent.get_reasoning_trace()
```

#### Key Methods
- `chat(query: str) -> str`: Natural language chemistry chat
- `detect_task_type(query: str) -> Tuple[str, float]`: Automatic task detection
- `get_reasoning_trace() -> List[ReasoningStep]`: Step-by-step analysis


## 🧪 How to Run & Test

### 1. Install Dependencies
```bash
git clone https://github.com/yourusername/chemistry-agents.git
cd chemistry-agents
pip install -r requirements.txt
```

### 2. Run Interactive Demo
```bash
# See automatic task detection and routing examples
python demo_intelligent_chemberta.py
```

### 3. Chat with the Agent
```bash
# Interactive chemistry chat interface
python test_intelligent_agent_interactive.py
```

**Try these example queries:**
- `"Is benzene toxic?"`
- `"How soluble is aspirin in water?"`
- `"What's the bioactivity of caffeine?"`
- `"Find molecules similar to ethanol"`

### 4. Train Specialized Models (Optional)
```bash
# Enable intelligent routing to specialized models
python train_task_specific_chemberta.py
```

### Expected Results
- ✅ **Task Detection**: Identifies toxicity/solubility/bioactivity queries
- ✅ **Molecule Recognition**: Extracts chemicals from natural language
- ✅ **Intelligent Routing**: Uses appropriate specialized models
- ✅ **Conversational AI**: Provides detailed molecular analysis

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

---

**Intelligent Chemistry AI for the Research Community**