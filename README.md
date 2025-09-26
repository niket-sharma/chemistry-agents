# Chemistry Agents: Intelligent ChemBERT-Powered Molecular Analysis

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An intelligent agentic framework for molecular property prediction using ChemBERT with automatic task detection and routing. Features conversational AI that understands chemistry queries and routes them to specialized models for superior accuracy over general-purpose LLMs.

## 🚀 Key Features

### 🔥 **NEW: Hybrid LLM+ChemBERT Agent**
- **Dual Intelligence**: Combines OpenAI GPT reasoning with specialized ChemBERT molecular predictions
- **Complex Problem Solving**: Handles drug design, safety assessment, regulatory analysis
- **Synthesis Pipeline**: LLM reasoning → ChemBERT predictions → Expert synthesis
- **Advanced Workflows**: Multi-step analysis with computational validation
- **Natural Language Expertise**: Chemistry domain knowledge with molecular intelligence

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
- [🔥 Hybrid LLM+ChemBERT Agent](#-hybrid-llmchemberta-agent)
- [🧠 Intelligent ChemBERT Agent](#-intelligent-chemberta-agent)
- [🚀 Quick Start](#-quick-start-all-agents)
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

## 🔥 Hybrid LLM+ChemBERT Agent

**Revolutionary chemistry AI that combines the reasoning power of large language models with the molecular intelligence of specialized ChemBERT models.**

### 🌟 What Makes It Special

The Hybrid Agent represents a breakthrough in computational chemistry by seamlessly integrating:

1. **🧠 LLM Reasoning** (OpenAI GPT-4/3.5): Natural language understanding, chemistry domain knowledge, regulatory context
2. **🔬 ChemBERT Intelligence**: Molecular property predictions, SMILES processing, specialized chemistry models
3. **🔄 Intelligent Synthesis**: Combines computational predictions with domain expertise for comprehensive answers

### 🚀 Hybrid Workflow

```
User Query → LLM Analysis → ChemBERT Predictions → Expert Synthesis → Final Answer
```

**Example:**
- **Input**: "I need a painkiller similar to ibuprofen but with better water solubility"
- **LLM**: Understands drug design requirements, safety concerns
- **ChemBERT**: Finds similar molecules, predicts solubility and toxicity
- **Synthesis**: Recommends specific alternatives with molecular reasoning

### 💡 Hybrid Agent Capabilities

**🎯 Advanced Use Cases:**
- **Drug Discovery**: Design variants of existing drugs with improved properties
- **Safety Assessment**: Comprehensive toxicity analysis with regulatory context
- **Educational Tutoring**: Chemistry concepts with computational examples
- **Environmental Analysis**: Impact assessment with predictive modeling
- **Research Acceleration**: Literature synthesis with computational validation

**🔥 Example Applications:**
- "Design a safer version of acetaminophen with reduced liver toxicity"
- "Compare bioavailability of oral vs topical drug delivery with molecular data"
- "Assess environmental impact of releasing benzene derivatives"
- "Explain structure-activity relationships in antibiotics with predictions"

### 🛠 Setup & Usage

#### 1. Configuration
```bash
# Add your OpenAI API key to .env file
echo "OPENAI_API_KEY=your_key_here" > .env

# Or export as environment variable
export OPENAI_API_KEY="your_key_here"
```

#### 2. Programmatic Usage
```python
from hybrid_agent_concept import HybridChemistryAgent
import asyncio

async def main():
    # Initialize hybrid agent
    agent = HybridChemistryAgent()
    await agent.initialize()

    # Complex chemistry query
    query = "I need a painkiller similar to ibuprofen but with better water solubility"
    result = await agent.analyze_complex_query(query)

    print("LLM Reasoning:", result.llm_reasoning)
    print("ChemBERT Predictions:", result.chemberta_predictions)
    print("Final Synthesis:", result.synthesis)
    print("Reasoning Steps:", result.reasoning_steps)

asyncio.run(main())
```

#### 3. Web UI Interface
```bash
# Launch interactive UI with both ChemBERT and Hybrid agents
python ui_intelligent_agent.py
# Open http://127.0.0.1:7860
```

#### 4. Command Line Demo
```bash
# Test hybrid functionality
python run_hybrid_agent.py
```

### 🎯 Hybrid vs ChemBERT Comparison

| Feature | ChemBERT Agent | Hybrid LLM+ChemBERT |
|---------|----------------|----------------------|
| **Molecular Predictions** | ✅ Excellent | ✅ Excellent |
| **Natural Language** | ⚠️ Limited | ✅ Advanced |
| **Complex Reasoning** | ❌ No | ✅ Yes |
| **Domain Knowledge** | ⚠️ Chemistry only | ✅ Broad + Chemistry |
| **Multi-step Analysis** | ❌ No | ✅ Yes |
| **Regulatory Context** | ❌ No | ✅ Yes |
| **Drug Design** | ⚠️ Basic | ✅ Advanced |
| **Educational Explanations** | ❌ No | ✅ Detailed |

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

## 🚀 Quick Start: All Agents

### 1. Web UI Interface (Recommended)
Launch the comprehensive web interface with both agents:

```bash
python ui_intelligent_agent.py
```
Then open http://127.0.0.1:7860 in your browser

**Features:**
- 🔥 **Hybrid Agent**: Complex reasoning + molecular predictions
- 🧠 **ChemBERT Agent**: Fast molecular property predictions
- 🎯 **Interactive Examples**: Click buttons to populate queries
- 📊 **Detailed Analysis**: View reasoning workflows and predictions

### 2. Hybrid Agent Demo
Test the hybrid LLM+ChemBERT functionality:

```bash
python run_hybrid_agent.py
```

**Example Queries for Hybrid:**
- `"I need a painkiller similar to ibuprofen but with better water solubility"`
- `"Compare toxicity profiles of methanol vs ethanol for industrial use"`
- `"Design a drug variant of acetaminophen with reduced liver toxicity"`

### 3. ChemBERT Agent Demo
Test pure ChemBERT intelligence:

```bash
python demo_intelligent_chemberta.py
```

**Example Queries for ChemBERT:**
- `"Is benzene toxic?"`
- `"How soluble is aspirin in water?"`
- `"What's the bioactivity of caffeine?"`
- `"Find molecules similar to ethanol"`

### 4. Interactive Testing
Chat directly with agents in terminal:

```bash
python test_intelligent_agent_interactive.py
```

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

### 🔥 Hybrid LLM+ChemBERT Architecture
- **LLM Component**: OpenAI GPT-4/3.5-turbo for reasoning and domain knowledge
- **ChemBERT Component**: Molecular intelligence and property predictions
- **Synthesis Layer**: Combines LLM insights with ChemBERT predictions
- **Async Workflow**: Parallel processing for optimal performance

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

### 🔥 Hybrid LLM+ChemBERT Agent

#### HybridChemistryAgent
Revolutionary agent combining LLM reasoning with ChemBERT predictions:

```python
from hybrid_agent_concept import HybridChemistryAgent
import asyncio

async def main():
    # Initialize hybrid agent
    agent = HybridChemistryAgent(openai_api_key="your_key")
    await agent.initialize()

    # Complex analysis
    result = await agent.analyze_complex_query(
        "Design a safer version of ibuprofen with better solubility"
    )

    print("LLM Reasoning:", result.llm_reasoning)
    print("ChemBERT Predictions:", result.chemberta_predictions)
    print("Final Synthesis:", result.synthesis)

asyncio.run(main())
```

#### Key Methods
- `analyze_complex_query(query: str) -> HybridAnalysisResult`: Full hybrid analysis
- `initialize()`: Async initialization of both LLM and ChemBERT components

#### HybridAnalysisResult
```python
@dataclass
class HybridAnalysisResult:
    query: str                           # Original user query
    llm_reasoning: str                   # LLM analysis and approach
    chemberta_predictions: Dict[str, Any] # ChemBERT molecular predictions
    synthesis: str                       # Final synthesized answer
    confidence: float                    # Overall confidence score
    reasoning_steps: List[str]           # Step-by-step workflow
```

### 🧠 Intelligent ChemBERT Agent

#### IntelligentChemBERTaAgent
Specialized molecular intelligence agent:

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

### 🎯 Specialized Hybrid Agents

```python
# Drug design assistant
class DrugDesignAssistant(HybridChemistryAgent):
    async def design_drug_variant(self, base_drug: str, requirements: str):
        query = f"Design a variant of {base_drug} that meets: {requirements}"
        return await self.analyze_complex_query(query)

# Chemistry tutor
class ChemistryTutor(HybridChemistryAgent):
    async def explain_with_examples(self, concept: str):
        query = f"Explain {concept} with molecular examples and predictions"
        return await self.analyze_complex_query(query)

# Safety assessment
class SafetyAssessmentAgent(HybridChemistryAgent):
    async def assess_safety(self, compound: str, use_case: str):
        query = f"Assess safety of {compound} for {use_case} with regulatory context"
        return await self.analyze_complex_query(query)
```


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