# Chemistry LLM Agent - Usage Guide

## Overview

The Chemistry LLM Agent is a **true conversational AI agent** that can reason about chemistry, plan multi-step analyses, and use specialized tools to answer complex chemistry questions.

## Quick Start

### 1. Basic Usage

```python
from chemistry_agents import ChemistryLLMAgent

# Create agent
agent = ChemistryLLMAgent()

# Ask chemistry questions
response = agent.chat("What makes benzene toxic?")
print(response)
```

### 2. Running Examples

```bash
# Quick test (no user input required)
python quick_chemistry_chat.py

# Comprehensive test suite
python test_chemistry_llm_agent.py

# Interactive chat interface
python chat_with_chemistry_agent.py

# Simple example
python simple_llm_example.py
```

## What Makes This a True AI Agent?

Unlike traditional prediction models, this agent has:

### 🧠 **Multi-Step Reasoning**
- Analyzes your question step by step
- Creates plans for complex queries
- Shows reasoning traces

### 💬 **Natural Language Conversation**
- Understands chemistry questions in plain English
- Maintains conversation context
- Provides detailed explanations

### 🔧 **Tool Orchestration**
- Uses specialized agents (solubility, toxicity, etc.) as needed
- Combines results from multiple tools
- Intelligent tool selection based on query type

### 🧪 **Chemistry Knowledge Integration**
- Built-in functional group analysis
- Structure-property relationships
- Chemical reasoning and explanations

## Example Conversations

### Basic Property Analysis
```python
agent.chat("What is the solubility of ethanol?")
```
**Response:** Multi-step analysis including molecular structure, polarity analysis, and prediction.

### Comparative Analysis
```python
agent.chat("Compare the toxicity of benzene and ethanol")
```
**Response:** Detailed comparison with reasoning about structural differences.

### Explanatory Queries
```python
agent.chat("Why are alcohols water-soluble?")
```
**Response:** Chemical explanation with molecular mechanisms.

### Molecule Analysis
```python
agent.chat("Analyze CCO for drug properties")
```
**Response:** Comprehensive analysis using multiple prediction tools.

## Agent Capabilities

### ✅ **Working Features**
- Natural language understanding
- Molecule extraction from text (SMILES and common names)
- Query type classification
- Multi-step reasoning and planning
- Chemistry knowledge integration
- Conversation history management
- Structural analysis and explanations

### ⚠️ **Requires Model Loading**
- Actual property predictions (solubility, toxicity)
- Numerical results from specialized agents
- Model-based confidence scores

## Architecture

```
ChemistryLLMAgent
├── Query Analysis → Understanding what you're asking
├── Molecule Extraction → Finding molecules in your text
├── Planning → Creating step-by-step analysis plan
├── Tool Orchestration → Using specialized agents
├── Knowledge Integration → Chemistry concepts and rules
└── Response Generation → Comprehensive explanations
```

## Available Tools

The agent can use these specialized tools:

- **SolubilityAgent**: Aqueous solubility prediction
- **ToxicityAgent**: Toxicity assessment
- **PropertyPredictionAgent**: General molecular properties
- **UnitOperationsAgent**: Chemical engineering calculations
- **Chemistry Knowledge**: Functional groups, rules, relationships

## Reasoning Traces

See how the agent thinks:

```python
response = agent.chat("Is benzene toxic?")
trace = agent.get_reasoning_trace()

for step in trace:
    print(f"{step.step_number}. {step.description}")
    if step.tool_used:
        print(f"   Tool: {step.tool_used}")
```

Example output:
```
1. Analyzing user query
2. Extracting molecules → benzene (c1ccccc1) found
3. Predicting toxicity → ToxicityAgent called
4. Structural analysis → aromatic ring identified
5. Generating explanation → toxicity mechanisms discussed
```

## Conversation History

```python
# Multi-turn conversation
agent.chat("What is ethanol?")
agent.chat("Is it toxic?")
agent.chat("How does it compare to benzene?")

# View history
history = agent.get_conversation_history()
for msg in history:
    print(f"{msg.role}: {msg.content[:50]}...")
```

## Error Handling

The agent gracefully handles:
- Invalid SMILES strings
- Unknown molecules
- Model loading failures
- Tool errors

If prediction models aren't loaded, it provides:
- Structural analysis
- Chemistry knowledge
- Qualitative explanations

## Interactive Chat Commands

When using `chat_with_chemistry_agent.py`:

- `quit` or `exit` - End conversation
- `trace` - Show reasoning for last response
- `history` - View conversation history
- `reset` - Start fresh conversation
- `help` - Show help message

## Example Use Cases

### 1. Education
```python
agent.chat("Explain why soap works to clean grease")
```

### 2. Research Support
```python
agent.chat("Compare the bioavailability of these drug candidates: CCO, c1ccccc1O")
```

### 3. Chemical Analysis
```python
agent.chat("What functional groups are in ibuprofen and how do they affect its properties?")
```

### 4. Safety Assessment
```python
agent.chat("Is CC(=O)O safe to handle? What precautions should I take?")
```

## Tips for Better Conversations

1. **Be specific**: "What makes benzene toxic?" vs "Tell me about benzene"
2. **Use SMILES**: "CCO" is better than "ethanol" for precise analysis
3. **Ask follow-ups**: "Why?" and "How?" for detailed explanations
4. **Request comparisons**: "Compare X and Y" for side-by-side analysis
5. **Ask for reasoning**: "Explain your thinking" to see the process

## Troubleshooting

### Import Errors
```python
# If ChemistryLLMAgent isn't available
from chemistry_agents.agents.chemistry_llm_agent import ChemistryLLMAgent
```

### Model Loading Issues
- The agent works without prediction models loaded
- Some features require ChemBERTa or other models
- Check model installation with `python download_huggingface_model.py`

### Performance
- All examples run on CPU
- Response time: 2-5 seconds per query
- Memory usage: ~200-500MB

## Advanced Usage

### Custom Configuration
```python
from chemistry_agents.agents.base_agent import AgentConfig

config = AgentConfig(
    device="cpu",
    log_level="INFO",
    cache_predictions=True
)

agent = ChemistryLLMAgent(config)
```

### Explanation Mode
```python
molecules = ["CCO", "c1ccccc1", "CC(=O)O"]
explanation = agent.explain_predictions(molecules)
print(explanation)
```

### Reset Conversation
```python
agent.reset_conversation()  # Start fresh
```

## Next Steps

1. **Try the examples** to see the agent in action
2. **Use the interactive chat** for real conversations
3. **Load prediction models** for numerical results
4. **Integrate into your applications** for chemistry Q&A

## Support

- Check the test scripts for working examples
- See the main README for framework overview
- Run `python test_chemistry_llm_agent.py` for diagnostics

---

**This is a true AI agent for chemistry - not just a prediction model!** 🧪🤖