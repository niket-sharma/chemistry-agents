#!/usr/bin/env python3
"""
Quick Training Demo for Task-Specific ChemBERTa

Demonstrates training a single task-specific model and then using it.
This is a simplified version that trains faster for demonstration.
"""

import os
import sys
import pandas as pd
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
import json

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def quick_demo():
    """Quick demo of the task-specific training process"""

    print("QUICK TASK-SPECIFIC CHEMBERTA DEMO")
    print("=" * 50)

    # Check what we have
    print("1. CHECKING CURRENT STATE:")
    print(f"   Base ChemBERTa available: {os.path.exists('models/huggingface_cache')}")
    print(f"   Solubility dataset: {os.path.exists('data/processed/esol_solubility.csv')}")
    print(f"   Task-specific models: {os.path.exists('models/task_specific_chemberta')}")

    # Load solubility dataset
    print("\n2. ANALYZING SOLUBILITY DATASET:")
    try:
        df = pd.read_csv('data/processed/esol_solubility.csv')
        print(f"   Dataset size: {len(df)} compounds")
        print(f"   Columns: {list(df.columns)}")
        print(f"   Solubility range: {df['solubility_logS'].min():.2f} to {df['solubility_logS'].max():.2f}")
        print(f"   Sample compounds:")
        for i in range(min(3, len(df))):
            smiles = df.iloc[i]['smiles']
            logS = df.iloc[i]['solubility_logS']
            name = df.iloc[i].get('compound_id', 'Unknown')
            print(f"     {name}: {smiles} (logS = {logS})")
    except Exception as e:
        print(f"   Error loading dataset: {e}")
        return

    # Show what training would do
    print("\n3. TRAINING PROCESS (what train_task_specific_chemberta.py does):")
    print("   a) Load ChemBERTa base model (DeepChem/ChemBERTa-77M-MLM)")
    print("   b) Add regression head for solubility prediction")
    print("   c) Fine-tune on ESOL dataset (1128 compounds)")
    print("   d) Save specialized model to models/task_specific_chemberta/")
    print("   e) Save task metadata (target range, training stats)")

    # Test current intelligent agent
    print("\n4. TESTING CURRENT INTELLIGENT AGENT:")
    try:
        from chemistry_agents.agents.intelligent_chemberta_agent import create_intelligent_chemberta_agent
        from chemistry_agents.agents.base_agent import AgentConfig

        agent = create_intelligent_chemberta_agent(AgentConfig(device="cpu", log_level="WARNING"))

        print(f"   Base ChemBERTa loaded: {agent.is_loaded}")
        print(f"   Specialized models: {len(agent.specialized_models)}")

        # Test task detection
        test_query = "How soluble is aspirin in water?"
        task_type, confidence = agent.detect_task_type(test_query)
        molecules = agent._extract_molecules_from_query(test_query)

        print(f"\n   TEST QUERY: '{test_query}'")
        print(f"   Detected task: {task_type} (confidence: {confidence:.2f})")
        print(f"   Molecules found: {molecules}")

        if task_type == "solubility":
            print("   ✓ Correctly detected solubility query")
            if len(agent.specialized_models) > 0:
                print("   → Would use specialized solubility model")
            else:
                print("   → Falls back to general ChemBERTa (no specialized model)")

    except Exception as e:
        print(f"   Error testing agent: {e}")

    # Show what happens with ChemBERTa inference
    print("\n5. CURRENT CHEMBERTA INFERENCE:")
    try:
        response = agent.chat("How soluble is aspirin?")
        lines = response.split('\n')
        print("   Agent response (first 10 lines):")
        for line in lines[:10]:
            print(f"     {line}")
        if len(lines) > 10:
            print("     ... (response continues)")

        # Show reasoning
        trace = agent.get_reasoning_trace()
        print(f"\n   Reasoning steps ({len(trace)}):")
        for step in trace:
            print(f"     {step.step_number}. {step.description}")

    except Exception as e:
        print(f"   Error with inference: {e}")

    print("\n6. NEXT STEPS:")
    print("   To create specialized models:")
    print("   → Run: python train_task_specific_chemberta.py")
    print("   → This will train models for solubility, toxicity, bioactivity")
    print("   → Training takes 10-20 minutes on CPU")
    print("   → After training, the intelligent agent will automatically use them")

    print("\n   To test with specialized models:")
    print("   → After training, run: python demo_intelligent_chemberta.py")
    print("   → Agent will route solubility queries to solubility model")
    print("   → Better accuracy on task-specific predictions")

if __name__ == "__main__":
    quick_demo()