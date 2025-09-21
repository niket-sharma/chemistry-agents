#!/usr/bin/env python3
"""
Example: Using Chemistry Agents with ChemBERTa Integration
"""

import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from chemistry_agents import (
    SolubilityAgent,
    ToxicityAgent,
    PropertyPredictionAgent
)
from chemistry_agents.agents.base_agent import AgentConfig

def main():
    print("🔬 Chemistry Agents with ChemBERTa Example")
    print("=" * 60)
    
    # Create configuration for CPU usage
    config = AgentConfig(
        device="cpu",
        log_level="INFO"
    )
    
    # Example molecules
    test_molecules = [
        ("CCO", "Ethanol"),
        ("c1ccccc1", "Benzene"),
        ("CC(=O)O", "Acetic acid"),
        ("CC(C)(C)O", "tert-Butanol"),
        ("c1ccc(cc1)O", "Phenol"),
    ]
    
    print("🧪 Test Molecules:")
    for smiles, name in test_molecules:
        print(f"   {smiles:15s} - {name}")
    
    # Test 1: Solubility Agent with ChemBERTa
    print("\n📊 Solubility Predictions (ChemBERTa)")
    print("-" * 40)
    
    try:
        solubility_agent = SolubilityAgent(config)
        print(f"Agent created: {solubility_agent.model_type} model")
        print(f"Transformer: {solubility_agent.transformer_model}")
        
        # Load the ChemBERTa model
        print("\n🔄 Loading ChemBERTa model...")
        solubility_agent.load_model()
        print("✅ Model loaded successfully!")
        
        print(f"Model loaded: {solubility_agent.is_loaded}")
        print(f"Vocabulary size: {solubility_agent.tokenizer.vocab_size:,}")
        
        # Make predictions
        print("\n🔮 Making predictions...")
        for smiles, name in test_molecules:
            try:
                result = solubility_agent.predict_single(smiles)
                solubility_class = solubility_agent.classify_solubility(result.prediction)
                
                print(f"{name:15s}: {result.prediction:+6.3f} (confidence: {result.confidence:.3f}) - {solubility_class}")
                
            except Exception as e:
                print(f"{name:15s}: ❌ Failed - {e}")
                
    except Exception as e:
        print(f"❌ Solubility agent failed: {e}")
        print("💡 Make sure you've run: python download_huggingface_model.py")
    
    # Test 2: Toxicity Agent with ChemBERTa
    print("\n☠️  Toxicity Predictions (ChemBERTa)")
    print("-" * 40)
    
    try:
        toxicity_agent = ToxicityAgent(config)
        print(f"Agent created: {toxicity_agent.model_type} model")
        
        # Load the ChemBERTa model
        toxicity_agent.load_model()
        print("✅ Toxicity model loaded!")
        
        # Make predictions
        for smiles, name in test_molecules[:3]:  # Test fewer molecules
            try:
                result = toxicity_agent.predict_single(smiles)
                safety_assessment = toxicity_agent.assess_safety_profile(result.prediction)
                
                print(f"{name:15s}: {result.prediction:+6.3f} - {safety_assessment}")
                
            except Exception as e:
                print(f"{name:15s}: ❌ Failed - {e}")
                
    except Exception as e:
        print(f"❌ Toxicity agent failed: {e}")
    
    # Test 3: Custom Property Agent
    print("\n⚗️  Custom Property Predictions (ChemBERTa)")
    print("-" * 40)
    
    try:
        custom_agent = PropertyPredictionAgent(
            config=config,
            property_name="bioactivity",
            model_type="transformer",
            transformer_model="DeepChem/ChemBERTa-77M-MLM"
        )
        
        print(f"Custom agent: {custom_agent.property_name}")
        custom_agent.load_model()
        print("✅ Custom model loaded!")
        
        # Batch prediction
        smiles_only = [smiles for smiles, _ in test_molecules]
        print("\n📦 Batch prediction...")
        
        results = custom_agent.predict_batch(smiles_only)
        
        print("Results:")
        for (smiles, name), result in zip(test_molecules, results):
            if 'error' not in result.additional_info:
                print(f"   {name:15s}: {result.prediction:+6.3f}")
            else:
                print(f"   {name:15s}: ❌ {result.additional_info.get('error', 'Unknown error')}")
                
    except Exception as e:
        print(f"❌ Custom agent failed: {e}")
    
    # Test 4: Model Information
    print("\n📋 Model Information")
    print("-" * 40)
    
    try:
        agent = SolubilityAgent(config)
        agent.load_model()
        
        print(f"Model name: {agent.transformer_model}")
        print(f"Device: {next(agent.model.parameters()).device}")
        print(f"Model parameters: {agent.model.num_parameters():,}")
        print(f"Hidden size: {agent.model.config.hidden_size}")
        print(f"Number of layers: {agent.model.config.num_hidden_layers}")
        print(f"Attention heads: {agent.model.config.num_attention_heads}")
        
    except Exception as e:
        print(f"❌ Model info failed: {e}")
    
    print("\n🎉 ChemBERTa example completed!")
    print("\n📋 Next steps:")
    print("   1. Try training: python scripts/train_model.py --model_type transformer")
    print("   2. Run full validation: python validate_codebase.py")
    print("   3. Test integration: python test_huggingface_integration.py")

if __name__ == "__main__":
    main()