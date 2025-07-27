#!/usr/bin/env python3
"""
CPU-Optimized Chemistry Agents Example

This example shows how to use chemistry agents efficiently on CPU
and with external APIs when no GPU is available.
"""

import os
import json
from chemistry_agents import PropertyPredictionAgent, SolubilityAgent
from chemistry_agents.agents.base_agent import AgentConfig
from chemistry_agents.utils.api_integration import show_free_alternatives

def load_cpu_config():
    """Load CPU-optimized configuration"""
    config_path = os.path.join(os.path.dirname(__file__), '..', 'configs', 'cpu_config.json')
    
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config_data = json.load(f)
        
        return AgentConfig(
            device="cpu",
            batch_size=config_data["agent_config"]["batch_size"],
            cpu_optimization=config_data["agent_config"]["cpu_optimization"],
            cache_predictions=config_data["agent_config"]["cache_predictions"],
            log_level=config_data["agent_config"]["log_level"]
        )
    else:
        # Default CPU config
        return AgentConfig(
            device="cpu",
            batch_size=4,
            cpu_optimization=True,
            cache_predictions=True
        )

def example_cpu_local_prediction():
    """Example using local CPU models"""
    print("🔧 CPU Local Prediction Example")
    print("=" * 50)
    
    # Load CPU-optimized configuration
    config = load_cpu_config()
    
    # Initialize agent with smaller model for CPU
    agent = PropertyPredictionAgent(
        config=config,
        property_name="solubility", 
        model_type="neural_network"  # Neural networks are faster on CPU
    )
    
    # Load model
    print("📥 Loading model for CPU...")
    agent.load_model()  # Uses default model
    
    # Test molecules
    test_smiles = [
        "CCO",  # Ethanol
        "CC(=O)O",  # Acetic acid
        "c1ccccc1",  # Benzene
        "CCN(CC)CC",  # Triethylamine
        "CC(C)C"  # Isobutane
    ]
    
    print("\n🧪 Making predictions...")
    results = agent.predict_batch(test_smiles)
    
    for result in results:
        print(f"  {result.smiles}: {result.prediction:.3f} (confidence: {result.confidence:.3f})")
    
    print(f"\n✅ Processed {len(results)} molecules on CPU")

def example_api_prediction():
    """Example using external API (requires API key)"""
    print("\n🌐 API Prediction Example")
    print("=" * 50)
    
    # Check for API key
    api_key = os.getenv("HUGGINGFACE_API_KEY")
    
    if not api_key:
        print("⚠️  No Hugging Face API key found.")
        print("💡 Set HUGGINGFACE_API_KEY environment variable to use API predictions")
        print("🔑 Get a free API key at: https://huggingface.co/settings/tokens")
        return
    
    # Configure for API usage
    config = AgentConfig(
        use_api=True,
        api_provider="huggingface",
        api_key=api_key,
        model_name="DeepChem/ChemBERTa-77M-MLM"
    )
    
    # Initialize agent
    agent = PropertyPredictionAgent(
        config=config,
        property_name="molecular_property"
    )
    
    print("🔗 Connecting to Hugging Face API...")
    try:
        agent.load_model()
        
        # Test prediction
        test_smiles = ["CCO", "CC(=O)O"]
        results = agent.predict_batch(test_smiles)
        
        print("✅ API predictions:")
        for result in results:
            print(f"  {result.smiles}: {result.prediction:.3f}")
            
    except Exception as e:
        print(f"❌ API prediction failed: {e}")
        print("💡 Try local CPU prediction instead")

def example_solubility_analysis():
    """Example of comprehensive solubility analysis on CPU"""
    print("\n💧 Solubility Analysis Example")
    print("=" * 50)
    
    # CPU-optimized config
    config = load_cpu_config()
    
    # Initialize solubility agent
    agent = SolubilityAgent(config=config, model_type="neural_network")
    agent.load_model()
    
    # Drug-like molecules for testing
    drug_molecules = [
        "CC1=CC=C(C=C1)C2=CC(=NN2C3=CC=C(C=C3)S(=O)(=O)N)C(F)(F)F",  # Celecoxib
        "CCN(CC)CCNC(=O)C1=CC=C(C=C1)N",  # Procainamide
        "CC(C)C1=CC=C(C=C1)C(C)C(=O)O",  # Ibuprofen
        "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",  # Caffeine
        "CC(C)(C)C1=CC=C(C=C1)C(C)(C)C"  # BHT
    ]
    
    print("🔬 Analyzing solubility and drug-likeness...")
    results = agent.predict_batch(drug_molecules)
    
    print("\n📊 Results:")
    for i, result in enumerate(results):
        if result.additional_info and "error" not in result.additional_info:
            solubility_class = result.additional_info.get("solubility_class", "unknown")
            lipinski_compliant = result.additional_info.get("lipinski_compliant", {}).get("compliant", False)
            
            print(f"\n  Molecule {i+1}: {result.smiles[:30]}...")
            print(f"    Solubility: {result.prediction:.2f} log S ({solubility_class})")
            print(f"    Lipinski compliant: {'✅' if lipinski_compliant else '❌'}")
            print(f"    Interpretation: {result.additional_info.get('interpretation', 'N/A')}")

def show_optimization_tips():
    """Show tips for CPU optimization"""
    print("\n💡 CPU Optimization Tips")
    print("=" * 50)
    
    tips = [
        "Use neural networks instead of transformers for faster CPU inference",
        "Reduce batch sizes (4-8) to avoid memory issues",
        "Enable prediction caching to avoid recomputing",
        "Use smaller pre-trained models (5M/10M parameters vs 77M)",
        "Set torch.set_num_threads(4) to limit CPU usage",
        "Consider API inference for large-scale predictions",
        "Use cloud platforms for training (Google Colab, Kaggle)",
        "Cache models locally to avoid repeated downloads"
    ]
    
    for i, tip in enumerate(tips, 1):
        print(f"  {i}. {tip}")

def main():
    """Main example function"""
    print("🧪 Chemistry Agents - CPU Optimization Examples")
    print("=" * 60)
    
    try:
        # Show free alternatives first
        print("🆓 Free GPU Alternatives:")
        show_free_alternatives()
        
        # Run examples
        example_cpu_local_prediction()
        example_api_prediction()
        example_solubility_analysis()
        show_optimization_tips()
        
        print("\n" + "="*60)
        print("✅ All examples completed!")
        print("💡 For faster training, use: python scripts/fine_tune_transformer.py --cloud_training")
        
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted by user")
    except Exception as e:
        print(f"\n❌ Example failed: {e}")
        print("💡 Try installing missing dependencies or check your environment")

if __name__ == "__main__":
    main()