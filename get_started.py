#!/usr/bin/env python3
"""
Getting Started with Chemistry Agents
Run this script to see a quick demo of the framework
"""

import sys
import os

# Add src to path for development
current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(current_dir, 'src')
if os.path.exists(src_path):
    sys.path.insert(0, src_path)

def demo_basic_usage():
    """Demo 1: Basic molecular property prediction"""
    print("🧪 Demo 1: Basic Property Prediction")
    print("-" * 40)
    
    try:
        from chemistry_agents import PropertyPredictionAgent
        
        # Create an agent for logP prediction
        agent = PropertyPredictionAgent(
            property_name="logP",
            model_type="neural_network"
        )
        
        # Mock model loading (in real usage, you'd load a trained model)
        agent.is_loaded = True
        print("✅ Agent created and 'loaded'")
        
        # Predict for common molecules
        molecules = [
            ("Water", "O"),
            ("Ethanol", "CCO"), 
            ("Benzene", "C1=CC=CC=C1"),
            ("Aspirin", "CC(=O)OC1=CC=CC=C1C(=O)O")
        ]
        
        print("\nPredictions:")
        for name, smiles in molecules:
            result = agent.predict_single(smiles)
            print(f"  {name:10s} ({smiles:25s}): logP = {result.prediction:.2f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Demo 1 failed: {e}")
        return False

def demo_solubility_agent():
    """Demo 2: Specialized solubility agent"""
    print("\n💧 Demo 2: Solubility Prediction Agent")
    print("-" * 40)
    
    try:
        from chemistry_agents import SolubilityAgent
        
        # Create solubility agent
        agent = SolubilityAgent()
        agent.is_loaded = True  # Mock loading
        print("✅ Solubility agent created")
        
        # Test drug-like molecules
        drugs = [
            ("Aspirin", "CC(=O)OC1=CC=CC=C1C(=O)O"),
            ("Caffeine", "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"),
            ("Ibuprofen", "CC(C)C1=CC=C(C=C1)C(C)C(=O)O")
        ]
        
        print("\nSolubility Analysis:")
        for name, smiles in drugs:
            result = agent.predict_single(smiles)
            
            print(f"\n  {name}:")
            print(f"    Log S: {result.prediction:.2f}")
            
            if result.additional_info and "error" not in result.additional_info:
                sol_class = result.additional_info.get('solubility_class', 'unknown')
                interpretation = result.additional_info.get('interpretation', 'N/A')
                print(f"    Class: {sol_class}")
                print(f"    Assessment: {interpretation}")
        
        return True
        
    except Exception as e:
        print(f"❌ Demo 2 failed: {e}")
        return False

def demo_batch_processing():
    """Demo 3: Batch processing"""
    print("\n📦 Demo 3: Batch Processing")
    print("-" * 40)
    
    try:
        from chemistry_agents import PropertyPredictionAgent
        
        agent = PropertyPredictionAgent(property_name="molecular_weight")
        agent.is_loaded = True
        print("✅ Batch processing agent ready")
        
        # Small molecule library
        molecule_library = [
            "CCO",           # Ethanol
            "CCC",           # Propane  
            "CCCC",          # Butane
            "CCCCC",         # Pentane
            "C1=CC=CC=C1",   # Benzene
            "CC1=CC=CC=C1",  # Toluene
            "CCN",           # Ethylamine
            "CC(=O)O",       # Acetic acid
        ]
        
        print(f"\nProcessing {len(molecule_library)} molecules...")
        
        # Batch prediction
        results = agent.predict_batch(molecule_library)
        
        print("\nResults:")
        for smiles, result in zip(molecule_library, results):
            print(f"  {smiles:15s}: {result.prediction:.1f}")
        
        # Summary statistics
        predictions = [r.prediction for r in results]
        print(f"\nSummary:")
        print(f"  Mean: {sum(predictions)/len(predictions):.1f}")
        print(f"  Range: {min(predictions):.1f} - {max(predictions):.1f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Demo 3 failed: {e}")
        return False

def demo_huggingface_models():
    """Demo 4: Hugging Face model integration"""
    print("\n🤗 Demo 4: Hugging Face Models")
    print("-" * 40)
    
    try:
        from chemistry_agents import PropertyPredictionAgent
        from chemistry_agents.utils import get_chemistry_models, suggest_model_for_property
        
        # Show available models
        print("Available chemistry models:")
        models = get_chemistry_models()
        for key, name in list(models.items())[:3]:  # Show first 3
            print(f"  {key:15s} → {name}")
        
        # Get model suggestion
        suggested = suggest_model_for_property("solubility")
        print(f"\nSuggested for solubility: {suggested}")
        
        # Create agent with HF model
        agent = PropertyPredictionAgent(
            property_name="solubility",
            model_type="transformer",
            transformer_model="chemberta-77m"  # Using short key
        )
        
        print("✅ Agent configured with ChemBERTa model")
        print("   (In real usage, the model would be downloaded from Hugging Face)")
        
        return True
        
    except Exception as e:
        print(f"❌ Demo 4 failed: {e}")
        return False

def demo_drug_discovery():
    """Demo 5: Drug discovery workflow"""
    print("\n💊 Demo 5: Drug Discovery Workflow")
    print("-" * 40)
    
    try:
        from chemistry_agents import DrugDiscoveryAgent
        
        # Create drug discovery agent
        dd_agent = DrugDiscoveryAgent()
        dd_agent.is_loaded = True
        
        # Mock sub-agents
        dd_agent.solubility_agent.is_loaded = True
        dd_agent.toxicity_agent.is_loaded = True
        dd_agent.bioactivity_agent.is_loaded = True
        
        print("✅ Drug discovery pipeline ready")
        
        # Test compounds
        candidates = [
            "CC1=CC=C(C=C1)C2=CC(=NN2C3=CC=C(C=C3)S(=O)(=O)N)C(F)(F)F",  # Drug-like
            "CCN(CC)CCNC(=O)C1=CC=C(C=C1)N",  # Drug-like
            "CCCCCCCCCCCCCCCCCC(=O)O",  # Fatty acid (not drug-like)
        ]
        
        # Screen compounds
        screening_results = dd_agent.screen_compound_library(
            candidates,
            criteria={"min_discovery_score": 30}  # Lower threshold for demo
        )
        
        print(f"\nScreening Results:")
        print(f"  Compounds screened: {screening_results['total_compounds']}")
        print(f"  Promising hits: {screening_results['promising_compounds']}")
        print(f"  Hit rate: {screening_results['hit_rate']:.1%}")
        
        return True
        
    except Exception as e:
        print(f"❌ Demo 5 failed: {e}")
        return False

def show_next_steps():
    """Show what to do next"""
    print("\n🚀 Next Steps")
    print("=" * 50)
    
    print("Now that you've seen the basics, here's what you can do:")
    
    print("\n📚 Explore Examples:")
    print("  • python examples/basic_usage.py - Comprehensive basic examples")
    print("  • python examples/quick_start_hf.py - Hugging Face integration")
    print("  • python examples/advanced_usage.py - Advanced workflows")
    
    print("\n🎓 Learn More:")
    print("  • README.md - Full documentation")
    print("  • INSTALLATION.md - Detailed setup guide") 
    print("  • examples/ directory - All example scripts")
    
    print("\n🔧 Build Your Own:")
    print("  • Fine-tune models: python scripts/fine_tune_transformer.py")
    print("  • Train from scratch: python scripts/train_model.py")
    print("  • Create custom agents: See examples/advanced_usage.py")
    
    print("\n💡 Pro Tips:")
    print("  • Start with pre-trained HF models (chemberta-77m)")
    print("  • Use specialized agents for specific tasks")
    print("  • Fine-tune on your own datasets for best results")
    print("  • Check out the utils/ for helpful tools")

def main():
    """Run all demos"""
    print("🧪 Welcome to Chemistry Agents!")
    print("=" * 50)
    print("This demo will show you the key features of the framework.")
    print("All predictions are mocked since we don't have trained models yet.")
    
    # Run demos
    demos = [
        demo_basic_usage,
        demo_solubility_agent,
        demo_batch_processing,
        demo_huggingface_models,
        demo_drug_discovery
    ]
    
    success_count = 0
    for demo in demos:
        try:
            if demo():
                success_count += 1
        except Exception as e:
            print(f"💥 Demo crashed: {e}")
    
    # Summary
    print("\n" + "=" * 50)
    print(f"📊 Demo Results: {success_count}/{len(demos)} successful")
    
    if success_count >= 4:
        print("🎉 Great! Chemistry Agents is working correctly.")
        show_next_steps()
    elif success_count >= 2:
        print("✅ Basic functionality works. Some advanced features may need setup.")
        print("💡 Check INSTALLATION.md if you encounter issues.")
    else:
        print("⚠️ Multiple demos failed. Please check your installation:")
        print("   1. Run: python quick_test.py")
        print("   2. Check: INSTALLATION.md")
        print("   3. Make sure all dependencies are installed")

if __name__ == "__main__":
    main()