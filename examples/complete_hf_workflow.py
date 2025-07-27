#!/usr/bin/env python3
"""
Complete workflow example: Using Hugging Face models with Chemistry Agents
This example shows the full pipeline from model selection to deployment
"""

import sys
import os

# Add src to path for development
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from chemistry_agents import (
    PropertyPredictionAgent, 
    SolubilityAgent, 
    ToxicityAgent, 
    DrugDiscoveryAgent
)
from chemistry_agents.utils import get_chemistry_models, suggest_model_for_property, get_model_info

def workflow_step_1_model_discovery():
    """Step 1: Discover and select appropriate models"""
    print("🔍 Step 1: Model Discovery and Selection")
    print("=" * 50)
    
    # Get available chemistry models
    available_models = get_chemistry_models()
    
    print("Available chemistry models:")
    for key, full_name in available_models.items():
        print(f"  {key:15s} → {full_name}")
    
    # Get model suggestions for different tasks
    tasks = ["solubility", "toxicity", "logp", "bioactivity"]
    
    print("\nRecommended models by task:")
    for task in tasks:
        suggested = suggest_model_for_property(task)
        print(f"  {task:12s} → {suggested}")
    
    # Get detailed model information
    print("\nModel details for ChemBERTa-77M:")
    info = get_model_info("chemberta-77m")
    for key, value in info.items():
        print(f"  {key:20s}: {value}")

def workflow_step_2_quick_setup():
    """Step 2: Quick setup with pre-trained models"""
    print("\n⚡ Step 2: Quick Setup with Pre-trained Models")
    print("=" * 50)
    
    # Method 1: Use model keys for convenience
    print("Method 1: Using model keys")
    agent1 = PropertyPredictionAgent(
        property_name="logP",
        model_type="transformer",
        transformer_model="chemberta-77m"  # Short key
    )
    print("  ✓ Agent created with ChemBERTa-77M")
    
    # Method 2: Use full Hugging Face names
    print("\nMethod 2: Using full HF names")
    agent2 = PropertyPredictionAgent(
        property_name="solubility",
        model_type="transformer", 
        transformer_model="DeepChem/ChemBERTa-77M-MLM"  # Full name
    )
    print("  ✓ Agent created with full model name")
    
    # Method 3: Your custom model from HF Hub
    print("\nMethod 3: Custom model from HF Hub")
    agent3 = PropertyPredictionAgent(
        property_name="custom_property",
        model_type="transformer",
        transformer_model="your-username/chembert-finetuned"  # Your model
    )
    print("  ✓ Agent configured for custom model")

def workflow_step_3_specialized_agents():
    """Step 3: Using specialized agents with HF models"""
    print("\n🎯 Step 3: Specialized Agents with HF Models")
    print("=" * 50)
    
    # Initialize specialized agents with different models
    agents = {
        "Solubility": SolubilityAgent(
            model_type="transformer",
            transformer_model="DeepChem/ChemBERTa-77M-MLM"
        ),
        "Toxicity": ToxicityAgent(
            model_type="transformer", 
            transformer_model="chemberta-77m",  # Using short key
            toxicity_endpoint="acute_toxicity"
        ),
        "Drug Discovery": DrugDiscoveryAgent()
    }
    
    print("Specialized agents configured:")
    for name, agent in agents.items():
        print(f"  ✓ {name} Agent")
    
    # Test molecules
    test_molecules = [
        ("Aspirin", "CC(=O)OC1=CC=CC=C1C(=O)O"),
        ("Caffeine", "CN1C=NC2=C1C(=O)N(C(=O)N2C)C")
    ]
    
    # Mock model loading and make predictions
    print("\nMaking predictions:")
    for agent_name, agent in agents.items():
        if hasattr(agent, 'is_loaded'):
            agent.is_loaded = True
        print(f"\n  {agent_name} predictions:")
        
        for mol_name, smiles in test_molecules:
            try:
                if agent_name == "Drug Discovery":
                    # Drug discovery agent needs sub-agents loaded
                    agent.solubility_agent.is_loaded = True
                    agent.toxicity_agent.is_loaded = True 
                    agent.bioactivity_agent.is_loaded = True
                
                result = agent.predict_single(smiles)
                print(f"    {mol_name:10s}: {result.prediction:.3f}")
            except Exception as e:
                print(f"    {mol_name:10s}: Error - {e}")

def workflow_step_4_batch_processing():
    """Step 4: Efficient batch processing"""
    print("\n📦 Step 4: Efficient Batch Processing")
    print("=" * 50)
    
    # Large-scale molecular library
    molecular_library = [
        "CCO", "CCC", "CCCC", "CCCCC",  # Alcohols
        "C1=CC=CC=C1", "CC1=CC=CC=C1", "CCC1=CC=CC=C1",  # Aromatics  
        "CCN", "CCCN", "CCCCN",  # Amines
        "CC(=O)O", "CCC(=O)O", "CCCC(=O)O",  # Carboxylic acids
        "CC(=O)OC1=CC=CC=C1C(=O)O",  # Aspirin
        "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",  # Caffeine
    ]
    
    print(f"Processing {len(molecular_library)} molecules...")
    
    # Initialize agent for batch processing
    agent = PropertyPredictionAgent(
        property_name="logP",
        model_type="transformer",
        transformer_model="chemberta-77m"
    )
    
    # Mock model loading
    agent.is_loaded = True
    
    # Batch prediction
    results = agent.predict_batch(molecular_library)
    
    print("\nBatch results summary:")
    predictions = [r.prediction for r in results]
    print(f"  Molecules processed: {len(results)}")
    print(f"  Mean prediction: {sum(predictions)/len(predictions):.3f}")
    print(f"  Prediction range: {min(predictions):.3f} to {max(predictions):.3f}")
    
    # Show top 5 results
    print("\nTop 5 predictions:")
    sorted_results = sorted(zip(molecular_library, results), 
                          key=lambda x: x[1].prediction, reverse=True)
    
    for i, (smiles, result) in enumerate(sorted_results[:5]):
        print(f"  {i+1}. {smiles:25s}: {result.prediction:.3f}")

def workflow_step_5_model_comparison():
    """Step 5: Comparing different HF models"""
    print("\n🏆 Step 5: Model Comparison")
    print("=" * 50)
    
    # Models to compare
    models_to_compare = {
        "ChemBERTa-5M": "chemberta-5m",
        "ChemBERTa-77M": "chemberta-77m", 
        "SciBERT": "scibert"
    }
    
    test_smiles = "CC(=O)OC1=CC=CC=C1C(=O)O"  # Aspirin
    
    print(f"Comparing models on: {test_smiles}")
    print(f"{'Model':<15} {'Prediction':<12} {'Confidence':<12} {'Speed':<10}")
    print("-" * 55)
    
    for model_name, model_key in models_to_compare.items():
        agent = PropertyPredictionAgent(
            property_name="logP",
            model_type="transformer",
            transformer_model=model_key
        )
        
        # Mock model loading and prediction
        agent.is_loaded = True
        result = agent.predict_single(test_smiles)
        
        # Simulate speed (smaller models are faster)
        speed = "Fast" if "5M" in model_name else "Medium" if "77M" in model_name else "Slow"
        
        print(f"{model_name:<15} {result.prediction:<12.3f} {result.confidence:<12.3f} {speed:<10}")

def workflow_step_6_fine_tuned_models():
    """Step 6: Using your own fine-tuned models"""
    print("\n🎨 Step 6: Fine-tuned Models Integration")
    print("=" * 50)
    
    # Scenario: You have fine-tuned models for specific tasks
    custom_models = {
        "solubility": "your-username/chembert-esol-finetuned",
        "toxicity": "your-username/chembert-tox21-finetuned", 
        "bioactivity": "your-username/chembert-chembl-finetuned"
    }
    
    print("Custom fine-tuned models:")
    for task, model_name in custom_models.items():
        print(f"  {task:12s} → {model_name}")
    
    # Use in drug discovery pipeline
    print("\nConfiguring drug discovery pipeline with custom models:")
    
    dd_agent = DrugDiscoveryAgent()
    
    # Override default models with your fine-tuned ones
    dd_agent.solubility_agent = SolubilityAgent(
        model_type="transformer",
        transformer_model=custom_models["solubility"]
    )
    
    dd_agent.toxicity_agent = ToxicityAgent(
        model_type="transformer", 
        transformer_model=custom_models["toxicity"]
    )
    
    dd_agent.bioactivity_agent = PropertyPredictionAgent(
        property_name="bioactivity",
        model_type="transformer",
        transformer_model=custom_models["bioactivity"]
    )
    
    print("  ✓ Drug discovery pipeline configured with custom models")
    
    # Example usage
    print("\nExample compound screening:")
    compounds = [
        "CC1=CC=C(C=C1)C2=CC(=NN2C3=CC=C(C=C3)S(=O)(=O)N)C(F)(F)F",
        "CCN(CC)CCNC(=O)C1=CC=C(C=C1)N"
    ]
    
    # Mock loading and screening
    dd_agent.is_loaded = True
    dd_agent.solubility_agent.is_loaded = True
    dd_agent.toxicity_agent.is_loaded = True
    dd_agent.bioactivity_agent.is_loaded = True
    
    screening_results = dd_agent.screen_compound_library(
        compounds,
        criteria={"min_discovery_score": 50}
    )
    
    print(f"  Screened: {screening_results['total_compounds']} compounds")
    print(f"  Hits: {screening_results['promising_compounds']}")

def workflow_step_7_deployment():
    """Step 7: Deployment considerations"""
    print("\n🚀 Step 7: Deployment and Production")
    print("=" * 50)
    
    deployment_options = [
        "💻 Local deployment with pre-downloaded models",
        "☁️  Cloud deployment with HF Hub integration", 
        "🐳 Docker containers with model caching",
        "⚡ GPU acceleration for large-scale screening",
        "🔄 Model versioning and A/B testing"
    ]
    
    print("Deployment considerations:")
    for option in deployment_options:
        print(f"  {option}")
    
    print("\nExample production setup:")
    print("```python")
    print("# Production agent with caching and error handling")
    print("from chemistry_agents import PropertyPredictionAgent, AgentConfig")
    print("")
    print("config = AgentConfig(")
    print("    batch_size=64,  # Larger batches for efficiency")
    print("    cache_predictions=True,  # Enable caching")
    print("    confidence_threshold=0.8  # Higher threshold")
    print(")")
    print("")
    print("agent = PropertyPredictionAgent(")
    print("    config=config,")
    print("    property_name='solubility',")
    print("    model_type='transformer',")
    print("    transformer_model='your-username/production-model'")
    print(")")
    print("```")

def main():
    """Run the complete workflow"""
    print("🧪 Complete Hugging Face Workflow with Chemistry Agents")
    print("=" * 70)
    
    workflow_step_1_model_discovery()
    workflow_step_2_quick_setup() 
    workflow_step_3_specialized_agents()
    workflow_step_4_batch_processing()
    workflow_step_5_model_comparison()
    workflow_step_6_fine_tuned_models()
    workflow_step_7_deployment()
    
    print("\n" + "=" * 70)
    print("🎉 Complete Workflow Finished!")
    print("\n📝 Summary:")
    print("  ✅ Model discovery and selection")
    print("  ✅ Quick setup with pre-trained models") 
    print("  ✅ Specialized agents configuration")
    print("  ✅ Efficient batch processing")
    print("  ✅ Model comparison and benchmarking")
    print("  ✅ Custom fine-tuned model integration")
    print("  ✅ Production deployment considerations")
    
    print("\n🔗 Key Benefits:")
    print("  • Easy integration with any HF model")
    print("  • Seamless switching between models")
    print("  • Specialized chemistry agents")
    print("  • Production-ready architecture")
    print("  • Comprehensive evaluation tools")

if __name__ == "__main__":
    main()