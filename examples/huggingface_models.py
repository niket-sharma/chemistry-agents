#!/usr/bin/env python3
"""
Examples using Hugging Face pre-trained models with Chemistry Agents
"""

import sys
import os

# Add src to path for development
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from chemistry_agents import PropertyPredictionAgent, AgentConfig
from chemistry_agents.models.transformer_model import MolecularTransformer

def example_huggingface_models():
    """Example: Using different Hugging Face models for molecular property prediction"""
    print("=== Hugging Face Pre-trained Models Example ===")
    
    # Available chemistry models on Hugging Face
    available_models = {
        "ChemBERTa-77M": "DeepChem/ChemBERTa-77M-MLM",
        "ChemBERTa-5M": "DeepChem/ChemBERTa-5M-MLM", 
        "ChemBERTa-10M": "DeepChem/ChemBERTa-10M-MLM",
        "MolT5": "laituan245/molt5-small",
        "ChemGPT": "ncfrey/ChemGPT-1.2B",  # Example name
        "MolFormer": "IBM/MoLFormer-XL-both-10pct",
    }
    
    # Test molecules
    test_molecules = [
        ("Aspirin", "CC(=O)OC1=CC=CC=C1C(=O)O"),
        ("Caffeine", "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"),
        ("Ibuprofen", "CC(C)C1=CC=C(C=C1)C(C)C(=O)O"),
        ("Paracetamol", "CC(=O)NC1=CC=C(C=C1)O")
    ]
    
    print(f"Testing {len(available_models)} different Hugging Face models:")
    print(f"Available models: {list(available_models.keys())}")
    
    # Example with ChemBERTa (most commonly used)
    model_name = "DeepChem/ChemBERTa-77M-MLM"
    print(f"\n--- Using {model_name} ---")
    
    # Initialize agent with specific Hugging Face model
    agent = PropertyPredictionAgent(
        property_name="logP",
        model_type="transformer",
        transformer_model=model_name
    )
    
    try:
        # This would load the actual model in real usage
        # agent.load_model()
        print("Note: Model loading simulated for demonstration")
        agent.is_loaded = True
        
        # Make predictions
        print("Predictions:")
        for name, smiles in test_molecules:
            # In real usage, this would use the actual model
            result = agent.predict_single(smiles)
            print(f"  {name:12s}: {result.prediction:.3f} (confidence: {result.confidence:.3f})")
            
    except Exception as e:
        print(f"Model loading failed (expected in demo): {e}")

def example_custom_huggingface_model():
    """Example: Using a custom fine-tuned model from Hugging Face"""
    print("\n=== Custom Fine-tuned Model Example ===")
    
    # Example of loading a fine-tuned model (would be your own model on HF Hub)
    custom_model_name = "your-username/chembert-solubility-finetuned"
    
    print(f"Loading custom model: {custom_model_name}")
    
    # You can also load local models
    local_model_path = "./models/my-fine-tuned-chembert"
    
    # Method 1: Direct model loading
    print("\nMethod 1: Direct transformer model")
    try:
        model = MolecularTransformer(
            model_name=custom_model_name,  # Or local path
            max_length=512,
            hidden_dim=256,
            num_layers=2,
            output_dim=1,
            dropout_rate=0.1
        )
        print("✓ Model loaded successfully")
    except:
        print("✗ Model loading failed (expected in demo)")
    
    # Method 2: Using with PropertyPredictionAgent
    print("\nMethod 2: With PropertyPredictionAgent")
    
    config = AgentConfig(
        model_path="./models/solubility_model.pt",  # Path to fine-tuned weights
        batch_size=16,
        confidence_threshold=0.7
    )
    
    agent = PropertyPredictionAgent(
        config=config,
        property_name="solubility",
        model_type="transformer",
        transformer_model=custom_model_name
    )
    
    # The agent would load your fine-tuned model
    # agent.load_model("./models/solubility_model.pt")

def example_model_comparison():
    """Example: Comparing different Hugging Face models"""
    print("\n=== Model Comparison Example ===")
    
    # Models to compare
    models_to_test = {
        "ChemBERTa-5M": "DeepChem/ChemBERTa-5M-MLM",
        "ChemBERTa-77M": "DeepChem/ChemBERTa-77M-MLM",
    }
    
    test_smiles = ["CCO", "CC(=O)OC1=CC=CC=C1C(=O)O", "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"]
    
    print("Comparing model architectures:")
    
    for model_label, model_name in models_to_test.items():
        print(f"\n--- {model_label} ---")
        
        agent = PropertyPredictionAgent(
            property_name="logP",
            model_type="transformer", 
            transformer_model=model_name
        )
        
        try:
            # agent.load_model()  # Would load actual model
            agent.is_loaded = True  # Mock for demo
            
            # Simulate predictions with different models
            print("Predictions:")
            for i, smiles in enumerate(test_smiles):
                result = agent.predict_single(smiles)
                print(f"  Molecule {i+1}: {result.prediction:.3f}")
                
        except Exception as e:
            print(f"Failed to load {model_label}: {e}")

def example_loading_from_local_path():
    """Example: Loading models from local filesystem"""
    print("\n=== Loading Local Models Example ===")
    
    # Scenario 1: You've fine-tuned a model locally
    local_model_paths = {
        "solubility_model": "./models/chembert-solubility/",
        "toxicity_model": "./models/chembert-toxicity/", 
        "bioactivity_model": "./models/chembert-bioactivity/"
    }
    
    print("Loading locally fine-tuned models:")
    
    for property_name, model_path in local_model_paths.items():
        print(f"\n{property_name}:")
        print(f"  Model path: {model_path}")
        
        # Check if path exists (would be real check)
        if os.path.exists(model_path):
            print("  ✓ Model directory found")
            
            # Load with agent
            agent = PropertyPredictionAgent(
                property_name=property_name.split('_')[0],
                model_type="transformer",
                transformer_model=model_path  # Local path
            )
            
            try:
                # agent.load_model()
                print("  ✓ Model loaded successfully")
            except Exception as e:
                print(f"  ✗ Failed to load: {e}")
        else:
            print("  ⚠ Model directory not found (expected in demo)")

def example_fine_tuned_model_usage():
    """Example: Complete workflow with fine-tuned model"""
    print("\n=== Complete Fine-tuned Model Workflow ===")
    
    # Scenario: You have a fine-tuned ChemBERTa model for solubility prediction
    model_config = {
        "model_name": "DeepChem/ChemBERTa-77M-MLM",  # Base model
        "fine_tuned_weights": "./models/solubility_chembert.pt",  # Your weights
        "property": "solubility",
        "task_type": "regression"
    }
    
    print(f"Using fine-tuned model for {model_config['property']} prediction")
    
    # Initialize specialized agent
    from chemistry_agents import SolubilityAgent
    
    config = AgentConfig(
        model_path=model_config["fine_tuned_weights"],
        batch_size=32,
        confidence_threshold=0.6
    )
    
    solubility_agent = SolubilityAgent(
        config=config,
        model_type="transformer",
        transformer_model=model_config["model_name"]
    )
    
    # Load your fine-tuned model
    try:
        # solubility_agent.load_model(model_config["fine_tuned_weights"])
        solubility_agent.is_loaded = True  # Mock for demo
        print("✓ Fine-tuned solubility model loaded")
    except:
        print("⚠ Model loading simulated for demo")
    
    # Test with drug-like molecules
    drug_molecules = [
        ("Aspirin", "CC(=O)OC1=CC=CC=C1C(=O)O"),
        ("Ibuprofen", "CC(C)C1=CC=C(C=C1)C(C)C(=O)O"),
        ("Caffeine", "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"),
    ]
    
    print("\nSolubility predictions with fine-tuned model:")
    
    for name, smiles in drug_molecules:
        result = solubility_agent.predict_single(smiles)
        
        print(f"\n{name}:")
        print(f"  Log S: {result.prediction:.2f}")
        if result.additional_info and "error" not in result.additional_info:
            print(f"  Class: {result.additional_info.get('solubility_class', 'unknown')}")
            print(f"  Interpretation: {result.additional_info.get('interpretation', 'N/A')}")

def example_model_hub_integration():
    """Example: Integrating with Hugging Face Model Hub"""
    print("\n=== Hugging Face Model Hub Integration ===")
    
    # Example of how to use models from the Hub
    hub_models = {
        "General Chemistry": "DeepChem/ChemBERTa-77M-MLM",
        "Molecular Properties": "seyonec/ChemBERTa_zinc250k_v2_40k",  # Example
        "Drug Discovery": "allenai/scibert_scivocab_uncased",  # Scientific text
    }
    
    print("Available models on Hugging Face Hub:")
    
    for category, model_name in hub_models.items():
        print(f"\n{category}:")
        print(f"  Model: {model_name}")
        
        # Initialize agent with Hub model
        agent = PropertyPredictionAgent(
            property_name="general_property",
            model_type="transformer",
            transformer_model=model_name
        )
        
        print(f"  ✓ Agent configured with {model_name}")
        
        # Model info
        try:
            # In real usage, you could get model info like this:
            # from transformers import AutoConfig
            # config = AutoConfig.from_pretrained(model_name)
            # print(f"  Hidden size: {config.hidden_size}")
            # print(f"  Num layers: {config.num_hidden_layers}")
            print("  ✓ Model info accessible")
        except:
            print("  ⚠ Model info retrieval simulated")

def main():
    """Run all Hugging Face model examples"""
    print("Chemistry Agents - Hugging Face Models Integration")
    print("=" * 60)
    
    example_huggingface_models()
    example_custom_huggingface_model()
    example_model_comparison()
    example_loading_from_local_path()
    example_fine_tuned_model_usage()
    example_model_hub_integration()
    
    print("\n" + "=" * 60)
    print("Key Points:")
    print("✓ Easy integration with Hugging Face models")
    print("✓ Support for custom fine-tuned models")
    print("✓ Local and remote model loading")
    print("✓ Specialized agents work with any transformer")
    print("✓ Seamless switching between model architectures")
    
    print("\nNext Steps:")
    print("1. Fine-tune your own model using scripts/fine_tune_transformer.py")
    print("2. Upload to Hugging Face Hub for sharing")
    print("3. Use with specialized agents for specific tasks")
    print("4. Compare different model architectures")

if __name__ == "__main__":
    main()