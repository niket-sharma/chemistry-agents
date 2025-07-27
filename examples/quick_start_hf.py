#!/usr/bin/env python3
"""
Quick start guide for using Hugging Face models with Chemistry Agents
"""

import sys
import os

# Add src to path for development
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from chemistry_agents import PropertyPredictionAgent, SolubilityAgent, ToxicityAgent

def quick_start():
    """Quickest way to get started with Hugging Face models"""
    
    print("🚀 Quick Start: Hugging Face Models with Chemistry Agents")
    print("=" * 60)
    
    # 1. SIMPLEST USAGE - Just specify the HF model
    print("\n1️⃣ Basic Usage:")
    agent = PropertyPredictionAgent(
        property_name="logP",
        model_type="transformer",
        transformer_model="DeepChem/ChemBERTa-77M-MLM"  # Any HF model
    )
    
    # Mock loading for demo
    agent.is_loaded = True
    
    # Predict for a molecule
    result = agent.predict_single("CCO")  # Ethanol
    print(f"   LogP prediction for ethanol: {result.prediction:.2f}")
    
    # 2. USING YOUR FINE-TUNED MODEL
    print("\n2️⃣ Your Fine-tuned Model:")
    
    # Option A: From Hugging Face Hub
    your_model = PropertyPredictionAgent(
        property_name="solubility",
        model_type="transformer",
        transformer_model="your-username/chembert-solubility"  # Your HF model
    )
    
    # Option B: From local files
    local_model = PropertyPredictionAgent(
        property_name="solubility", 
        model_type="transformer",
        transformer_model="./my-models/chembert-finetuned"  # Local path
    )
    
    print("   ✓ Configured agents with custom models")
    
    # 3. SPECIALIZED AGENTS WITH HF MODELS
    print("\n3️⃣ Specialized Agents:")
    
    # Solubility agent with your model
    solubility_agent = SolubilityAgent(
        model_type="transformer",
        transformer_model="DeepChem/ChemBERTa-77M-MLM"
    )
    
    # Toxicity agent with your model  
    toxicity_agent = ToxicityAgent(
        model_type="transformer",
        transformer_model="your-username/chembert-toxicity"
    )
    
    print("   ✓ Specialized agents configured")
    
    # 4. BATCH PROCESSING
    print("\n4️⃣ Batch Processing:")
    
    molecules = ["CCO", "CCC", "CCCC", "C1=CC=CC=C1"]
    
    # Mock for demo
    solubility_agent.is_loaded = True
    results = solubility_agent.predict_batch(molecules)
    
    print("   Batch predictions:")
    for smiles, result in zip(molecules, results):
        print(f"     {smiles}: {result.prediction:.2f}")

def available_models():
    """Show available models for different tasks"""
    
    print("\n📚 Available Hugging Face Models for Chemistry:")
    print("=" * 50)
    
    models = {
        "General Chemistry": [
            "DeepChem/ChemBERTa-77M-MLM",
            "DeepChem/ChemBERTa-10M-MLM", 
            "DeepChem/ChemBERTa-5M-MLM"
        ],
        "Molecular Generation": [
            "laituan245/molt5-small",
            "laituan245/molt5-base"
        ],
        "Scientific Text": [
            "allenai/scibert_scivocab_uncased",
            "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract"
        ],
        "Custom Fine-tuned": [
            "your-username/chembert-solubility",
            "your-username/chembert-toxicity",
            "your-username/chembert-bioactivity"
        ]
    }
    
    for category, model_list in models.items():
        print(f"\n{category}:")
        for model in model_list:
            print(f"  • {model}")

def model_performance_comparison():
    """Compare different model sizes/types"""
    
    print("\n⚡ Model Performance Comparison:")
    print("=" * 40)
    
    # Simulated performance data
    models = [
        {"name": "ChemBERTa-5M", "params": "5M", "speed": "Fast", "accuracy": "Good"},
        {"name": "ChemBERTa-77M", "params": "77M", "speed": "Medium", "accuracy": "Better"}, 
        {"name": "Your Fine-tuned", "params": "77M", "speed": "Medium", "accuracy": "Best*"},
    ]
    
    print(f"{'Model':<20} {'Parameters':<12} {'Speed':<10} {'Accuracy':<10}")
    print("-" * 55)
    
    for model in models:
        print(f"{model['name']:<20} {model['params']:<12} {model['speed']:<10} {model['accuracy']:<10}")
    
    print("\n* Best for your specific task/data")

def fine_tuning_workflow():
    """Show the fine-tuning workflow"""
    
    print("\n🎯 Fine-tuning Workflow:")
    print("=" * 30)
    
    steps = [
        "1. Prepare your dataset (SMILES + property values)",
        "2. Run: python scripts/fine_tune_transformer.py --data_path your_data.csv",
        "3. Model saves to ./fine_tuned_models/",
        "4. Upload to Hugging Face Hub (optional)",
        "5. Use with agents: transformer_model='your-username/model-name'"
    ]
    
    for step in steps:
        print(f"   {step}")
    
    print("\n   Example command:")
    print("   python scripts/fine_tune_transformer.py \\")
    print("     --data_path data/solubility.csv \\")
    print("     --model_name DeepChem/ChemBERTa-77M-MLM \\")
    print("     --epochs 10 \\")
    print("     --output_dir ./models/solubility")

def real_world_example():
    """Real-world usage example"""
    
    print("\n🌍 Real-world Example:")
    print("=" * 25)
    
    print("Scenario: Drug discovery pipeline with fine-tuned models")
    
    # Step 1: Screen compounds for solubility
    print("\n   Step 1: Solubility screening")
    print("   Model: your-username/chembert-solubility-esol")
    
    # Step 2: Toxicity assessment
    print("\n   Step 2: Toxicity assessment") 
    print("   Model: your-username/chembert-toxicity-tox21")
    
    # Step 3: Bioactivity prediction
    print("\n   Step 3: Bioactivity prediction")
    print("   Model: your-username/chembert-bioactivity-chembl")
    
    # Show code
    print("\n   Code:")
    print("   ```python")
    print("   from chemistry_agents import DrugDiscoveryAgent")
    print("   ")
    print("   # Use your fine-tuned models")
    print("   agent = DrugDiscoveryAgent()")
    print("   agent.solubility_agent.transformer_model = 'user/chembert-solubility'")
    print("   agent.toxicity_agent.transformer_model = 'user/chembert-toxicity'")
    print("   ")
    print("   # Screen compound library")
    print("   results = agent.screen_compound_library(compounds)")
    print("   ```")

def main():
    """Run the quick start guide"""
    
    quick_start()
    available_models()
    model_performance_comparison()
    fine_tuning_workflow()
    real_world_example()
    
    print("\n" + "=" * 60)
    print("🎉 You're ready to use Hugging Face models with Chemistry Agents!")
    print("\n💡 Pro Tips:")
    print("   • Start with DeepChem/ChemBERTa-77M-MLM for best results")
    print("   • Fine-tune on your specific task/dataset")
    print("   • Use smaller models (5M) for faster inference")
    print("   • Upload your models to HF Hub for easy sharing")
    print("\n📖 Next: Check out examples/huggingface_models.py for more details")

if __name__ == "__main__":
    main()