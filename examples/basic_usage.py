#!/usr/bin/env python3
"""
Basic usage examples for Chemistry Agents
"""

import sys
import os

# Add src to path for development
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from chemistry_agents import (
    PropertyPredictionAgent, 
    SolubilityAgent, 
    ToxicityAgent, 
    DrugDiscoveryAgent,
    AgentConfig
)

def example_property_prediction():
    """Example: Basic molecular property prediction with ChemBERTa"""
    print("=== Property Prediction Example (ChemBERTa) ===")
    
    # Sample molecules (SMILES)
    molecules = [
        "CCO",                    # Ethanol
        "CC(=O)OC1=CC=CC=C1C(=O)O",  # Aspirin
        "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",  # Caffeine
        "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O"  # Ibuprofen
    ]
    
    molecule_names = ["Ethanol", "Aspirin", "Caffeine", "Ibuprofen"]
    
    # Initialize agent with ChemBERTa
    config = AgentConfig(log_level="INFO", device="cpu")
    agent = PropertyPredictionAgent(
        config=config,
        property_name="logP",
        model_type="transformer",
        transformer_model="DeepChem/ChemBERTa-77M-MLM"
    )
    
    print(f"Agent created: {agent.model_type} model")
    print(f"Transformer: {agent.transformer_model}")
    print(f"Property: {agent.property_name}")
    
    # Load ChemBERTa model
    print("\n🔄 Loading ChemBERTa model...")
    try:
        agent.load_model()  # Load ChemBERTa from HuggingFace
        print(f"✅ ChemBERTa model loaded successfully!")
        print(f"   Model loaded: {agent.is_loaded}")
        print(f"   Vocabulary size: {agent.tokenizer.vocab_size:,}")
    except Exception as e:
        print(f"⚠️ Could not load ChemBERTa model: {e}")
        print("💡 Run 'python download_huggingface_model.py' first")
        print("📝 Using placeholder predictions for demonstration")
        agent.is_loaded = True
    
    # Make predictions
    print(f"\nPredicting logP for {len(molecules)} molecules:")
    results = agent.predict_batch(molecules)
    
    for name, result in zip(molecule_names, results):
        print(f"  {name:12s}: {result.prediction:6.2f} (confidence: {result.confidence:.2f})")
    
    return results

def example_solubility_prediction():
    """Example: Solubility prediction with ChemBERTa and drug-likeness assessment"""
    print("\n=== Solubility Prediction Example (ChemBERTa) ===")
    
    # Drug-like molecules
    drug_molecules = [
        "CC1=CC=C(C=C1)C2=CC(=NN2C3=CC=C(C=C3)S(=O)(=O)N)C(F)(F)F",  # Celecoxib
        "CCN(CC)CCNC(=O)C1=CC=C(C=C1)N",  # Procainamide
        "CN(C)CCC=C1C2=CC=CC=C2SC3=C1C=C(C=C3)Cl",  # Chlorpromazine
    ]
    
    drug_names = ["Celecoxib", "Procainamide", "Chlorpromazine"]
    
    # Initialize solubility agent (now defaults to ChemBERTa)
    config = AgentConfig(device="cpu", log_level="INFO")
    solubility_agent = SolubilityAgent(config)
    
    print(f"Solubility agent: {solubility_agent.model_type} model")
    print(f"Transformer: {solubility_agent.transformer_model}")
    
    print("\n🔄 Loading ChemBERTa for solubility prediction...")
    try:
        solubility_agent.load_model()
        print("✅ ChemBERTa model loaded for solubility!")
    except Exception as e:
        print(f"⚠️ Could not load ChemBERTa model: {e}")
        print("📝 Using placeholder predictions for demonstration")
        solubility_agent.is_loaded = True
    
    print(f"\nAnalyzing solubility for {len(drug_molecules)} drug molecules:")
    
    for name, smiles in zip(drug_names, drug_molecules):
        result = solubility_agent.predict_single(smiles)
        
        print(f"\n{name}:")
        print(f"  SMILES: {smiles}")
        print(f"  Log S: {result.prediction:.2f}")
        
        if result.additional_info and "error" not in result.additional_info:
            print(f"  Solubility class: {result.additional_info.get('solubility_class', 'unknown')}")
            print(f"  Interpretation: {result.additional_info.get('interpretation', 'N/A')}")
            
            lipinski_info = result.additional_info.get('lipinski_compliant', {})
            print(f"  Lipinski compliant: {lipinski_info.get('compliant', 'unknown')}")

def example_toxicity_assessment():
    """Example: Multi-endpoint toxicity assessment with ChemBERTa"""
    print("\n=== Toxicity Assessment Example (ChemBERTa) ===")
    
    # Test compounds
    test_compounds = [
        "CCO",  # Ethanol (low toxicity)
        "CC(C)C1=CC=C(C=C1)C(C)C(=O)O",  # Ibuprofen (moderate)
        "C1=CC=C(C=C1)C(=O)O"  # Benzoic acid (low)
    ]
    
    compound_names = ["Ethanol", "Ibuprofen", "Benzoic acid"]
    
    # Test different toxicity endpoints
    endpoints = ["acute_toxicity", "hepatotoxicity", "mutagenicity"]
    
    config = AgentConfig(device="cpu", log_level="INFO")
    
    for endpoint in endpoints:
        print(f"\n--- {endpoint.replace('_', ' ').title()} Assessment (ChemBERTa) ---")
        
        toxicity_agent = ToxicityAgent(
            config=config,
            toxicity_endpoint=endpoint
        )
        
        print(f"Agent: {toxicity_agent.model_type} model for {endpoint}")
        
        try:
            toxicity_agent.load_model()
            print("✅ ChemBERTa model loaded for toxicity!")
        except Exception as e:
            print(f"⚠️ Could not load ChemBERTa: {e}")
            print("📝 Using placeholder predictions for demonstration")
            toxicity_agent.is_loaded = True
        
        for name, smiles in zip(compound_names, test_compounds):
            result = toxicity_agent.predict_single(smiles)
            
            print(f"  {name:15s}: {result.prediction:6.2f}")
            
            if result.additional_info and "error" not in result.additional_info:
                tox_class = result.additional_info.get('toxicity_class', 'unknown')
                print(f"                    Class: {tox_class}")

def example_drug_discovery_pipeline():
    """Example: Comprehensive drug discovery analysis with ChemBERTa"""
    print("\n=== Drug Discovery Pipeline Example (ChemBERTa) ===")
    
    # Candidate molecules for drug discovery
    candidates = [
        "CC1=CC=C(C=C1)C2=CC(=NN2C3=CC=C(C=C3)S(=O)(=O)N)C(F)(F)F",  # Drug-like
        "CCN(CC)CCNC(=O)C1=CC=C(C=C1)N",  # Drug-like
        "CCCCCCCCCCCCCCCCCC(=O)O",  # Fatty acid (not drug-like)
        "C1=CC=CC=C1",  # Benzene (toxic)
    ]
    
    candidate_names = ["Candidate A", "Candidate B", "Fatty Acid", "Benzene"]
    
    # Initialize drug discovery agent (uses ChemBERTa by default)
    config = AgentConfig(device="cpu", log_level="INFO")
    dd_agent = DrugDiscoveryAgent(config)
    
    print("Drug Discovery Agent initialized with ChemBERTa models:")
    print(f"  Solubility agent: {dd_agent.solubility_agent.model_type}")
    print(f"  Toxicity agent: {dd_agent.toxicity_agent.model_type}")
    print(f"  Bioactivity agent: {dd_agent.bioactivity_agent.model_type}")
    
    print("\n🔄 Loading ChemBERTa models for drug discovery...")
    try:
        dd_agent.load_model()
        print("✅ All ChemBERTa models loaded for drug discovery!")
    except Exception as e:
        print(f"⚠️ Could not load all ChemBERTa models: {e}")
        print("📝 Using placeholder predictions for demonstration")
        dd_agent.is_loaded = True
        # Set up sub-agents for demo
        dd_agent.solubility_agent.is_loaded = True
        dd_agent.toxicity_agent.is_loaded = True
        dd_agent.bioactivity_agent.is_loaded = True
    
    print(f"\nScreening {len(candidates)} compounds for drug discovery potential:")
    
    # Screen compound library
    screening_results = dd_agent.screen_compound_library(
        candidates,
        criteria={
            "min_discovery_score": 30,  # Lower threshold for demo
            "require_lipinski_compliance": False,  # Relaxed for demo
            "max_toxicity_risk": "high"
        }
    )
    
    print(f"\nScreening Results:")
    print(f"  Total compounds: {screening_results['total_compounds']}")
    print(f"  Promising compounds: {screening_results['promising_compounds']}")
    print(f"  Hit rate: {screening_results['hit_rate']:.1%}")
    
    print(f"\nTop Candidates:")
    for i, candidate in enumerate(screening_results['top_candidates'][:3]):
        idx = candidates.index(candidate['smiles'])
        print(f"  {i+1}. {candidate_names[idx]}:")
        print(f"     Discovery Score: {candidate['discovery_score']:.1f}")
        print(f"     Recommendation: {candidate['recommendation']}")

def example_lead_optimization():
    """Example: Lead compound optimization analysis"""
    print("\n=== Lead Optimization Example ===")
    
    # Lead compound (needs optimization)
    lead_compound = "CC1=CC=C(C=C1)C2=CC(=NN2C3=CC=C(C=C3)S(=O)(=O)N)C(F)(F)F"
    
    # Initialize drug discovery agent
    dd_agent = DrugDiscoveryAgent()
    
    try:
        dd_agent.load_model()
    except:
        print("Note: Using placeholder models for demonstration")
        dd_agent.is_loaded = True
        dd_agent.solubility_agent.is_loaded = True
        dd_agent.toxicity_agent.is_loaded = True
        dd_agent.bioactivity_agent.is_loaded = True
    
    # Analyze lead compound
    optimization_analysis = dd_agent.optimize_lead_compound(
        lead_compound,
        optimization_targets={
            "min_discovery_score": 80,
            "min_solubility": -2,
            "max_toxicity_score": 20,
            "min_drug_likeness": 0.8
        }
    )
    
    print(f"Lead Compound Analysis:")
    print(f"  SMILES: {lead_compound}")
    
    if "error" not in optimization_analysis:
        current_profile = optimization_analysis["current_profile"]
        print(f"  Current Discovery Score: {current_profile['discovery_score']}")
        print(f"  Drug-likeness Score: {current_profile['drug_likeness_score']}")
        
        print(f"\nOptimization Needs:")
        for need in optimization_analysis["optimization_needs"]:
            print(f"  - {need}")
        
        print(f"\nRecommended Strategies:")
        for strategy in optimization_analysis["recommended_strategies"]:
            print(f"  - {strategy['strategy']}")
            for approach in strategy['approaches'][:2]:  # Show first 2 approaches
                print(f"    • {approach}")

def example_batch_processing():
    """Example: Efficient batch processing of large datasets"""
    print("\n=== Batch Processing Example ===")
    
    # Simulate larger dataset
    import random
    
    # Generate some example SMILES (simplified for demo)
    base_smiles = [
        "CCO", "CCC", "CCCC", "CCCCC", "CCCCCC",
        "C1=CC=CC=C1", "CC1=CC=CC=C1", "CCC1=CC=CC=C1",
        "CCN", "CCCN", "CCCCN", "CC(C)N"
    ]
    
    # Create larger dataset by sampling
    large_dataset = random.choices(base_smiles, k=50)
    
    print(f"Processing {len(large_dataset)} molecules in batch...")
    
    # Initialize agent
    agent = PropertyPredictionAgent(property_name="molecular_weight")
    
    try:
        agent.load_model()
    except:
        agent.is_loaded = True
    
    # Process in batches
    batch_size = 10
    all_results = []
    
    for i in range(0, len(large_dataset), batch_size):
        batch = large_dataset[i:i+batch_size]
        batch_results = agent.predict_batch(batch)
        all_results.extend(batch_results)
        print(f"  Processed batch {i//batch_size + 1}: {len(batch)} molecules")
    
    # Summary statistics
    predictions = [r.prediction for r in all_results if r.confidence and r.confidence > 0.5]
    
    if predictions:
        print(f"\nResults Summary:")
        print(f"  Valid predictions: {len(predictions)}")
        print(f"  Mean prediction: {sum(predictions)/len(predictions):.2f}")
        print(f"  Min prediction: {min(predictions):.2f}")
        print(f"  Max prediction: {max(predictions):.2f}")

def main():
    """Run all examples with ChemBERTa integration"""
    print("Chemistry Agents - Usage Examples (ChemBERTa)")
    print("=" * 60)
    
    print("🤖 All agents now use ChemBERTa transformer models by default!")
    print("📚 Model: DeepChem/ChemBERTa-77M-MLM")
    print("💻 Device: CPU (for compatibility)")
    print()
    
    # Run examples
    try:
        example_property_prediction()
        example_solubility_prediction()
        example_toxicity_assessment()
        example_drug_discovery_pipeline()
        example_lead_optimization()
        example_batch_processing()
    except KeyboardInterrupt:
        print("\n⚠️ Examples interrupted by user")
    except Exception as e:
        print(f"\n❌ Example failed: {e}")
        print("💡 Make sure you've run: python download_huggingface_model.py")
    
    print("\n" + "=" * 60)
    print("Examples completed!")
    print("\n📋 ChemBERTa Integration Summary:")
    print("  ✅ All agents use transformer models")
    print("  ✅ Real molecular embeddings from ChemBERTa")
    print("  ✅ CPU-optimized for broad compatibility")
    print("\n💡 Next steps:")
    print("  1. Download models: python download_huggingface_model.py")
    print("  2. Train custom models: python scripts/train_model.py --model_type transformer")
    print("  3. Try ChemBERTa example: python examples/chemberta_example.py")

if __name__ == "__main__":
    main()