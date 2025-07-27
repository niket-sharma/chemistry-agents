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
    """Example: Basic molecular property prediction"""
    print("=== Property Prediction Example ===")
    
    # Sample molecules (SMILES)
    molecules = [
        "CCO",                    # Ethanol
        "CC(=O)OC1=CC=CC=C1C(=O)O",  # Aspirin
        "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",  # Caffeine
        "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O"  # Ibuprofen
    ]
    
    molecule_names = ["Ethanol", "Aspirin", "Caffeine", "Ibuprofen"]
    
    # Initialize agent
    config = AgentConfig(log_level="INFO")
    agent = PropertyPredictionAgent(
        config=config,
        property_name="logP",
        model_type="transformer"
    )
    
    # Load pre-trained model (in real usage, you'd have trained models)
    try:
        agent.load_model()  # This would load a real model
    except:
        print("Note: Using placeholder model for demonstration")
        agent.is_loaded = True
    
    # Make predictions
    print(f"\nPredicting logP for {len(molecules)} molecules:")
    results = agent.predict_batch(molecules)
    
    for name, result in zip(molecule_names, results):
        print(f"  {name:12s}: {result.prediction:6.2f} (confidence: {result.confidence:.2f})")
    
    return results

def example_solubility_prediction():
    """Example: Solubility prediction with drug-likeness assessment"""
    print("\n=== Solubility Prediction Example ===")
    
    # Drug-like molecules
    drug_molecules = [
        "CC1=CC=C(C=C1)C2=CC(=NN2C3=CC=C(C=C3)S(=O)(=O)N)C(F)(F)F",  # Celecoxib
        "CCN(CC)CCNC(=O)C1=CC=C(C=C1)N",  # Procainamide
        "CN(C)CCC=C1C2=CC=CC=C2SC3=C1C=C(C=C3)Cl",  # Chlorpromazine
    ]
    
    drug_names = ["Celecoxib", "Procainamide", "Chlorpromazine"]
    
    # Initialize solubility agent
    solubility_agent = SolubilityAgent()
    
    try:
        solubility_agent.load_model()
    except:
        print("Note: Using placeholder model for demonstration")
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
    """Example: Multi-endpoint toxicity assessment"""
    print("\n=== Toxicity Assessment Example ===")
    
    # Test compounds
    test_compounds = [
        "CCO",  # Ethanol (low toxicity)
        "CC(C)C1=CC=C(C=C1)C(C)C(=O)O",  # Ibuprofen (moderate)
        "C1=CC=C(C=C1)C(=O)O"  # Benzoic acid (low)
    ]
    
    compound_names = ["Ethanol", "Ibuprofen", "Benzoic acid"]
    
    # Test different toxicity endpoints
    endpoints = ["acute_toxicity", "hepatotoxicity", "mutagenicity"]
    
    for endpoint in endpoints:
        print(f"\n--- {endpoint.replace('_', ' ').title()} Assessment ---")
        
        toxicity_agent = ToxicityAgent(toxicity_endpoint=endpoint)
        
        try:
            toxicity_agent.load_model()
        except:
            print("Note: Using placeholder model for demonstration")
            toxicity_agent.is_loaded = True
        
        for name, smiles in zip(compound_names, test_compounds):
            result = toxicity_agent.predict_single(smiles)
            
            print(f"  {name:15s}: {result.prediction:6.2f}")
            
            if result.additional_info and "error" not in result.additional_info:
                tox_class = result.additional_info.get('toxicity_class', 'unknown')
                print(f"                    Class: {tox_class}")

def example_drug_discovery_pipeline():
    """Example: Comprehensive drug discovery analysis"""
    print("\n=== Drug Discovery Pipeline Example ===")
    
    # Candidate molecules for drug discovery
    candidates = [
        "CC1=CC=C(C=C1)C2=CC(=NN2C3=CC=C(C=C3)S(=O)(=O)N)C(F)(F)F",  # Drug-like
        "CCN(CC)CCNC(=O)C1=CC=C(C=C1)N",  # Drug-like
        "CCCCCCCCCCCCCCCCCC(=O)O",  # Fatty acid (not drug-like)
        "C1=CC=CC=C1",  # Benzene (toxic)
    ]
    
    candidate_names = ["Candidate A", "Candidate B", "Fatty Acid", "Benzene"]
    
    # Initialize drug discovery agent
    dd_agent = DrugDiscoveryAgent()
    
    try:
        dd_agent.load_model()
    except:
        print("Note: Using placeholder models for demonstration")
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
    """Run all examples"""
    print("Chemistry Agents - Usage Examples")
    print("=" * 50)
    
    # Run examples
    example_property_prediction()
    example_solubility_prediction()
    example_toxicity_assessment()
    example_drug_discovery_pipeline()
    example_lead_optimization()
    example_batch_processing()
    
    print("\n" + "=" * 50)
    print("Examples completed!")
    print("\nNote: These examples use placeholder models for demonstration.")
    print("In real usage, you would train or load actual trained models.")
    print("See the training scripts in the scripts/ directory for model training.")

if __name__ == "__main__":
    main()