#!/usr/bin/env python3
"""
Advanced usage examples for Chemistry Agents
"""

import sys
import os
import pandas as pd
import numpy as np
from typing import List, Dict, Any

# Add src to path for development
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from chemistry_agents import (
    PropertyPredictionAgent, 
    SolubilityAgent, 
    ToxicityAgent, 
    DrugDiscoveryAgent,
    AgentConfig
)
from chemistry_agents.utils.data_processing import DataProcessor, DataAugmentation
from chemistry_agents.utils.evaluation import ModelEvaluator, MetricsCalculator
from chemistry_agents.models.molecular_predictor import MolecularFeatureExtractor

def example_custom_agent():
    """Example: Creating a custom specialized agent"""
    print("=== Custom Agent Development Example ===")
    
    from chemistry_agents.agents.base_agent import BaseChemistryAgent, PredictionResult
    
    class PermeabilityAgent(BaseChemistryAgent):
        """Custom agent for blood-brain barrier permeability prediction"""
        
        def __init__(self, config=None):
            super().__init__(config)
            self.property_name = "bbb_permeability"
            self.feature_extractor = MolecularFeatureExtractor()
        
        def load_model(self, model_path=None):
            # In real implementation, load trained model
            self.is_loaded = True
            self.logger.info("BBB permeability model loaded")
        
        def predict_single(self, smiles: str) -> PredictionResult:
            if not self.is_loaded:
                raise RuntimeError("Model not loaded")
            
            # Extract features (simplified for demo)
            features = self.feature_extractor.extract_features(smiles)
            
            if features is None:
                return PredictionResult(
                    smiles=smiles,
                    prediction=0.0,
                    confidence=0.0,
                    additional_info={"error": "Invalid SMILES"}
                )
            
            # Simplified prediction logic (replace with real model)
            # BBB permeability often correlates with molecular weight and polarity
            mw_feature = features[0] if len(features) > 0 else 200  # Approximate MW
            prediction = max(0, min(1, 1 - (mw_feature - 150) / 500))  # Simplified
            
            # Classify permeability
            if prediction > 0.7:
                permeability_class = "high"
            elif prediction > 0.3:
                permeability_class = "moderate"
            else:
                permeability_class = "low"
            
            return PredictionResult(
                smiles=smiles,
                prediction=prediction,
                confidence=0.85,
                additional_info={
                    "permeability_class": permeability_class,
                    "property_name": self.property_name,
                    "interpretation": f"{permeability_class.title()} BBB permeability expected"
                }
            )
    
    # Use custom agent
    print("Creating custom BBB permeability agent...")
    
    bbb_agent = PermeabilityAgent()
    bbb_agent.load_model()
    
    # Test molecules
    test_molecules = [
        "CCO",  # Ethanol (small, should cross)
        "CC(C)(C)C1=CC=C(C=C1)C(=O)O",  # Large molecule
        "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",  # Caffeine
    ]
    
    molecule_names = ["Ethanol", "Large molecule", "Caffeine"]
    
    print("\nBBB Permeability Predictions:")
    for name, smiles in zip(molecule_names, test_molecules):
        result = bbb_agent.predict_single(smiles)
        print(f"  {name:15s}: {result.prediction:.3f} ({result.additional_info.get('permeability_class', 'unknown')})")

def example_data_processing_pipeline():
    """Example: Advanced data processing and preparation"""
    print("\n=== Data Processing Pipeline Example ===")
    
    # Create sample dataset
    sample_data = {
        'smiles': [
            'CCO',
            'CCC',
            'CC(C)C',
            'CCCC',
            'CC(C)CC',
            'CCCCC',
            'CC(C)CCC',
            'CCCCCC',
            'CC(C)CCCC',
            'CCCCCCC',
            'CC(=O)O',  # Acetic acid
            'CCC(=O)O',  # Propionic acid
            'CCCC(=O)O',  # Butyric acid
            'C1=CC=CC=C1',  # Benzene
            'CC1=CC=CC=C1',  # Toluene
        ],
        'logP': [
            -0.31, 0.25, 0.26, 0.88, 1.09, 1.51, 1.98, 2.13, 2.56, 3.0,
            -0.17, 0.33, 0.79, 2.13, 2.73
        ],
        'solubility': [
            0.2, -0.5, -0.8, -1.2, -1.5, -2.0, -2.3, -2.8, -3.1, -3.5,
            1.2, 0.8, 0.3, -2.1, -2.4
        ]
    }
    
    df = pd.DataFrame(sample_data)
    print(f"Created sample dataset with {len(df)} compounds")
    
    # Initialize data processor
    processor = DataProcessor(
        scaler_type="standard",
        remove_duplicates=True,
        handle_missing="drop"
    )
    
    # Analyze dataset
    analysis = processor.analyze_dataset(df, target_column="logP")
    print(f"\nDataset Analysis:")
    print(f"  Dataset size: {analysis['dataset_size']}")
    print(f"  Target statistics: {analysis['target_statistics']}")
    
    # Clean and split dataset
    df_clean = processor.clean_dataset(df, "smiles", "logP")
    train_df, val_df, test_df = processor.split_dataset(
        df_clean, test_size=0.3, val_size=0.2, random_state=42
    )
    
    print(f"\nDataset splits:")
    print(f"  Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    
    # Create molecular descriptors
    train_with_descriptors = processor.create_molecular_descriptors(train_df, "smiles")
    print(f"  Created molecular descriptors: {train_with_descriptors.shape[1] - len(train_df.columns)} features")
    
    # Data augmentation example
    print(f"\nData Augmentation:")
    augmenter = DataAugmentation()
    
    # Augment SMILES for training
    original_smiles = "CC(C)C1=CC=C(C=C1)C(C)C(=O)O"  # Ibuprofen
    augmented_smiles = augmenter.augment_smiles(original_smiles, num_augmentations=3)
    
    print(f"  Original SMILES: {original_smiles}")
    print(f"  Augmented variants:")
    for i, aug_smiles in enumerate(augmented_smiles[1:], 1):  # Skip original
        print(f"    {i}: {aug_smiles}")

def example_comprehensive_evaluation():
    """Example: Comprehensive model evaluation and benchmarking"""
    print("\n=== Comprehensive Model Evaluation Example ===")
    
    # Generate synthetic prediction data for demonstration
    np.random.seed(42)
    n_samples = 100
    
    # Simulate true values and predictions
    y_true = np.random.normal(2.0, 1.5, n_samples)  # True logP values
    y_pred = y_true + np.random.normal(0, 0.3, n_samples)  # Predictions with noise
    
    # Add some systematic bias to make it realistic
    y_pred = y_pred * 0.95 + 0.1
    
    print(f"Evaluating model with {n_samples} predictions...")
    
    # Initialize evaluator
    evaluator = ModelEvaluator(task_type="regression")
    
    # Calculate metrics
    metrics = evaluator.evaluate_predictions(y_true, y_pred)
    
    print(f"\nRegression Metrics:")
    for metric_name, value in metrics.items():
        print(f"  {metric_name:20s}: {value:.4f}")
    
    # Generate evaluation report
    report = evaluator.generate_evaluation_report(
        y_true, y_pred, model_name="Demo LogP Predictor"
    )
    
    print(f"\nPerformance Summary: {report['performance_summary']}")
    
    # Chemistry-specific metrics
    print(f"\nChemistry-Specific Metrics:")
    
    metrics_calc = MetricsCalculator()
    
    # Sample molecules for diversity analysis
    sample_smiles = [
        "CCO", "CCC", "CCCC", "CCCCC",
        "C1=CC=CC=C1", "CC1=CC=CC=C1", "CCC1=CC=CC=C1",
        "CCN", "CCCN", "CCCCN",
        "CC(=O)O", "CCC(=O)O", "CCCC(=O)O"
    ]
    
    # Calculate molecular diversity
    diversity_metrics = metrics_calc.calculate_molecular_diversity(sample_smiles)
    
    if "error" not in diversity_metrics:
        print(f"  Molecular diversity index: {diversity_metrics['diversity_index']:.3f}")
        print(f"  Mean Tanimoto similarity: {diversity_metrics['mean_similarity']:.3f}")
    
    # Calculate scaffold diversity
    scaffold_metrics = metrics_calc.calculate_scaffold_diversity(sample_smiles)
    
    if "error" not in scaffold_metrics:
        print(f"  Unique scaffolds: {scaffold_metrics['unique_scaffolds']}")
        print(f"  Scaffold diversity ratio: {scaffold_metrics['scaffold_diversity_ratio']:.3f}")

def example_multi_property_prediction():
    """Example: Multi-property prediction and correlation analysis"""
    print("\n=== Multi-Property Prediction Example ===")
    
    # Test molecules
    test_molecules = [
        ("Aspirin", "CC(=O)OC1=CC=CC=C1C(=O)O"),
        ("Ibuprofen", "CC(C)C1=CC=C(C=C1)C(C)C(=O)O"),
        ("Caffeine", "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"),
        ("Paracetamol", "CC(=O)NC1=CC=C(C=C1)O"),
        ("Morphine", "CN1CCC23C4C1CC5=C2C(=C(C=C5)O)OC3C(C=C4)O")
    ]
    
    # Properties to predict
    properties = [
        ("logP", "transformer"),
        ("solubility", "neural_network"),
        ("molecular_weight", "neural_network")
    ]
    
    print(f"Predicting {len(properties)} properties for {len(test_molecules)} molecules:")
    
    # Initialize agents for different properties
    agents = {}
    for prop_name, model_type in properties:
        agent = PropertyPredictionAgent(
            property_name=prop_name,
            model_type=model_type
        )
        try:
            agent.load_model()
        except:
            agent.is_loaded = True  # Mock for demo
        agents[prop_name] = agent
    
    # Predict all properties
    results_matrix = []
    
    for mol_name, smiles in test_molecules:
        mol_results = {"name": mol_name, "smiles": smiles}
        
        for prop_name in agents.keys():
            result = agents[prop_name].predict_single(smiles)
            mol_results[prop_name] = result.prediction
            mol_results[f"{prop_name}_confidence"] = result.confidence or 0.8
        
        results_matrix.append(mol_results)
    
    # Display results in table format
    print(f"\n{'Molecule':<12} {'logP':<8} {'Solub.':<8} {'MW':<8}")
    print("-" * 40)
    
    for result in results_matrix:
        print(f"{result['name']:<12} {result['logP']:<8.2f} {result['solubility']:<8.2f} {result['molecular_weight']:<8.1f}")
    
    # Correlation analysis (simplified)
    print(f"\nProperty Correlations (simplified demo):")
    logp_values = [r['logP'] for r in results_matrix]
    sol_values = [r['solubility'] for r in results_matrix]
    
    correlation = np.corrcoef(logp_values, sol_values)[0, 1]
    print(f"  logP vs Solubility correlation: {correlation:.3f}")

def example_virtual_screening():
    """Example: Virtual screening workflow"""
    print("\n=== Virtual Screening Example ===")
    
    # Simulate a compound library
    compound_library = {
        "ZINC001": "CC1=CC=C(C=C1)C2=CC(=NN2C3=CC=C(C=C3)S(=O)(=O)N)C(F)(F)F",
        "ZINC002": "CCN(CC)CCNC(=O)C1=CC=C(C=C1)N", 
        "ZINC003": "CC(C)C1=CC=C(C=C1)C(C)C(=O)O",
        "ZINC004": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
        "ZINC005": "CC1=CC=CC=C1",
        "ZINC006": "CCCCCCCCCCCCCCCCCC(=O)O",  # Fatty acid (not drug-like)
        "ZINC007": "C1=CC=CC=C1",  # Benzene (toxic)
        "ZINC008": "CCO",  # Ethanol
        "ZINC009": "CC(=O)OC1=CC=CC=C1C(=O)O",  # Aspirin
        "ZINC010": "CN(C)CCC=C1C2=CC=CC=C2SC3=C1C=C(C=C3)Cl",
    }
    
    print(f"Virtual screening of {len(compound_library)} compounds...")
    
    # Initialize drug discovery agent for screening
    dd_agent = DrugDiscoveryAgent()
    
    try:
        dd_agent.load_model()
    except:
        # Mock setup for demo
        dd_agent.is_loaded = True
        dd_agent.solubility_agent.is_loaded = True
        dd_agent.toxicity_agent.is_loaded = True
        dd_agent.bioactivity_agent.is_loaded = True
    
    # Screen the library
    smiles_list = list(compound_library.values())
    
    screening_results = dd_agent.screen_compound_library(
        smiles_list,
        criteria={
            "min_discovery_score": 40,
            "require_lipinski_compliance": False,  # Relaxed for demo
            "max_toxicity_risk": "high"
        }
    )
    
    print(f"\nScreening Results:")
    print(f"  Compounds screened: {screening_results['total_compounds']}")
    print(f"  Hits identified: {screening_results['promising_compounds']}")
    print(f"  Hit rate: {screening_results['hit_rate']:.1%}")
    
    # Show top hits
    print(f"\nTop Hits:")
    for i, hit in enumerate(screening_results['top_candidates'][:5], 1):
        # Find compound ID
        compound_id = None
        for cid, smi in compound_library.items():
            if smi == hit['smiles']:
                compound_id = cid
                break
        
        print(f"  {i}. {compound_id or 'Unknown'}")
        print(f"     Discovery Score: {hit['discovery_score']:.1f}")
        print(f"     Solubility: {hit['solubility_class']}")
        print(f"     Recommendation: {hit['recommendation']}")
        print()

def example_model_comparison():
    """Example: Comparing different model architectures"""
    print("\n=== Model Architecture Comparison Example ===")
    
    # Test molecules
    test_smiles = [
        "CCO", "CCC", "CCCC", "C1=CC=CC=C1", "CC1=CC=CC=C1",
        "CC(=O)O", "CCN", "CC(C)C", "CCCCC", "CC(=O)OC1=CC=CC=C1C(=O)O"
    ]
    
    print(f"Comparing model architectures on {len(test_smiles)} molecules...")
    
    # Initialize different model types
    models = {
        "Neural Network": PropertyPredictionAgent(
            property_name="logP",
            model_type="neural_network"
        ),
        "Transformer": PropertyPredictionAgent(
            property_name="logP", 
            model_type="transformer"
        )
    }
    
    # Mock model loading for demo
    for model_name, agent in models.items():
        try:
            agent.load_model()
        except:
            agent.is_loaded = True
    
    # Compare predictions
    comparison_results = {}
    
    for model_name, agent in models.items():
        results = agent.predict_batch(test_smiles)
        predictions = [r.prediction for r in results]
        confidences = [r.confidence or 0.8 for r in results]
        
        comparison_results[model_name] = {
            "predictions": predictions,
            "confidences": confidences,
            "mean_confidence": np.mean(confidences)
        }
    
    # Display comparison
    print(f"\n{'SMILES':<20} {'Neural Net':<12} {'Transformer':<12}")
    print("-" * 50)
    
    for i, smiles in enumerate(test_smiles[:5]):  # Show first 5
        nn_pred = comparison_results["Neural Network"]["predictions"][i]
        trans_pred = comparison_results["Transformer"]["predictions"][i]
        print(f"{smiles:<20} {nn_pred:<12.2f} {trans_pred:<12.2f}")
    
    # Summary statistics
    print(f"\nModel Performance Summary:")
    for model_name, results in comparison_results.items():
        print(f"  {model_name}:")
        print(f"    Mean confidence: {results['mean_confidence']:.3f}")
        print(f"    Prediction range: {min(results['predictions']):.2f} to {max(results['predictions']):.2f}")

def main():
    """Run all advanced examples"""
    print("Chemistry Agents - Advanced Usage Examples")
    print("=" * 60)
    
    # Run advanced examples
    example_custom_agent()
    example_data_processing_pipeline()
    example_comprehensive_evaluation()
    example_multi_property_prediction()
    example_virtual_screening()
    example_model_comparison()
    
    print("\n" + "=" * 60)
    print("Advanced examples completed!")
    print("\nThese examples demonstrate:")
    print("- Custom agent development")
    print("- Advanced data processing pipelines")
    print("- Comprehensive model evaluation")
    print("- Multi-property prediction workflows")
    print("- Virtual screening applications")
    print("- Model architecture comparisons")

if __name__ == "__main__":
    main()