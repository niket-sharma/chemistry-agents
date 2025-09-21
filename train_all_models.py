#!/usr/bin/env python3
"""
Automated training script for all molecular property datasets with ChemBERTa
"""

import os
import sys
import subprocess
import pandas as pd
from pathlib import Path
import json
import time

def check_dataset_exists(filepath):
    """Check if dataset file exists and is valid"""
    if not os.path.exists(filepath):
        return False, f"File not found: {filepath}"
    
    try:
        df = pd.read_csv(filepath)
        if len(df) == 0:
            return False, f"Empty dataset: {filepath}"
        return True, f"Valid dataset with {len(df)} samples"
    except Exception as e:
        return False, f"Invalid CSV: {e}"

def train_model(data_path, target_column, model_name, property_type="regression", epochs=20):
    """Train a single ChemBERTa model"""
    print(f"\n[TRAIN] Training ChemBERTa model: {model_name}")
    print(f"   Dataset: {data_path}")
    print(f"   Target: {target_column}")
    print(f"   Type: {property_type}")
    
    # Check if dataset exists
    exists, message = check_dataset_exists(data_path)
    if not exists:
        print(f"   [ERROR] {message}")
        return False
    
    print(f"   [SUCCESS] {message}")
    
    # Prepare training command
    cmd = [
        sys.executable, 'scripts/train_model.py',
        '--data_path', data_path,
        '--target_column', target_column,
        '--model_type', 'transformer',
        '--transformer_model', 'DeepChem/ChemBERTa-77M-MLM',
        '--epochs', str(epochs),
        '--batch_size', '8',
        '--learning_rate', '5e-5',
        '--output_dir', 'models/trained',
        '--model_name', model_name,
        '--device', 'cpu',
        '--log_level', 'INFO'
    ]
    
    print(f"   [INFO] Starting training...")
    start_time = time.time()
    
    try:
        # Run training
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)  # 30 min timeout
        
        training_time = time.time() - start_time
        
        if result.returncode == 0:
            print(f"   [SUCCESS] Training completed in {training_time:.1f}s")
            
            # Check if model was saved
            model_path = f"models/trained/{model_name}.pt"
            if os.path.exists(model_path):
                print(f"   [SAVE] Model saved: {model_path}")
                return True
            else:
                print(f"   [WARNING] Training succeeded but model file not found")
                return False
        else:
            print(f"   [ERROR] Training failed after {training_time:.1f}s")
            print(f"   Error: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"   [TIMEOUT] Training timed out after 30 minutes")
        return False
    except Exception as e:
        print(f"   [ERROR] Training crashed: {e}")
        return False

def test_trained_model(model_path, model_name):
    """Test a trained model"""
    print(f"\n[TEST] Testing trained model: {model_name}")
    
    if not os.path.exists(model_path):
        print(f"   [ERROR] Model file not found: {model_path}")
        return False
    
    # Create test script
    test_script = f'''
import sys
import os
sys.path.append("src")

try:
    from chemistry_agents import SolubilityAgent
    from chemistry_agents.agents.base_agent import AgentConfig
    
    # Test loading the trained model
    config = AgentConfig(device="cpu")
    agent = SolubilityAgent(config)
    
    print("Loading trained model...")
    agent.load_model("{model_path}")
    print("Model loaded successfully!")
    
    # Test prediction
    test_smiles = ["CCO", "c1ccccc1", "CC(=O)O"]
    results = agent.predict_batch(test_smiles)
    
    print("Test predictions:")
    for smiles, result in zip(test_smiles, results):
        print(f"  {{smiles:10s}}: {{result.prediction:.4f}}")
    
    print("[SUCCESS] Model test successful!")
    
except Exception as e:
    print(f"[ERROR] Model test failed: {{e}}")
    sys.exit(1)
'''
    
    try:
        result = subprocess.run([sys.executable, '-c', test_script], 
                              capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            print(f"   [SUCCESS] Model test passed")
            print(f"   {result.stdout}")
            return True
        else:
            print(f"   [ERROR] Model test failed")
            print(f"   Error: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"   [ERROR] Model test crashed: {e}")
        return False

def create_training_summary(results):
    """Create a summary of training results"""
    summary = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_models": len(results),
        "successful_models": sum(1 for r in results if r['success']),
        "failed_models": sum(1 for r in results if not r['success']),
        "models": results
    }
    
    # Save summary
    os.makedirs("models/trained", exist_ok=True)
    summary_path = "models/trained/training_summary.json"
    
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n[SUMMARY] Training Summary saved to: {summary_path}")
    
    return summary

def main():
    """Train all molecular property models"""
    print("[PIPELINE] ChemBERTa Multi-Property Model Training Pipeline")
    print("=" * 70)
    
    # Create output directory
    os.makedirs("models/trained", exist_ok=True)
    
    # Define training configurations
    training_configs = [
        {
            "name": "solubility_chemberta",
            "data_path": "data/processed/esol_solubility.csv", 
            "target_column": "solubility_logS",
            "property_type": "regression",
            "epochs": 25
        },
        {
            "name": "toxicity_chemberta",
            "data_path": "data/processed/toxicity_dataset.csv",
            "target_column": "toxicity_score", 
            "property_type": "regression",
            "epochs": 20
        },
        {
            "name": "bioactivity_chemberta",
            "data_path": "data/processed/bioactivity_dataset.csv",
            "target_column": "bioactivity_score",
            "property_type": "regression", 
            "epochs": 20
        },
        {
            "name": "molecular_weight_chemberta",
            "data_path": "data/processed/molecular_weight_dataset.csv",
            "target_column": "molecular_weight",
            "property_type": "regression",
            "epochs": 15
        }
    ]
    
    # Check if we have any datasets
    available_configs = []
    for config in training_configs:
        exists, message = check_dataset_exists(config["data_path"])
        if exists:
            available_configs.append(config)
            print(f"[SUCCESS] {config['name']}: {message}")
        else:
            print(f"[WARNING] {config['name']}: {message}")
    
    if not available_configs:
        print("\n[ERROR] No valid datasets found!")
        print("[INFO] Run: python download_datasets.py")
        return
    
    print(f"\n[START] Training {len(available_configs)} models...")
    
    # Train all models
    results = []
    
    for i, config in enumerate(available_configs, 1):
        print(f"\n{'='*50}")
        print(f"Model {i}/{len(available_configs)}: {config['name']}")
        print(f"{'='*50}")
        
        success = train_model(
            data_path=config["data_path"],
            target_column=config["target_column"], 
            model_name=config["name"],
            property_type=config["property_type"],
            epochs=config["epochs"]
        )
        
        result = {
            "model_name": config["name"],
            "data_path": config["data_path"],
            "target_column": config["target_column"],
            "success": success,
            "model_path": f"models/trained/{config['name']}.pt" if success else None
        }
        
        # Test the model if training succeeded
        if success:
            model_path = f"models/trained/{config['name']}.pt"
            test_success = test_trained_model(model_path, config['name'])
            result["test_success"] = test_success
        
        results.append(result)
    
    # Create training summary
    summary = create_training_summary(results)
    
    # Print final results
    print("\n" + "=" * 70)
    print("[COMPLETE] Training Pipeline Complete!")
    print("=" * 70)
    
    print(f"\n[RESULTS] Results Summary:")
    print(f"   Total models: {summary['total_models']}")
    print(f"   [SUCCESS] Successful: {summary['successful_models']}")
    print(f"   [FAILED] Failed: {summary['failed_models']}")
    
    print(f"\n[MODELS] Trained Models:")
    for result in results:
        status = "[OK]" if result['success'] else "[FAIL]"
        test_status = ""
        if result.get('test_success') is not None:
            test_status = " ([TESTED])" if result['test_success'] else " ([TEST FAILED])"
        
        print(f"   {status} {result['model_name']}{test_status}")
        if result['success'] and result.get('model_path'):
            print(f"      [PATH] {result['model_path']}")
    
    print(f"\n[NEXT] Next steps:")
    if summary['successful_models'] > 0:
        print("   1. Test models: python examples/chemberta_example.py")
        print("   2. Validate models: python validate_codebase.py") 
        print("   3. Use trained models in your applications!")
    else:
        print("   1. Check dataset availability: python download_datasets.py")
        print("   2. Debug training issues in logs")
        print("   3. Try smaller datasets or fewer epochs")
    
    print(f"\n[READY] Your ChemBERTa models are ready for molecular property prediction!")

if __name__ == "__main__":
    main()