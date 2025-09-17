#!/usr/bin/env python3
"""
Test the training pipeline with minimal epochs for validation
"""

import subprocess
import os
import sys

def test_training_pipeline():
    """Test the complete training pipeline"""
    print("🔬 Testing Training Pipeline")
    print("=" * 40)
    
    # 1. Create sample data (if not exists)
    if not os.path.exists('data/sample_solubility.csv'):
        print("📊 Creating sample dataset...")
        result = subprocess.run([sys.executable, 'create_sample_data.py'], 
                              capture_output=True, text=True)
        if result.returncode != 0:
            print(f"❌ Failed to create sample data: {result.stderr}")
            return False
        print("✅ Sample dataset created")
    
    # 2. Test neural network training (minimal epochs)
    print("\n🧠 Testing Neural Network Training...")
    cmd = [
        sys.executable, 'scripts/train_model.py',
        '--data_path', 'data/sample_solubility.csv',
        '--target_column', 'logS',
        '--model_type', 'neural_network',
        '--epochs', '5',
        '--batch_size', '16',
        '--output_dir', 'test_models',
        '--model_name', 'test_nn_model'
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        print("✅ Neural network training completed")
    else:
        print(f"❌ Neural network training failed: {result.stderr}")
        return False
    
    # 3. Test model loading and inference
    print("\n🔮 Testing Model Loading and Inference...")
    test_inference_code = '''
import sys
import os
sys.path.append("src")
from chemistry_agents import SolubilityAgent

try:
    agent = SolubilityAgent()
    # Try to load the trained model
    if os.path.exists("test_models/test_nn_model.pt"):
        agent.load_model("test_models/test_nn_model.pt")
        print("✅ Model loaded successfully")
    else:
        print("⚠️ No trained model found, using default behavior")
    
    # Test prediction
    result = agent.predict_single("CCO")
    print(f"✅ Prediction successful: {result.prediction:.3f}")
    
except Exception as e:
    print(f"❌ Inference failed: {e}")
    sys.exit(1)
'''
    
    result = subprocess.run([sys.executable, '-c', test_inference_code], 
                          capture_output=True, text=True)
    if result.returncode == 0:
        print(result.stdout)
    else:
        print(f"❌ Inference test failed: {result.stderr}")
        return False
    
    return True

if __name__ == "__main__":
    success = test_training_pipeline()
    if success:
        print("\n🎉 Training pipeline validation passed!")
    else:
        print("\n❌ Training pipeline validation failed!")
        sys.exit(1)