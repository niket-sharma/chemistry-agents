#!/usr/bin/env python3
"""
Comprehensive validation script for chemistry-agents codebase
"""

import sys
import os
import traceback
from typing import List, Tuple

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_imports() -> Tuple[bool, str]:
    """Test all core imports"""
    try:
        from chemistry_agents import (
            PropertyPredictionAgent,
            SolubilityAgent,
            ToxicityAgent,
            DrugDiscoveryAgent,
            UnitOperationsAgent,
            MolecularFeatureExtractor,
            SMILESProcessor
        )
        return True, "✅ All core imports successful"
    except Exception as e:
        return False, f"❌ Import failed: {e}"

def test_feature_extraction() -> Tuple[bool, str]:
    """Test molecular feature extraction"""
    try:
        from chemistry_agents.utils import MolecularFeatureExtractor
        
        extractor = MolecularFeatureExtractor()
        
        # Test single molecule
        features = extractor.extract_features("CCO")  # Ethanol
        if features is None or len(features) == 0:
            return False, "❌ Feature extraction failed"
        
        # Test batch processing
        smiles_list = ["CCO", "c1ccccc1", "invalid_smiles", "CC(C)O"]
        batch_features, valid_mask = extractor.extract_batch_features(smiles_list)
        
        if len(batch_features) != 3:  # Should have 3 valid molecules
            return False, f"❌ Batch processing failed: expected 3, got {len(batch_features)}"
            
        return True, f"✅ Feature extraction works (extracted {len(features)} features per molecule)"
        
    except Exception as e:
        return False, f"❌ Feature extraction failed: {e}"

def test_smiles_processing() -> Tuple[bool, str]:
    """Test SMILES processing"""
    try:
        from chemistry_agents.utils import SMILESProcessor
        
        processor = SMILESProcessor()
        
        # Test batch processing
        smiles_list = ["CCO", "c1ccccc1", "CC(C)O"]
        processed, valid_mask = processor.process_smiles_batch(smiles_list)
        
        if not all(valid_mask):
            return False, "❌ SMILES processing failed for valid molecules"
            
        return True, f"✅ SMILES processing works (processed {len(processed)} molecules)"
        
    except Exception as e:
        return False, f"❌ SMILES processing failed: {e}"

def test_agents() -> Tuple[bool, str]:
    """Test agent initialization"""
    try:
        from chemistry_agents import (
            PropertyPredictionAgent,
            SolubilityAgent,
            ToxicityAgent,
            DrugDiscoveryAgent,
            UnitOperationsAgent
        )
        
        agents = []
        agent_names = []
        
        # Test each agent
        agents.append(PropertyPredictionAgent())
        agent_names.append("PropertyPredictionAgent")
        
        agents.append(SolubilityAgent())
        agent_names.append("SolubilityAgent")
        
        agents.append(ToxicityAgent())
        agent_names.append("ToxicityAgent")
        
        agents.append(DrugDiscoveryAgent())
        agent_names.append("DrugDiscoveryAgent")
        
        agents.append(UnitOperationsAgent())
        agent_names.append("UnitOperationsAgent")
        
        return True, f"✅ All agents initialized successfully: {', '.join(agent_names)}"
        
    except Exception as e:
        return False, f"❌ Agent initialization failed: {e}"

def test_data_processing() -> Tuple[bool, str]:
    """Test data processing utilities"""
    try:
        from chemistry_agents.utils import DataProcessor
        import pandas as pd
        
        # Create test data
        test_data = pd.DataFrame({
            'smiles': ['CCO', 'c1ccccc1', 'CC(C)O', 'invalid', 'CCCC'],
            'target': [1.0, 2.0, 1.5, 0.0, 1.8]
        })
        
        processor = DataProcessor()
        
        # Test data cleaning
        clean_data = processor.clean_dataset(test_data)
        
        if len(clean_data) < 3:  # Should remove invalid SMILES
            return False, f"❌ Data cleaning failed: only {len(clean_data)} valid molecules"
        
        # Test dataset splitting
        train, val, test = processor.split_dataset(clean_data, test_size=0.2, val_size=0.2)
        
        total_size = len(train) + len(val) + len(test)
        if total_size != len(clean_data):
            return False, f"❌ Dataset splitting failed: size mismatch"
        
        return True, f"✅ Data processing works (cleaned: {len(clean_data)}, split: {len(train)}/{len(val)}/{len(test)})"
        
    except Exception as e:
        return False, f"❌ Data processing failed: {e}"

def test_unit_operations() -> Tuple[bool, str]:
    """Test unit operations functionality"""
    try:
        from chemistry_agents import UnitOperationsAgent
        from chemistry_agents.agents.base_agent import AgentConfig
        
        config = AgentConfig()
        agent = UnitOperationsAgent(config)
        
        # Test that agent has expected methods
        if not hasattr(agent, 'get_supported_operations'):
            return False, "❌ Agent missing get_supported_operations method"
        
        # Test supported operations
        supported_ops = agent.get_supported_operations()
        if not supported_ops or not isinstance(supported_ops, list):
            return False, "❌ get_supported_operations failed"
        
        # Test operation parameters
        if 'distillation' in supported_ops:
            params = agent.get_operation_parameters('distillation')
            if not isinstance(params, dict):
                return False, "❌ get_operation_parameters failed"
        
        return True, f"✅ Unit operations work (supports {len(supported_ops)} operations: {', '.join(supported_ops[:3])})"
        
    except Exception as e:
        return False, f"❌ Unit operations failed: {e}"

def test_model_loading() -> Tuple[bool, str]:
    """Test model loading capabilities"""
    try:
        from chemistry_agents import SolubilityAgent
        
        agent = SolubilityAgent()
        
        # Test that agent initializes properly
        if not hasattr(agent, 'feature_extractor'):
            return False, "❌ Agent initialization failed - missing feature extractor"
        
        # Test that model is initially None (expected)
        if agent.model is not None:
            return False, "❌ Model should be None initially"
        
        # Test that is_loaded is initially False
        if agent.is_loaded:
            return False, "❌ is_loaded should be False initially"
        
        # Test error handling when model not loaded
        try:
            result = agent.predict_single("CCO")
            return False, "❌ Should raise RuntimeError when model not loaded"
        except RuntimeError as e:
            if "Model not loaded" not in str(e):
                return False, f"❌ Wrong error message: {e}"
        except Exception as e:
            return False, f"❌ Wrong exception type: {type(e).__name__}: {e}"
        
        return True, "✅ Model interface works (proper initialization and error handling)"
        
    except Exception as e:
        return False, f"❌ Model loading test failed: {e}"

def run_validation() -> None:
    """Run all validation tests"""
    print("🧪 Chemistry Agents Validation Suite")
    print("=" * 50)
    
    tests = [
        ("Core Imports", test_imports),
        ("Feature Extraction", test_feature_extraction),
        ("SMILES Processing", test_smiles_processing),
        ("Agent Initialization", test_agents),
        ("Data Processing", test_data_processing),
        ("Unit Operations", test_unit_operations),
        ("Model Interface", test_model_loading),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n🔍 Testing {test_name}...")
        try:
            success, message = test_func()
            print(f"   {message}")
            if success:
                passed += 1
        except Exception as e:
            print(f"   ❌ {test_name} crashed: {e}")
            traceback.print_exc()
    
    print("\n" + "=" * 50)
    print(f"📊 Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Codebase is working correctly.")
        print("\n📋 Next steps:")
        print("   1. Train a model: cd scripts && python train_model.py --data_path ../data/sample_solubility.csv --target_column logS")
        print("   2. Run full examples: python examples/basic_usage.py")
        print("   3. Try unit operations: python examples/unit_operations_example.py")
    else:
        print("⚠️  Some tests failed. Check the errors above.")
        sys.exit(1)

if __name__ == "__main__":
    run_validation()