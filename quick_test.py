#!/usr/bin/env python3
"""
Quick test script to verify Chemistry Agents is working
Run this after installation to check everything is set up correctly
"""

import sys
import os

def test_1_basic_imports():
    """Test 1: Basic Python libraries"""
    print("🔬 Test 1: Basic imports...")
    
    try:
        import numpy as np
        import pandas as pd
        print("   ✅ NumPy and Pandas work")
    except ImportError as e:
        print(f"   ❌ Basic imports failed: {e}")
        return False
    
    try:
        import torch
        print(f"   ✅ PyTorch works (version {torch.__version__})")
    except ImportError as e:
        print(f"   ❌ PyTorch failed: {e}")
        return False
    
    return True

def test_2_chemistry_libs():
    """Test 2: Chemistry-specific libraries"""
    print("\n🧪 Test 2: Chemistry libraries...")
    
    try:
        from rdkit import Chem
        
        # Test basic SMILES parsing
        mol = Chem.MolFromSmiles("CCO")
        if mol is not None:
            print("   ✅ RDKit works - can parse SMILES")
        else:
            print("   ⚠️ RDKit imported but SMILES parsing failed")
            return False
            
    except ImportError:
        print("   ❌ RDKit not available (some features will be limited)")
        print("   💡 Install with: pip install rdkit-pypi")
        return False
    
    try:
        from transformers import AutoTokenizer
        print("   ✅ Transformers library works")
    except ImportError:
        print("   ❌ Transformers not available")
        print("   💡 Install with: pip install transformers")
        return False
    
    return True

def test_3_chemistry_agents():
    """Test 3: Chemistry Agents framework"""
    print("\n🤖 Test 3: Chemistry Agents...")
    
    # Add src directory to path for development
    current_dir = os.path.dirname(os.path.abspath(__file__))
    src_path = os.path.join(current_dir, 'src')
    
    if os.path.exists(src_path):
        sys.path.insert(0, src_path)
        print(f"   📁 Added {src_path} to Python path")
    
    try:
        # Test importing the main classes
        from chemistry_agents import PropertyPredictionAgent
        print("   ✅ PropertyPredictionAgent imported")
        
        from chemistry_agents import SolubilityAgent
        print("   ✅ SolubilityAgent imported")
        
        # Test creating an agent
        agent = PropertyPredictionAgent(
            property_name="test_property",
            model_type="neural_network"
        )
        print("   ✅ Agent created successfully")
        
        return True
        
    except ImportError as e:
        print(f"   ❌ Chemistry Agents import failed: {e}")
        print("   💡 Make sure you're in the chemistry-agents directory")
        return False
    except Exception as e:
        print(f"   ❌ Chemistry Agents test failed: {e}")
        return False

def test_4_mock_prediction():
    """Test 4: Mock prediction to test workflow"""
    print("\n🎯 Test 4: Mock prediction...")
    
    try:
        # Import again (may be needed if previous test failed)
        current_dir = os.path.dirname(os.path.abspath(__file__))
        src_path = os.path.join(current_dir, 'src')
        if os.path.exists(src_path):
            sys.path.insert(0, src_path)
        
        from chemistry_agents import PropertyPredictionAgent
        
        # Create agent
        agent = PropertyPredictionAgent(
            property_name="logP",
            model_type="neural_network"
        )
        
        # Mock the model as loaded (since we don't have a trained model yet)
        agent.is_loaded = True
        
        # Test prediction
        result = agent.predict_single("CCO")  # Ethanol
        
        print(f"   ✅ Mock prediction successful!")
        print(f"   📊 SMILES: {result.smiles}")
        print(f"   📊 Prediction: {result.prediction:.3f}")
        print(f"   📊 Confidence: {result.confidence:.3f}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Mock prediction failed: {e}")
        return False

def test_5_batch_prediction():
    """Test 5: Batch prediction"""
    print("\n📦 Test 5: Batch prediction...")
    
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        src_path = os.path.join(current_dir, 'src')
        if os.path.exists(src_path):
            sys.path.insert(0, src_path)
        
        from chemistry_agents import PropertyPredictionAgent
        
        agent = PropertyPredictionAgent(property_name="logP")
        agent.is_loaded = True  # Mock loaded
        
        # Test molecules
        molecules = ["CCO", "CCC", "CCCC"]  # Ethanol, Propane, Butane
        
        # Batch prediction
        results = agent.predict_batch(molecules)
        
        print(f"   ✅ Batch prediction successful!")
        print(f"   📊 Processed {len(results)} molecules")
        
        for mol, result in zip(molecules, results):
            print(f"      {mol}: {result.prediction:.3f}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Batch prediction failed: {e}")
        return False

def run_all_tests():
    """Run all tests and show summary"""
    print("🧪 Chemistry Agents - Quick Test Suite")
    print("=" * 50)
    
    tests = [
        ("Basic imports", test_1_basic_imports),
        ("Chemistry libraries", test_2_chemistry_libs),
        ("Chemistry Agents", test_3_chemistry_agents),
        ("Mock prediction", test_4_mock_prediction),
        ("Batch prediction", test_5_batch_prediction)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"   💥 Test '{test_name}' crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Results Summary:")
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {test_name:20s}: {status}")
        if result:
            passed += 1
    
    print(f"\n📈 Overall: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("\n🎉 All tests passed! Chemistry Agents is ready to use!")
        print("\n🚀 Next steps:")
        print("   • Run: python examples/basic_usage.py")
        print("   • Run: python examples/quick_start_hf.py")
        print("   • Check out the documentation in README.md")
        
    elif passed >= 3:
        print("\n✅ Core functionality works! Some advanced features may be limited.")
        print("   You can start using Chemistry Agents with basic functionality.")
        
    else:
        print("\n⚠️ Multiple tests failed. Please check the installation:")
        print("   1. Make sure you're in the chemistry-agents directory")
        print("   2. Activate the virtual environment")
        print("   3. Install dependencies: pip install -r requirements.txt")
        print("   4. Check INSTALLATION.md for detailed setup instructions")
    
    return passed == len(results)

def main():
    """Main function"""
    # Check if we're in the right directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    expected_files = ['README.md', 'requirements.txt', 'src']
    
    missing_files = [f for f in expected_files if not os.path.exists(os.path.join(current_dir, f))]
    
    if missing_files:
        print("⚠️ Warning: Some expected files/directories not found:")
        for f in missing_files:
            print(f"   Missing: {f}")
        print("\n💡 Make sure you're running this from the chemistry-agents directory")
        print("   Current directory:", current_dir)
    
    # Run tests
    success = run_all_tests()
    
    return 0 if success else 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)