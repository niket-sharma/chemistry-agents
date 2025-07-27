"""
Test runner script to verify all functionality works correctly
"""

import pytest
import sys
import os
import subprocess
from pathlib import Path


def run_specific_test_suite(test_file: str, verbose: bool = True) -> bool:
    """Run a specific test suite and return success status"""
    cmd = [sys.executable, "-m", "pytest", test_file, "-v" if verbose else "-q"]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        print(f"\n{'='*60}")
        print(f"Running: {test_file}")
        print(f"{'='*60}")
        
        if result.returncode == 0:
            print(f"✅ {test_file} - All tests PASSED")
            if verbose:
                print(result.stdout)
        else:
            print(f"❌ {test_file} - Some tests FAILED")
            print("STDOUT:", result.stdout)
            print("STDERR:", result.stderr)
        
        return result.returncode == 0
        
    except subprocess.TimeoutExpired:
        print(f"⏰ {test_file} - Tests TIMED OUT")
        return False
    except Exception as e:
        print(f"💥 {test_file} - Error running tests: {e}")
        return False


def check_dependencies() -> bool:
    """Check if required dependencies are installed"""
    required_packages = [
        'pytest',
        'torch',
        'rdkit',
        'transformers',
        'sklearn',
        'pandas',
        'numpy'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'rdkit':
                import rdkit
            elif package == 'sklearn':
                import sklearn
            else:
                __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("❌ Missing required packages:")
        for pkg in missing_packages:
            print(f"   - {pkg}")
        print("\nInstall missing packages with:")
        print(f"   pip install {' '.join(missing_packages)}")
        return False
    
    print("✅ All required dependencies are installed")
    return True


def run_all_tests() -> bool:
    """Run all test suites"""
    test_files = [
        "tests/test_cpu_optimization.py",
        "tests/test_api_integration.py", 
        "tests/test_agent_integration.py",
        "tests/test_cloud_training.py"
    ]
    
    print("🧪 Chemistry Agents - Test Suite Runner")
    print("="*60)
    
    # Check dependencies first
    if not check_dependencies():
        return False
    
    # Run each test suite
    results = {}
    all_passed = True
    
    for test_file in test_files:
        if os.path.exists(test_file):
            success = run_specific_test_suite(test_file)
            results[test_file] = success
            if not success:
                all_passed = False
        else:
            print(f"⚠️  Test file not found: {test_file}")
            results[test_file] = False
            all_passed = False
    
    # Summary
    print(f"\n{'='*60}")
    print("📊 TEST SUMMARY")
    print(f"{'='*60}")
    
    for test_file, success in results.items():
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{status:<12} {test_file}")
    
    print(f"\n{'🎉 ALL TESTS PASSED!' if all_passed else '⚠️  SOME TESTS FAILED'}")
    
    if all_passed:
        print("\n🚀 Your Chemistry Agents setup is working correctly!")
        print("💡 You can now use the CPU-optimized features and API integration.")
    else:
        print("\n🔧 Some tests failed. Check the output above for details.")
        print("💡 You may need to install missing dependencies or fix configuration issues.")
    
    return all_passed


def run_quick_smoke_test() -> bool:
    """Run a quick smoke test to verify basic functionality"""
    print("🔥 Running Quick Smoke Test...")
    
    try:
        # Test basic imports
        print("  - Testing imports...")
        from chemistry_agents.agents.base_agent import AgentConfig, PredictionResult
        from chemistry_agents.agents.property_prediction_agent import PropertyPredictionAgent
        print("    ✅ Core imports successful")
        
        # Test CPU configuration
        print("  - Testing CPU configuration...")
        config = AgentConfig(device="cpu", batch_size=2, cpu_optimization=True)
        agent = PropertyPredictionAgent(config=config)
        print("    ✅ CPU configuration successful")
        
        # Test API integration imports
        print("  - Testing API integration...")
        try:
            from chemistry_agents.utils.api_integration import APIConfig, HuggingFaceInferenceAPI
            print("    ✅ API integration imports successful")
        except ImportError as e:
            print(f"    ⚠️  API integration not available: {e}")
        
        # Test SMILES validation
        print("  - Testing SMILES validation...")
        try:
            from rdkit import Chem
            mol = Chem.MolFromSmiles("CCO")
            if mol is not None:
                print("    ✅ RDKit SMILES validation successful")
            else:
                print("    ❌ RDKit SMILES validation failed")
                return False
        except ImportError:
            print("    ⚠️  RDKit not available for SMILES validation")
        
        print("🎉 Smoke test completed successfully!")
        return True
        
    except Exception as e:
        print(f"💥 Smoke test failed: {e}")
        return False


def main():
    """Main test runner"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Chemistry Agents Test Runner")
    parser.add_argument("--quick", action="store_true", help="Run quick smoke test only")
    parser.add_argument("--test", type=str, help="Run specific test file")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    if args.quick:
        success = run_quick_smoke_test()
    elif args.test:
        success = run_specific_test_suite(args.test, args.verbose)
    else:
        success = run_all_tests()
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()