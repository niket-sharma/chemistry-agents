#!/usr/bin/env python3
"""
Environment setup script for Chemistry Agents
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"🔄 {description}")
    print(f"   Running: {command}")
    
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"   ✅ Success!")
        if result.stdout:
            print(f"   Output: {result.stdout.strip()}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"   ❌ Error: {e}")
        if e.stderr:
            print(f"   Error details: {e.stderr.strip()}")
        return False

def check_python_version():
    """Check if Python version is compatible"""
    print("🐍 Checking Python version...")
    
    version = sys.version_info
    if version.major >= 3 and version.minor >= 8:
        print(f"   ✅ Python {version.major}.{version.minor}.{version.micro} is compatible")
        return True
    else:
        print(f"   ❌ Python {version.major}.{version.minor}.{version.micro} is too old")
        print("   Please install Python 3.8 or newer")
        return False

def setup_virtual_environment():
    """Set up virtual environment"""
    print("\n📦 Setting up virtual environment...")
    
    # Create virtual environment
    if not run_command("python -m venv chemistry_agents_env", "Creating virtual environment"):
        # Try with python3
        if not run_command("python3 -m venv chemistry_agents_env", "Creating virtual environment (python3)"):
            return False
    
    # Check what activation script to use
    if os.name == 'nt':  # Windows
        activate_script = "chemistry_agents_env\\Scripts\\activate"
        pip_command = "chemistry_agents_env\\Scripts\\pip"
    else:  # Linux/Mac
        activate_script = "source chemistry_agents_env/bin/activate"
        pip_command = "chemistry_agents_env/bin/pip"
    
    print(f"\n✅ Virtual environment created!")
    print(f"   To activate: {activate_script}")
    print(f"   Pip location: {pip_command}")
    
    return True, pip_command

def install_basic_dependencies(pip_command):
    """Install basic dependencies"""
    print("\n📚 Installing basic dependencies...")
    
    # Core dependencies first
    core_deps = [
        "torch>=1.9.0",
        "numpy>=1.21.0", 
        "pandas>=1.3.0",
        "scikit-learn>=1.0.0",
        "matplotlib>=3.5.0",
        "seaborn>=0.11.0",
        "tqdm>=4.64.0"
    ]
    
    for dep in core_deps:
        if not run_command(f"{pip_command} install {dep}", f"Installing {dep}"):
            print(f"   ⚠️ Failed to install {dep}, continuing...")
    
    return True

def install_chemistry_dependencies(pip_command):
    """Install chemistry-specific dependencies"""
    print("\n🧪 Installing chemistry dependencies...")
    
    # RDKit (chemistry toolkit)
    if not run_command(f"{pip_command} install rdkit-pypi", "Installing RDKit"):
        print("   ⚠️ RDKit installation failed - some features may not work")
    
    # Transformers
    if not run_command(f"{pip_command} install transformers>=4.20.0", "Installing Transformers"):
        print("   ⚠️ Transformers installation failed")
    
    # Optional but useful
    optional_deps = [
        "tokenizers>=0.13.0",
        "datasets>=2.0.0"
    ]
    
    for dep in optional_deps:
        run_command(f"{pip_command} install {dep}", f"Installing {dep}")

def install_project_in_dev_mode(pip_command):
    """Install the project in development mode"""
    print("\n🔧 Installing Chemistry Agents in development mode...")
    
    if run_command(f"{pip_command} install -e .", "Installing project in dev mode"):
        print("   ✅ Project installed successfully!")
        return True
    else:
        print("   ⚠️ Development installation failed, but you can still run examples")
        return False

def create_test_script():
    """Create a simple test script"""
    test_script = """#!/usr/bin/env python3
'''
Simple test script to verify Chemistry Agents installation
'''

def test_basic_imports():
    print("🧪 Testing basic imports...")
    
    try:
        import numpy as np
        print("   ✅ NumPy imported successfully")
    except ImportError as e:
        print(f"   ❌ NumPy import failed: {e}")
        return False
    
    try:
        import pandas as pd
        print("   ✅ Pandas imported successfully")
    except ImportError as e:
        print(f"   ❌ Pandas import failed: {e}")
        return False
    
    try:
        import torch
        print(f"   ✅ PyTorch imported successfully (version: {torch.__version__})")
    except ImportError as e:
        print(f"   ❌ PyTorch import failed: {e}")
        return False
    
    return True

def test_chemistry_imports():
    print("\\n🔬 Testing chemistry libraries...")
    
    try:
        from rdkit import Chem
        print("   ✅ RDKit imported successfully")
        
        # Test basic RDKit functionality
        mol = Chem.MolFromSmiles("CCO")
        if mol is not None:
            print("   ✅ RDKit SMILES parsing works")
        else:
            print("   ⚠️ RDKit SMILES parsing failed")
    except ImportError as e:
        print(f"   ❌ RDKit import failed: {e}")
        return False
    
    try:
        from transformers import AutoTokenizer
        print("   ✅ Transformers imported successfully")
    except ImportError as e:
        print(f"   ❌ Transformers import failed: {e}")
        return False
    
    return True

def test_chemistry_agents():
    print("\\n🤖 Testing Chemistry Agents...")
    
    try:
        import sys
        import os
        
        # Add src to path
        src_path = os.path.join(os.path.dirname(__file__), 'src')
        if os.path.exists(src_path):
            sys.path.insert(0, src_path)
        
        from chemistry_agents import PropertyPredictionAgent
        print("   ✅ PropertyPredictionAgent imported successfully")
        
        # Test agent creation
        agent = PropertyPredictionAgent(
            property_name="logP",
            model_type="neural_network"  # Start with simpler model
        )
        print("   ✅ Agent created successfully")
        
        # Test mock prediction
        agent.is_loaded = True  # Mock loaded state
        result = agent.predict_single("CCO")
        print(f"   ✅ Mock prediction successful: {result.prediction:.3f}")
        
        return True
        
    except ImportError as e:
        print(f"   ❌ Chemistry Agents import failed: {e}")
        return False
    except Exception as e:
        print(f"   ❌ Chemistry Agents test failed: {e}")
        return False

def main():
    print("🧪 Chemistry Agents Test Suite")
    print("=" * 50)
    
    # Run tests
    basic_ok = test_basic_imports()
    chemistry_ok = test_chemistry_imports()
    agents_ok = test_chemistry_agents()
    
    print("\\n" + "=" * 50)
    print("📊 Test Results Summary:")
    print(f"   Basic imports: {'✅ PASS' if basic_ok else '❌ FAIL'}")
    print(f"   Chemistry libs: {'✅ PASS' if chemistry_ok else '❌ FAIL'}")
    print(f"   Chemistry Agents: {'✅ PASS' if agents_ok else '❌ FAIL'}")
    
    if basic_ok and chemistry_ok and agents_ok:
        print("\\n🎉 All tests passed! Chemistry Agents is ready to use.")
        print("\\nNext steps:")
        print("   • Run: python examples/basic_usage.py")
        print("   • Run: python examples/quick_start_hf.py")
        print("   • Check out the README.md for more examples")
    else:
        print("\\n⚠️ Some tests failed. Check the installation steps.")
        
    return basic_ok and chemistry_ok and agents_ok

if __name__ == "__main__":
    main()
"""
    
    with open("test_installation.py", "w") as f:
        f.write(test_script)
    
    print("   ✅ Test script created: test_installation.py")

def main():
    print("🧪 Chemistry Agents Environment Setup")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        return False
    
    # Setup virtual environment
    success, pip_command = setup_virtual_environment()
    if not success:
        return False
    
    # Install dependencies
    install_basic_dependencies(pip_command)
    install_chemistry_dependencies(pip_command)
    install_project_in_dev_mode(pip_command)
    
    # Create test script
    create_test_script()
    
    print("\n🎉 Setup completed!")
    print("\n📋 Next Steps:")
    
    if os.name == 'nt':  # Windows
        print("   1. Activate environment: chemistry_agents_env\\Scripts\\activate")
    else:  # Linux/Mac
        print("   1. Activate environment: source chemistry_agents_env/bin/activate")
    
    print("   2. Test installation: python test_installation.py")
    print("   3. Run examples: python examples/basic_usage.py")
    print("   4. Check README.md for more information")
    
    return True

if __name__ == "__main__":
    main()