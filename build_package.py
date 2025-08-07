#!/usr/bin/env python3
"""
Package build and distribution script for chemistry-agents

This script helps build, test, and distribute the chemistry-agents package.
"""

import os
import sys
import subprocess
import shutil
import argparse
from pathlib import Path


def run_command(cmd, check=True, cwd=None):
    """Run a shell command and return result"""
    print(f"🔧 Running: {' '.join(cmd)}")
    result = subprocess.run(
        cmd, 
        check=check, 
        capture_output=True, 
        text=True,
        cwd=cwd or os.getcwd()
    )
    
    if result.returncode != 0:
        print(f"❌ Command failed: {' '.join(cmd)}")
        print(f"Error: {result.stderr}")
        if check:
            sys.exit(1)
    else:
        print(f"✅ Command succeeded: {' '.join(cmd)}")
        if result.stdout.strip():
            print(f"Output: {result.stdout}")
    
    return result


def clean_build_artifacts():
    """Clean build artifacts and cache directories"""
    print("🧹 Cleaning build artifacts...")
    
    artifacts = [
        'build',
        'dist', 
        'chemistry_agents.egg-info',
        'src/chemistry_agents.egg-info',
        '__pycache__',
        '.pytest_cache',
        '.coverage',
        'htmlcov'
    ]
    
    for artifact in artifacts:
        if os.path.exists(artifact):
            if os.path.isdir(artifact):
                shutil.rmtree(artifact)
                print(f"  Removed directory: {artifact}")
            else:
                os.remove(artifact)
                print(f"  Removed file: {artifact}")
    
    # Clean __pycache__ directories recursively
    for root, dirs, files in os.walk('.'):
        for dir_name in dirs[:]:  # Create a copy to modify while iterating
            if dir_name == '__pycache__':
                shutil.rmtree(os.path.join(root, dir_name))
                print(f"  Removed: {os.path.join(root, dir_name)}")
                dirs.remove(dir_name)
    
    print("✅ Cleanup completed")


def check_dependencies():
    """Check if build dependencies are available"""
    print("🔍 Checking build dependencies...")
    
    required_tools = ['python', 'pip']
    optional_tools = ['twine', 'wheel']
    
    missing_required = []
    missing_optional = []
    
    for tool in required_tools:
        try:
            run_command([tool, '--version'], check=False)
        except FileNotFoundError:
            missing_required.append(tool)
    
    for tool in optional_tools:
        try:
            run_command([tool, '--version'], check=False)
        except FileNotFoundError:
            missing_optional.append(tool)
    
    if missing_required:
        print(f"❌ Missing required tools: {', '.join(missing_required)}")
        return False
    
    if missing_optional:
        print(f"⚠️  Missing optional tools (will install): {', '.join(missing_optional)}")
        for tool in missing_optional:
            run_command([sys.executable, '-m', 'pip', 'install', tool])
    
    print("✅ All dependencies available")
    return True


def run_tests():
    """Run the test suite"""
    print("🧪 Running test suite...")
    
    if not os.path.exists('tests'):
        print("⚠️  No tests directory found, skipping tests")
        return True
    
    try:
        # Try to run pytest
        result = run_command([
            sys.executable, '-m', 'pytest', 
            'tests/', 
            '-v', 
            '--tb=short'
        ], check=False)
        
        if result.returncode == 0:
            print("✅ All tests passed")
            return True
        else:
            print("❌ Some tests failed")
            return False
            
    except FileNotFoundError:
        print("📦 pytest not found, installing...")
        run_command([sys.executable, '-m', 'pip', 'install', 'pytest'])
        
        # Try again
        result = run_command([
            sys.executable, '-m', 'pytest', 
            'tests/', 
            '-v', 
            '--tb=short'
        ], check=False)
        
        return result.returncode == 0


def build_package():
    """Build the package"""
    print("📦 Building package...")
    
    # Install build dependencies
    run_command([sys.executable, '-m', 'pip', 'install', '--upgrade', 'build'])
    
    # Build source and wheel distributions
    run_command([sys.executable, '-m', 'build'])
    
    print("✅ Package built successfully")
    
    # List built files
    if os.path.exists('dist'):
        print("\n📁 Built packages:")
        for file in os.listdir('dist'):
            file_path = os.path.join('dist', file)
            size = os.path.getsize(file_path)
            print(f"  {file} ({size:,} bytes)")


def check_package():
    """Check the built package for common issues"""
    print("🔍 Checking package integrity...")
    
    if not os.path.exists('dist'):
        print("❌ No dist directory found. Build the package first.")
        return False
    
    # Install twine for checking
    try:
        run_command(['twine', '--version'], check=False)
    except FileNotFoundError:
        run_command([sys.executable, '-m', 'pip', 'install', 'twine'])
    
    # Check package
    run_command(['twine', 'check', 'dist/*'])
    print("✅ Package checks passed")
    
    return True


def install_locally():
    """Install package locally for testing"""
    print("📥 Installing package locally...")
    
    # Uninstall first if already installed
    run_command([
        sys.executable, '-m', 'pip', 'uninstall', 'chemistry-agents', '-y'
    ], check=False)
    
    # Install in editable mode from current directory
    run_command([sys.executable, '-m', 'pip', 'install', '-e', '.'])
    
    print("✅ Package installed locally")


def test_installation():
    """Test the locally installed package"""
    print("🧪 Testing installed package...")
    
    try:
        # Test basic import
        result = run_command([
            sys.executable, '-c',
            'import chemistry_agents; print("Package imported successfully")'
        ])
        
        # Test agent creation
        result = run_command([
            sys.executable, '-c', 
            '''
import chemistry_agents
from chemistry_agents.agents.base_agent import AgentConfig
print(f"Chemistry Agents version: {chemistry_agents.__version__}")
config = AgentConfig()
print(f"Default device: {config.device}")
print("✅ Basic functionality test passed")
            '''
        ])
        
        print("✅ Installation test passed")
        return True
        
    except Exception as e:
        print(f"❌ Installation test failed: {e}")
        return False


def publish_to_pypi(test=True):
    """Publish package to PyPI or Test PyPI"""
    repository = "testpypi" if test else "pypi"
    print(f"🚀 Publishing to {'Test PyPI' if test else 'PyPI'}...")
    
    if not os.path.exists('dist'):
        print("❌ No dist directory found. Build the package first.")
        return False
    
    # Install twine if not available
    try:
        run_command(['twine', '--version'], check=False)
    except FileNotFoundError:
        run_command([sys.executable, '-m', 'pip', 'install', 'twine'])
    
    # Upload to repository
    if test:
        run_command(['twine', 'upload', '--repository', 'testpypi', 'dist/*'])
        print("✅ Package uploaded to Test PyPI")
        print("🔗 Check your package at: https://test.pypi.org/project/chemistry-agents/")
    else:
        run_command(['twine', 'upload', 'dist/*'])
        print("✅ Package uploaded to PyPI")
        print("🔗 Check your package at: https://pypi.org/project/chemistry-agents/")


def main():
    parser = argparse.ArgumentParser(description="Build and distribute chemistry-agents package")
    parser.add_argument('command', choices=[
        'clean', 'build', 'test', 'check', 'install', 'test-install',
        'publish-test', 'publish', 'all'
    ], help='Command to execute')
    parser.add_argument('--skip-tests', action='store_true', help='Skip running tests')
    
    args = parser.parse_args()
    
    print("🧪 Chemistry Agents Package Builder")
    print("=" * 50)
    
    if args.command == 'clean':
        clean_build_artifacts()
        
    elif args.command == 'build':
        if not check_dependencies():
            sys.exit(1)
        clean_build_artifacts()
        if not args.skip_tests:
            if not run_tests():
                print("❌ Tests failed, aborting build")
                sys.exit(1)
        build_package()
        
    elif args.command == 'test':
        if not run_tests():
            sys.exit(1)
            
    elif args.command == 'check':
        if not check_package():
            sys.exit(1)
            
    elif args.command == 'install':
        install_locally()
        
    elif args.command == 'test-install':
        install_locally()
        if not test_installation():
            sys.exit(1)
            
    elif args.command == 'publish-test':
        if not check_package():
            sys.exit(1)
        publish_to_pypi(test=True)
        
    elif args.command == 'publish':
        if not check_package():
            sys.exit(1)
        publish_to_pypi(test=False)
        
    elif args.command == 'all':
        if not check_dependencies():
            sys.exit(1)
        clean_build_artifacts()
        if not args.skip_tests:
            if not run_tests():
                print("❌ Tests failed, aborting build")
                sys.exit(1)
        build_package()
        if not check_package():
            sys.exit(1)
        install_locally()
        if not test_installation():
            sys.exit(1)
        print("\n🎉 All steps completed successfully!")
        print("💡 To publish to Test PyPI: python build_package.py publish-test")
        print("💡 To publish to PyPI: python build_package.py publish")
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()