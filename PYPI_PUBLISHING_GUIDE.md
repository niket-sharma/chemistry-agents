# PyPI Publishing Guide for Chemistry Agents

This guide provides complete step-by-step instructions for publishing the `chemistry-agents` package to PyPI.

## 📋 Prerequisites Checklist

- [x] Package built successfully (`dist/` folder contains `.whl` and `.tar.gz`)
- [x] All tests passing
- [x] Package checks passed (`twine check dist/*`)
- [ ] PyPI accounts created
- [ ] API tokens obtained
- [ ] `.pypirc` configured

## 🔑 Step 1: Create PyPI Accounts

### 1.1 Test PyPI Account (Required First)
1. Go to https://test.pypi.org/account/register/
2. Fill in your details:
   - **Username**: Choose a unique username
   - **Email**: Use a valid email address
   - **Password**: Strong password
3. Verify your email address
4. **Important**: Test PyPI is separate from production PyPI!

### 1.2 Production PyPI Account
1. Go to https://pypi.org/account/register/
2. Fill in the same details (can use same username if available)
3. Verify your email address

## 🎫 Step 2: Get API Tokens

### 2.1 Test PyPI Token
1. Login to https://test.pypi.org/
2. Go to **Account Settings** → **API Tokens**
3. Click **"Add API Token"**
4. **Token name**: `chemistry-agents-testpypi`
5. **Scope**: Select "Entire account" (for first upload)
6. **Copy the token** - you won't see it again!
   - Format: `pypi-AgEIcHlwaS5vcmcC...` (starts with `pypi-`)

### 2.2 Production PyPI Token
1. Login to https://pypi.org/
2. Go to **Account Settings** → **API Tokens**
3. Click **"Add API Token"**
4. **Token name**: `chemistry-agents-pypi`
5. **Scope**: Select "Entire account"
6. **Copy the token** - save it securely!

## ⚙️ Step 3: Configure Authentication

### Option A: Using .pypirc (Recommended)

Create/edit `~/.pypirc` (Windows: `C:\Users\YourName\.pypirc`):

```ini
[distutils]
index-servers =
    pypi
    testpypi

[testpypi]
repository = https://test.pypi.org/legacy/
username = __token__
password = pypi-YOUR-TESTPYPI-TOKEN-HERE

[pypi]
username = __token__
password = pypi-YOUR-PRODUCTION-TOKEN-HERE
```

### Option B: Environment Variables

```bash
# Windows Command Prompt
set TWINE_USERNAME=__token__
set TWINE_PASSWORD=pypi-YOUR-TOKEN-HERE

# Windows PowerShell
$env:TWINE_USERNAME="__token__"
$env:TWINE_PASSWORD="pypi-YOUR-TOKEN-HERE"

# Linux/Mac
export TWINE_USERNAME=__token__
export TWINE_PASSWORD=pypi-YOUR-TOKEN-HERE
```

## 🧪 Step 4: Test Upload to Test PyPI

**Always test on Test PyPI first!** This prevents mistakes on production.

### 4.1 Upload to Test PyPI

```bash
# From your chemistry-agents directory
cd "C:\Users\Niket Sharma\llmapp\chemistry-agents"

# Upload to Test PyPI
twine upload --repository testpypi dist/*
```

**Expected Output:**
```
Uploading distributions to https://test.pypi.org/legacy/
Enter your username: __token__
Enter your password: [your token]
Uploading chemistry_agents-0.1.0-py3-none-any.whl
Uploading chemistry_agents-0.1.0.tar.gz
View at: https://test.pypi.org/project/chemistry-agents/
```

### 4.2 Check Your Package on Test PyPI
- Visit: https://test.pypi.org/project/chemistry-agents/
- Verify all information looks correct
- Check that README displays properly

## ✅ Step 5: Test Installation from Test PyPI

### 5.1 Create a Clean Test Environment

```bash
# Create new virtual environment for testing
python -m venv test-chemistry-agents
cd test-chemistry-agents

# Windows
Scripts\activate

# Linux/Mac  
source bin/activate
```

### 5.2 Install from Test PyPI

```bash
# Install from Test PyPI (dependencies from regular PyPI)
pip install -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ chemistry-agents
```

**Note**: The `--extra-index-url` allows dependencies to be installed from regular PyPI while getting your package from Test PyPI.

### 5.3 Test Your Package

```bash
# Test basic import
python -c "import chemistry_agents; print('Success!')"

# Test unit operations agent
python -c "
from chemistry_agents import UnitOperationsAgent, UnitOperationConfig
agent = UnitOperationsAgent()
agent.load_model()
print('Unit operations agent working!')
"
```

If everything works, you're ready for production!

## 🚀 Step 6: Upload to Production PyPI

**⚠️ Warning**: Once uploaded to PyPI, you **cannot** delete or modify a release. Make sure everything is correct!

### 6.1 Final Pre-Upload Checklist

- [ ] Package tested successfully on Test PyPI
- [ ] All documentation is correct
- [ ] Version number is correct in `setup.py` and `pyproject.toml`
- [ ] README displays correctly on Test PyPI
- [ ] All functionality tested

### 6.2 Upload to Production PyPI

```bash
# Upload to production PyPI
twine upload dist/*
```

**Expected Output:**
```
Uploading distributions to https://upload.pypi.org/legacy/
Enter your username: __token__
Enter your password: [your production token]
Uploading chemistry_agents-0.1.0-py3-none-any.whl
Uploading chemistry_agents-0.1.0.tar.gz
View at: https://pypi.org/project/chemistry-agents/
```

## 🎉 Step 7: Verify Production Installation

### 7.1 Install from PyPI

```bash
# Create fresh environment
python -m venv test-production
cd test-production
Scripts\activate  # Windows

# Install from production PyPI
pip install chemistry-agents
```

### 7.2 Test Installation

```bash
# Basic test
python -c "import chemistry_agents; print(f'Version: {chemistry_agents.__version__}')"

# Full functionality test
python -c "
from chemistry_agents import PropertyPredictionAgent, UnitOperationsAgent
print('✅ All agents imported successfully!')
print('🎉 chemistry-agents is live on PyPI!')
"
```

## 📚 Step 8: Post-Publication Tasks

### 8.1 Update GitHub Repository

Add PyPI badge to your README:

```markdown
[![PyPI version](https://badge.fury.io/py/chemistry-agents.svg)](https://badge.fury.io/py/chemistry-agents)
[![PyPI downloads](https://img.shields.io/pypi/dm/chemistry-agents.svg)](https://pypi.org/project/chemistry-agents/)
```

### 8.2 Create GitHub Release

1. Go to https://github.com/niket-sharma/CHEMISTRY-AGENTS/releases
2. Click **"Create a new release"**
3. **Tag**: `v0.1.0`
4. **Title**: `Chemistry Agents v0.1.0 - Initial Release`
5. **Description**: Copy from your commit message
6. Attach the `.whl` and `.tar.gz` files
7. Publish release

### 8.3 Update Documentation

Update your README with installation instructions:

```markdown
## Installation

```bash
pip install chemistry-agents
```

## Quick Start

```python
from chemistry_agents import PropertyPredictionAgent

agent = PropertyPredictionAgent()
agent.load_model()
results = agent.predict_batch(["CCO", "CC(=O)O"])
```
```

## 🔄 Future Updates

### Updating Your Package

1. **Update version** in `setup.py`, `pyproject.toml`, and `__init__.py`
2. **Test changes** thoroughly
3. **Rebuild package**: `python -m build`
4. **Test on Test PyPI first**
5. **Upload to production PyPI**

### Version Numbering

- **Patch** (0.1.0 → 0.1.1): Bug fixes
- **Minor** (0.1.0 → 0.2.0): New features
- **Major** (0.1.0 → 1.0.0): Breaking changes

## ⚡ Quick Command Reference

```bash
# Check package
twine check dist/*

# Upload to Test PyPI
twine upload --repository testpypi dist/*

# Upload to PyPI
twine upload dist/*

# Install from Test PyPI
pip install -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ chemistry-agents

# Install from PyPI
pip install chemistry-agents
```

## 🆘 Troubleshooting

### Common Upload Issues

**Error: "Repository not found"**
- Check your `.pypirc` configuration
- Verify API tokens are correct

**Error: "File already exists"**
- You cannot re-upload the same version
- Increment version number and rebuild

**Error: "Invalid authentication"**
- Check API token format (starts with `pypi-`)
- Verify token has correct permissions

**Error: "Package name already taken"**
- Choose a different package name
- Update `name` in `setup.py` and `pyproject.toml`

### Package Installation Issues

**Error: "No module named 'chemistry_agents'"**
- Package may not be installed correctly
- Try reinstalling: `pip uninstall chemistry-agents && pip install chemistry-agents`

**Dependency Issues:**
- Use `pip install chemistry-agents[all]` for full installation
- Check Python version compatibility (>=3.8 required)

## 🎯 Success Metrics

After publishing, you can track:
- **Downloads**: https://pypistats.org/packages/chemistry-agents
- **GitHub Stars**: Repository popularity
- **Issues/Discussions**: Community engagement
- **Usage**: Featured in other projects

---

## 🎉 Congratulations!

Once published, your `chemistry-agents` package will be:

✅ **Publicly available**: `pip install chemistry-agents`  
✅ **Discoverable**: Listed on PyPI search  
✅ **Professional**: Complete package with documentation  
✅ **Useful**: Molecular property prediction + chemical engineering  

**Your contribution to the Python/Chemistry community is now live!** 🧪⚗️🚀

<function_calls>
<invoke name="TodoWrite">
<parameter name="todos">[{"content": "Create PyPI account and get API tokens", "status": "completed", "priority": "high", "id": "1"}, {"content": "Install and configure twine for PyPI upload", "status": "completed", "priority": "high", "id": "2"}, {"content": "Test upload to Test PyPI first", "status": "in_progress", "priority": "high", "id": "3"}, {"content": "Verify package installation from Test PyPI", "status": "pending", "priority": "medium", "id": "4"}, {"content": "Upload to production PyPI", "status": "pending", "priority": "high", "id": "5"}, {"content": "Verify final package installation", "status": "pending", "priority": "medium", "id": "6"}, {"content": "Create PyPI project documentation", "status": "pending", "priority": "low", "id": "7"}]