#!/usr/bin/env python3
"""
Setup script for Chemistry Agents package
"""

from setuptools import setup, find_packages
import os

# Read README for long description
def read_readme():
    readme_path = os.path.join(os.path.dirname(__file__), 'README.md')
    if os.path.exists(readme_path):
        with open(readme_path, 'r', encoding='utf-8') as f:
            return f.read()
    return ""

# Read requirements
def read_requirements():
    requirements_path = os.path.join(os.path.dirname(__file__), 'requirements.txt')
    if os.path.exists(requirements_path):
        with open(requirements_path, 'r', encoding='utf-8') as f:
            requirements = []
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    # Handle optional dependencies
                    if '# optional' in line.lower():
                        continue
                    requirements.append(line.split('#')[0].strip())
            return requirements
    return []

# Core requirements (essential for basic functionality)
core_requirements = [
    "torch>=1.9.0",
    "numpy>=1.21.0",
    "pandas>=1.3.0",
    "scikit-learn>=1.0.0",
    "rdkit-pypi>=2022.3.0",
    "transformers>=4.20.0",
    "tqdm>=4.64.0",
    "matplotlib>=3.5.0",
    "seaborn>=0.11.0"
]

# Optional dependencies
extras_require = {
    'gpu': [
        'cupy>=10.6.0',
        'torch-geometric>=2.0.0'
    ],
    'visualization': [
        'py3Dmol>=1.8.0',
        'nglview>=3.0.0',
        'plotly>=5.0.0',
        'ipywidgets>=7.6.0'
    ],
    'notebooks': [
        'jupyter>=1.0.0',
        'jupyterlab>=3.4.0',
        'ipython>=8.0.0'
    ],
    'dev': [
        'pytest>=7.0.0',
        'pytest-cov>=3.0.0',
        'black>=22.0.0',
        'flake8>=4.0.0',
        'isort>=5.10.0',
        'mypy>=0.961',
        'pre-commit>=2.19.0'
    ],
    'api': [
        'fastapi>=0.78.0',
        'uvicorn>=0.18.0',
        'streamlit>=1.11.0'
    ],
    'cloud': [
        'boto3>=1.24.0',
        'google-cloud-storage>=2.4.0',
        'azure-storage-blob>=12.12.0'
    ],
    'experiment_tracking': [
        'wandb>=0.12.0',
        'mlflow>=1.27.0',
        'tensorboard>=2.9.0'
    ]
}

# All optional dependencies
extras_require['all'] = list(set([
    req for reqs in extras_require.values() for req in reqs
]))

setup(
    name="chemistry-agents",
    version="0.1.0",
    author="Niket Sharma",
    author_email="niket.sharma@example.com",
    description="AI-powered molecular property prediction and chemical engineering unit operations framework",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/niket-sharma/CHEMISTRY-AGENTS",
    project_urls={
        "Bug Tracker": "https://github.com/niket-sharma/CHEMISTRY-AGENTS/issues",
        "Documentation": "https://github.com/niket-sharma/CHEMISTRY-AGENTS#readme",
        "Source Code": "https://github.com/niket-sharma/CHEMISTRY-AGENTS",
    },
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Healthcare Industry",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Chemistry",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=core_requirements,
    extras_require=extras_require,
    include_package_data=True,
    package_data={
        "chemistry_agents": [
            "configs/*.json",
            "data/*.csv",
            "models/*.pt"
        ]
    },
    entry_points={
        "console_scripts": [
            "chemistry-agents-train=chemistry_agents.scripts.train_model:main",
            "chemistry-agents-finetune=chemistry_agents.scripts.fine_tune_transformer:main",
            "chemistry-agents-predict=chemistry_agents.scripts.predict:main",
        ],
    },
    keywords=[
        "chemistry", "molecular-property-prediction", "drug-discovery", 
        "cheminformatics", "machine-learning", "deep-learning", 
        "transformers", "graph-neural-networks", "pytorch", "rdkit",
        "chemical-engineering", "unit-operations", "distillation", 
        "heat-exchangers", "reactors", "separation-processes"
    ],
    license="MIT",
    zip_safe=False,
    platforms=["any"],
)