#!/usr/bin/env python3
"""
Create sample molecular dataset for training
"""

import pandas as pd
import numpy as np
import os

def create_sample_dataset():
    """Create a sample solubility dataset"""
    
    # Sample SMILES and corresponding logS values (solubility)
    sample_data = [
        ("CCO", -0.77),  # Ethanol
        ("CCCCCCCC", 5.15),  # Octane
        ("c1ccccc1", 2.13),  # Benzene
        ("CC(C)O", -0.05),  # Isopropanol
        ("CCCCCCCCCCCCCCCC", 8.23),  # Hexadecane
        ("c1ccc(cc1)O", 1.46),  # Phenol
        ("CCC(=O)O", -0.17),  # Propionic acid
        ("CCCCO", 0.88),  # Butanol
        ("c1ccc2ccccc2c1", 3.30),  # Naphthalene
        ("CC(C)(C)O", 0.35),  # tert-Butanol
        ("CCCCCO", 1.36),  # Pentanol
        ("c1ccc(cc1)N", 1.37),  # Aniline
        ("CCc1ccccc1", 3.15),  # Ethylbenzene
        ("CCCCCCCO", 1.76),  # Heptanol
        ("c1ccc(cc1)C", 2.73),  # Toluene
        ("CCCCCCO", 1.25),  # Hexanol
        ("CC(C)C", 2.76),  # Isobutane
        ("c1ccc(cc1)CC", 3.15),  # Ethylbenzene
        ("CCCCCCCCCO", 2.13),  # Nonanol
        ("CC(C)CC", 3.11),  # Methylbutane
        ("c1ccc(cc1)CCC", 3.69),  # Propylbenzene
        ("CCCCCCCCCCCO", 2.54),  # Undecanol
        ("CC(C)(C)C", 3.11),  # Neopentane
        ("c1ccc2c(c1)cccc2", 3.30),  # Naphthalene
        ("CCCCCCCCCCCCO", 2.92),  # Dodecanol
        ("CCC(C)C", 3.11),  # Methylbutane
        ("c1ccc(cc1)CCCC", 4.26),  # Butylbenzene
        ("CCCCCCCCCCCCCO", 3.28),  # Tridecanol
        ("CCCC(C)C", 3.11),  # Methylpentane
        ("c1ccc(cc1)CCCCC", 4.90),  # Pentylbenzene
        # Add more diverse molecules
        ("CN(C)C", -1.26),  # Trimethylamine
        ("O", -1.38),  # Water
        ("CO", -0.77),  # Methanol
        ("CCN(CC)CC", 0.58),  # Triethylamine
        ("c1ccc(cc1)C(=O)O", 1.87),  # Benzoic acid
        ("CC(=O)O", -0.17),  # Acetic acid
        ("c1ccc(cc1)S", 2.52),  # Thiophenol
        ("CCCCN", 0.17),  # Butylamine
        ("c1ccc(cc1)Cl", 2.84),  # Chlorobenzene
        ("CC(C)N", -0.40),  # Isopropylamine
        ("c1ccc(cc1)F", 2.27),  # Fluorobenzene
        ("CCCCCN", 0.64),  # Pentylamine
        ("c1ccc(cc1)Br", 2.99),  # Bromobenzene
        ("CC(C)(C)N", 0.40),  # tert-Butylamine
        ("c1ccc(cc1)I", 3.25),  # Iodobenzene
        ("CCCCCCN", 1.12),  # Hexylamine
        ("c1ccc(cc1)[N+](=O)[O-]", 1.85),  # Nitrobenzene
        ("CCCCCCCN", 1.59),  # Heptylamine
        ("c1ccc(cc1)C#N", 1.56),  # Benzonitrile
        ("CCCCCCCCN", 2.07),  # Octylamine
    ]
    
    # Create DataFrame
    df = pd.DataFrame(sample_data, columns=['smiles', 'logS'])
    
    # Add some noise to make it more realistic
    np.random.seed(42)
    df['logS'] = df['logS'] + np.random.normal(0, 0.1, len(df))
    
    # Create data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)
    
    # Save dataset
    output_path = 'data/sample_solubility.csv'
    df.to_csv(output_path, index=False)
    
    print(f"Sample dataset created: {output_path}")
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print("\nFirst 5 rows:")
    print(df.head())
    
    return output_path

if __name__ == "__main__":
    create_sample_dataset()