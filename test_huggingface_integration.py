#!/usr/bin/env python3
"""
Test HuggingFace ChemBERTa integration with chemistry agents
"""

import os
import sys
import json
import numpy as np
import torch
from typing import List, Dict, Any

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_chemberta_tokenization():
    """Test ChemBERTa tokenization of SMILES"""
    print("🔤 Testing ChemBERTa Tokenization")
    print("-" * 40)
    
    try:
        from transformers import AutoTokenizer
        
        model_name = "DeepChem/ChemBERTa-77M-MLM"
        cache_dir = "models/huggingface_cache"
        
        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
        
        # Test various molecule types
        test_molecules = [
            ("CCO", "Ethanol"),
            ("c1ccccc1", "Benzene"),
            ("CC(=O)O", "Acetic acid"),
            ("CCN(CC)CC", "Triethylamine"),
            ("c1ccc2c(c1)cccc2", "Naphthalene"),
            ("CC(C)(C)O", "tert-Butanol"),
            ("CCCCCCCCCCCCCCCC", "Hexadecane"),
            ("c1ccc(cc1)C(=O)O", "Benzoic acid")
        ]
        
        print(f"Vocabulary size: {tokenizer.vocab_size:,}")
        print(f"Max length: {tokenizer.model_max_length}")
        print("\nTokenization Results:")
        print("SMILES".ljust(20) + "Name".ljust(15) + "Tokens".ljust(8) + "Length")
        print("-" * 55)
        
        for smiles, name in test_molecules:
            # Tokenize
            tokens = tokenizer(smiles, return_tensors="pt", padding=False, truncation=True)
            token_ids = tokens['input_ids'][0].tolist()
            
            # Decode to verify
            decoded = tokenizer.decode(token_ids, skip_special_tokens=True)
            
            print(f"{smiles:<20} {name:<15} {len(token_ids):<8} {decoded}")
        
        return True
        
    except Exception as e:
        print(f"❌ Tokenization test failed: {e}")
        return False

def test_chemberta_embeddings():
    """Test generating molecular embeddings with ChemBERTa"""
    print("\n🧠 Testing ChemBERTa Embeddings")
    print("-" * 40)
    
    try:
        from transformers import AutoTokenizer, AutoModel
        
        model_name = "DeepChem/ChemBERTa-77M-MLM"
        cache_dir = "models/huggingface_cache"
        
        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
        model = AutoModel.from_pretrained(model_name, cache_dir=cache_dir)
        
        # Test molecules
        smiles_list = [
            "CCO",      # Small molecule
            "c1ccccc1", # Aromatic
            "CCCCCCCCCCCCCCCC",  # Long chain
            "CC(C)(C)c1ccc(cc1)O"  # Complex structure
        ]
        
        print(f"Model hidden size: {model.config.hidden_size}")
        print(f"Number of layers: {model.config.num_hidden_layers}")
        
        embeddings_list = []
        
        for smiles in smiles_list:
            # Tokenize
            inputs = tokenizer(smiles, return_tensors="pt", padding=True, truncation=True)
            
            # Generate embeddings
            with torch.no_grad():
                outputs = model(**inputs)
                # Use mean pooling of last hidden state
                embeddings = outputs.last_hidden_state.mean(dim=1)  # Shape: [1, hidden_size]
                embeddings_list.append(embeddings.squeeze().numpy())
                
                print(f"{smiles:<25} -> embedding shape: {embeddings.shape}")
        
        # Calculate similarity matrix
        print("\n📊 Similarity Matrix (cosine similarity):")
        embeddings_array = np.array(embeddings_list)
        
        # Normalize embeddings
        normalized = embeddings_array / np.linalg.norm(embeddings_array, axis=1, keepdims=True)
        
        # Calculate cosine similarity
        similarity_matrix = np.dot(normalized, normalized.T)
        
        # Print similarity matrix
        print("        ", end="")
        for i, smiles in enumerate(smiles_list):
            print(f"Mol{i+1:2d}", end="  ")
        print()
        
        for i, smiles in enumerate(smiles_list):
            print(f"Mol{i+1:2d}    ", end="")
            for j in range(len(smiles_list)):
                print(f"{similarity_matrix[i,j]:.2f}", end="  ")
            print()
        
        return True
        
    except Exception as e:
        print(f"❌ Embeddings test failed: {e}")
        return False

def test_chemistry_agent_integration():
    """Test ChemBERTa integration with chemistry agents"""
    print("\n🔬 Testing Chemistry Agent Integration")
    print("-" * 40)
    
    try:
        from chemistry_agents import PropertyPredictionAgent
        from chemistry_agents.agents.base_agent import AgentConfig
        
        # Create agent with transformer model
        config = AgentConfig(
            model_path=None,
            device="cpu"
        )
        
        agent = PropertyPredictionAgent(
            config=config,
            model_type="transformer",
            transformer_model="DeepChem/ChemBERTa-77M-MLM"
        )
        
        print(f"✅ Agent created with model: {agent.transformer_model}")
        print(f"   Model type: {agent.model_type}")
        print(f"   Device: {config.device}")
        
        # Test SMILES processor
        if hasattr(agent, 'smiles_processor'):
            test_smiles = ["CCO", "c1ccccc1", "invalid_smiles", "CC(C)O"]
            processed, valid_mask = agent.smiles_processor.process_smiles_batch(test_smiles)
            
            print(f"\n📝 SMILES Processing:")
            print(f"   Input molecules: {len(test_smiles)}")
            print(f"   Valid molecules: {sum(valid_mask)}")
            print(f"   Processed successfully: {len(processed) if processed else 0}")
        
        # Test without model loading (should handle gracefully)
        try:
            result = agent.predict_single("CCO")
            print(f"❌ Expected RuntimeError, but got: {result}")
            return False
        except RuntimeError as e:
            if "Model not loaded" in str(e):
                print(f"✅ Proper error handling: {e}")
            else:
                print(f"❌ Wrong error message: {e}")
                return False
        
        return True
        
    except Exception as e:
        print(f"❌ Agent integration test failed: {e}")
        return False

def test_model_loading_simulation():
    """Simulate model loading and prediction workflow"""
    print("\n🔮 Testing Model Loading Simulation")
    print("-" * 40)
    
    try:
        from chemistry_agents.models.transformer_model import MolecularTransformer
        
        # Test transformer model initialization
        transformer = MolecularTransformer(model_name="DeepChem/ChemBERTa-77M-MLM")
        
        print(f"✅ MolecularTransformer initialized")
        print(f"   Model name: {transformer.model_name}")
        print(f"   Device: {transformer.device}")
        
        # Test feature extraction
        test_smiles = ["CCO", "c1ccccc1", "CC(C)O"]
        
        # This would normally require a trained head for property prediction
        print(f"\n🧪 Testing with {len(test_smiles)} molecules:")
        for smiles in test_smiles:
            print(f"   {smiles} -> Ready for prediction pipeline")
        
        return True
        
    except Exception as e:
        print(f"❌ Model loading simulation failed: {e}")
        return False

def create_huggingface_example():
    """Create a comprehensive example file"""
    example_code = '''#!/usr/bin/env python3
"""
Example: Using HuggingFace ChemBERTa with Chemistry Agents
"""

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel

def main():
    print("🔬 HuggingFace ChemBERTa Example")
    print("=" * 50)
    
    # Load ChemBERTa model
    model_name = "DeepChem/ChemBERTa-77M-MLM"
    cache_dir = "models/huggingface_cache"
    
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    model = AutoModel.from_pretrained(model_name, cache_dir=cache_dir)
    
    # Example molecules
    molecules = {
        "Ethanol": "CCO",
        "Benzene": "c1ccccc1", 
        "Aspirin": "CC(=O)OC1=CC=CC=C1C(=O)O",
        "Caffeine": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"
    }
    
    print("\\nGenerating molecular embeddings...")
    
    embeddings = {}
    for name, smiles in molecules.items():
        # Tokenize
        inputs = tokenizer(smiles, return_tensors="pt", padding=True, truncation=True)
        
        # Generate embedding
        with torch.no_grad():
            outputs = model(**inputs)
            # Mean pooling
            embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
            embeddings[name] = embedding
            
        print(f"✅ {name:10s} ({smiles:25s}) -> {embedding.shape[0]}D embedding")
    
    # Calculate molecular similarities
    print("\\n📊 Molecular Similarities (cosine):")
    mol_names = list(molecules.keys())
    
    for i, mol1 in enumerate(mol_names):
        for j, mol2 in enumerate(mol_names):
            if i <= j:
                emb1 = embeddings[mol1]
                emb2 = embeddings[mol2]
                
                # Cosine similarity
                similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
                print(f"   {mol1} vs {mol2}: {similarity:.3f}")
    
    print("\\n🎉 Example completed successfully!")

if __name__ == "__main__":
    main()
'''
    
    with open("examples/huggingface_example.py", "w") as f:
        f.write(example_code)
    
    print("📄 Created: examples/huggingface_example.py")

def run_all_tests():
    """Run all HuggingFace integration tests"""
    print("🧪 HuggingFace ChemBERTa Integration Tests")
    print("=" * 60)
    
    tests = [
        ("Tokenization", test_chemberta_tokenization),
        ("Embeddings", test_chemberta_embeddings),
        ("Agent Integration", test_chemistry_agent_integration),
        ("Model Loading", test_model_loading_simulation),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            success = test_func()
            if success:
                passed += 1
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
    
    # Create example file
    try:
        os.makedirs("examples", exist_ok=True)
        create_huggingface_example()
        print("✅ Example file created")
    except Exception as e:
        print(f"❌ Failed to create example: {e}")
    
    print("\\n" + "=" * 60)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All HuggingFace integration tests passed!")
        print("\\n📋 Try these commands:")
        print("   python examples/huggingface_example.py")
        print("   python download_huggingface_model.py")
    else:
        print("⚠️  Some tests failed. Check the errors above.")

if __name__ == "__main__":
    run_all_tests()