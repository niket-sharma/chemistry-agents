#!/usr/bin/env python3
"""
Download and cache HuggingFace ChemBERTa model for chemistry agents
"""

import os
import sys
from pathlib import Path

def download_chemberta_model():
    """Download and cache ChemBERTa model"""
    
    try:
        from transformers import AutoTokenizer, AutoModel
        
        print("🔬 Downloading ChemBERTa model for chemistry agents...")
        print("=" * 60)
        
        # Model details
        model_name = "DeepChem/ChemBERTa-77M-MLM"
        cache_dir = "models/huggingface_cache"
        
        # Create cache directory
        os.makedirs(cache_dir, exist_ok=True)
        
        print(f"📥 Downloading model: {model_name}")
        print(f"📁 Cache directory: {cache_dir}")
        
        # Download tokenizer
        print("\n🔤 Downloading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=cache_dir
        )
        print(f"✅ Tokenizer downloaded (vocab size: {tokenizer.vocab_size})")
        
        # Download model
        print("\n🧠 Downloading model...")
        model = AutoModel.from_pretrained(
            model_name,
            cache_dir=cache_dir
        )
        print(f"✅ Model downloaded (parameters: {model.num_parameters():,})")
        
        # Test tokenization with sample molecules
        print("\n🧪 Testing with sample molecules...")
        sample_smiles = [
            "CCO",  # Ethanol
            "c1ccccc1",  # Benzene
            "CC(C)O",  # Isopropanol
            "CCCCO"  # Butanol
        ]
        
        for smiles in sample_smiles:
            # Test tokenization
            tokens = tokenizer(smiles, return_tensors="pt", padding=True, truncation=True)
            decoded = tokenizer.decode(tokens['input_ids'][0], skip_special_tokens=True)
            print(f"   {smiles:10} -> {len(tokens['input_ids'][0])} tokens -> {decoded}")
        
        # Test model inference
        print("\n🔮 Testing model inference...")
        import torch
        
        # Tokenize a batch
        batch_smiles = ["CCO", "c1ccccc1"]
        inputs = tokenizer(batch_smiles, return_tensors="pt", padding=True, truncation=True)
        
        # Get embeddings
        with torch.no_grad():
            outputs = model(**inputs)
            embeddings = outputs.last_hidden_state
            
        print(f"✅ Generated embeddings: {embeddings.shape}")
        print(f"   Batch size: {embeddings.shape[0]}")
        print(f"   Sequence length: {embeddings.shape[1]}")
        print(f"   Hidden size: {embeddings.shape[2]}")
        
        # Save model info
        model_info = {
            "model_name": model_name,
            "cache_dir": cache_dir,
            "vocab_size": tokenizer.vocab_size,
            "hidden_size": model.config.hidden_size,
            "num_layers": model.config.num_hidden_layers,
            "num_attention_heads": model.config.num_attention_heads,
            "parameters": model.num_parameters(),
            "sample_embedding_shape": list(embeddings.shape)
        }
        
        import json
        with open("models/chemberta_info.json", "w") as f:
            json.dump(model_info, f, indent=2)
        
        print(f"\n📊 Model Information saved to: models/chemberta_info.json")
        print("\n🎉 ChemBERTa model successfully downloaded and tested!")
        
        print("\n📋 Next steps:")
        print("   1. Run: python test_huggingface_integration.py")
        print("   2. Try: python examples/huggingface_example.py")
        
        return True
        
    except ImportError as e:
        print(f"❌ Missing dependencies: {e}")
        print("💡 Install with: pip install transformers torch")
        return False
        
    except Exception as e:
        print(f"❌ Download failed: {e}")
        return False

if __name__ == "__main__":
    # Create models directory
    os.makedirs("models", exist_ok=True)
    
    success = download_chemberta_model()
    if not success:
        sys.exit(1)