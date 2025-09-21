#!/usr/bin/env python3
"""
ChemBERTa Molecular Intelligence Demo

This demonstrates what ChemBERTa does BETTER than GPT:
- Molecular similarity beyond human intuition
- Chemical space exploration
- Hidden pattern discovery
- Molecular clustering by learned features

ChemBERTa was trained on 77 MILLION molecules!
"""

import sys
import os
import numpy as np

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def main():
    print("CHEMBERTA MOLECULAR INTELLIGENCE DEMO")
    print("=" * 60)
    print("ChemBERTa excels at tasks where molecular understanding matters!")
    print("This is where it BEATS GPT hands down! 🧬")

    try:
        from chemistry_agents.agents.chemberta_agent import ChemBERTaAgent
        from chemistry_agents.agents.base_agent import AgentConfig

        # Create ChemBERTa agent
        config = AgentConfig(device="cpu", log_level="INFO")
        agent = ChemBERTaAgent(config)

        print(f"\n1. LOADING CHEMBERTA MODEL...")
        agent.load_model()

        if not agent.is_loaded:
            print("❌ ChemBERTa model failed to load")
            print("💡 Try running: python download_huggingface_model.py")
            return

        print(f"✅ ChemBERTa loaded successfully!")
        print(f"   This model understands 77 MILLION molecules!")

        # Demo 1: Molecular Similarity (ChemBERTa's superpower)
        print(f"\n2. MOLECULAR SIMILARITY ANALYSIS")
        print("-" * 40)
        print("ChemBERTa finds similarity patterns humans would miss!")

        molecules = ["CCO", "c1ccccc1", "CC(=O)O", "CCN", "c1ccc(cc1)O"]
        molecule_names = ["ethanol", "benzene", "acetic_acid", "ethylamine", "phenol"]

        print(f"\nMolecules to analyze:")
        for mol, name in zip(molecules, molecule_names):
            print(f"  {mol:15s} - {name}")

        # Calculate all pairwise similarities
        print(f"\nChemBERTa Similarity Matrix:")
        print("(1.0 = identical, 0.0 = completely different)")

        for i, (mol1, name1) in enumerate(zip(molecules, molecule_names)):
            for j, (mol2, name2) in enumerate(zip(molecules, molecule_names)):
                if i <= j:
                    if i == j:
                        sim = 1.0
                    else:
                        sim = agent.calculate_molecular_similarity(mol1, mol2)

                    print(f"{name1:12s} vs {name2:12s}: {sim:.3f}")

        # Demo 2: Find Similar Molecules
        print(f"\n3. FINDING SIMILAR MOLECULES")
        print("-" * 40)
        target = "CCO"  # ethanol
        candidates = ["CCN", "CCC", "CC(C)O", "c1ccccc1", "CC(=O)O"]

        print(f"Target molecule: {target} (ethanol)")
        print(f"Finding most similar from candidates...")

        similar = agent.find_similar_molecules(target, candidates, top_k=3)
        for mol, sim in similar:
            print(f"  {mol:10s} similarity: {sim:.3f}")

        # Demo 3: Molecular Clustering
        print(f"\n4. MOLECULAR CLUSTERING")
        print("-" * 40)
        print("ChemBERTa groups molecules by learned chemical features!")

        cluster_molecules = [
            "CCO", "CCN", "CCC",  # Small organic molecules
            "c1ccccc1", "c1ccc(cc1)O", "Cc1ccccc1",  # Aromatics
            "CC(=O)O", "CC(=O)C", "O"  # Small polar molecules
        ]

        clusters = agent.cluster_molecules(cluster_molecules, n_clusters=3)
        print(f"ChemBERTa found {len(clusters)} clusters:")

        for cluster_id, cluster_mols in clusters.items():
            print(f"  Cluster {cluster_id}: {cluster_mols}")

        # Demo 4: Chemical Space Analysis
        print(f"\n5. CHEMICAL SPACE ANALYSIS")
        print("-" * 40)
        print("Analyzing molecular diversity and patterns...")

        space_analysis = agent.analyze_chemical_space(cluster_molecules)

        if "error" not in space_analysis:
            print(f"Molecules analyzed: {space_analysis['num_molecules']}")
            print(f"Embedding dimension: {space_analysis['embedding_dimension']}")

            diversity = space_analysis["diversity_metrics"]
            print(f"Chemical diversity score: {diversity['diversity_score']:.3f}")
            print(f"Mean similarity: {diversity['mean_similarity']:.3f}")

            if "most_similar_pair" in space_analysis["similarity_matrix"]:
                most_sim = space_analysis["similarity_matrix"]["most_similar_pair"]
                print(f"Most similar pair: {most_sim['molecules']} ({most_sim['similarity']:.3f})")

            if "cluster_analysis" in space_analysis:
                clusters = space_analysis["cluster_analysis"]
                print(f"Chemical clusters found: {clusters['num_clusters']}")

        # Demo 5: Molecular Embeddings
        print(f"\n6. MOLECULAR EMBEDDINGS")
        print("-" * 40)
        print("ChemBERTa converts molecules to 384-dimensional vectors!")

        test_molecule = "CCO"
        embedding = agent.get_molecular_embedding(test_molecule)

        print(f"Molecule: {test_molecule}")
        print(f"Embedding shape: {embedding.shape}")
        print(f"Embedding magnitude: {np.linalg.norm(embedding):.3f}")
        print(f"Sample values: {embedding[:5]}")  # Show first 5 dimensions

        # Demo 6: What makes ChemBERTa special
        print(f"\n7. WHY CHEMBERTA BEATS GPT")
        print("-" * 40)
        advantages = agent.explain_chemberta_advantages()
        print(advantages)

        # Demo 7: Practical Applications
        print(f"\n8. PRACTICAL APPLICATIONS")
        print("-" * 40)
        print("🔬 Drug Discovery:")
        print("   - Find molecules similar to known drugs")
        print("   - Cluster compound libraries")
        print("   - Discover structure-activity relationships")

        print(f"\n🧪 Chemical Research:")
        print("   - Explore chemical space")
        print("   - Find unexpected molecular similarities")
        print("   - Guide synthesis planning")

        print(f"\n⚗️ Materials Science:")
        print("   - Cluster materials by properties")
        print("   - Find similar polymers or catalysts")
        print("   - Predict material behavior")

        print(f"\n📊 Data Analysis:")
        print("   - Analyze molecular datasets")
        print("   - Find outliers and patterns")
        print("   - Reduce molecular data dimensionality")

    except ImportError as e:
        print(f"❌ Import failed: {e}")
        print("💡 Make sure ChemBERTa model is downloaded")
    except Exception as e:
        print(f"❌ Demo failed: {e}")

    print(f"\n🎉 CHEMBERTA DEMO COMPLETED!")
    print("=" * 60)
    print("ChemBERTa provides molecular intelligence that GPT simply can't match!")
    print("It understands chemistry at the molecular level. 🧬🤖")

def interactive_similarity_search():
    """Interactive similarity search demo"""
    print("\n" + "="*60)
    print("INTERACTIVE CHEMBERTA SIMILARITY SEARCH")
    print("="*60)

    try:
        from chemistry_agents.agents.chemberta_agent import ChemBERTaAgent
        from chemistry_agents.agents.base_agent import AgentConfig

        agent = ChemBERTaAgent(AgentConfig(device="cpu"))
        agent.load_model()

        if not agent.is_loaded:
            print("❌ ChemBERTa not available")
            return

        # Drug molecules database
        drug_database = {
            "aspirin": "CC(=O)OC1=CC=CC=C1C(=O)O",
            "ibuprofen": "CC(C)Cc1ccc(C(C)C(=O)O)cc1",
            "caffeine": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
            "acetaminophen": "CC(=O)Nc1ccc(O)cc1",
            "ethanol": "CCO",
            "benzene": "c1ccccc1",
            "morphine": "CN1CCC23C4C1CC5=C2C(=C(C=C5)O)OC3C(C=C4)O"
        }

        print("Available molecules:")
        for name, smiles in drug_database.items():
            print(f"  {name:15s}: {smiles}")

        print(f"\nFinding similarities for aspirin...")
        target = drug_database["aspirin"]
        candidates = [smiles for name, smiles in drug_database.items() if name != "aspirin"]

        similar_drugs = agent.find_similar_molecules(target, candidates, top_k=3)

        print(f"Most similar to aspirin:")
        for smiles, similarity in similar_drugs:
            # Find the name
            name = [name for name, s in drug_database.items() if s == smiles][0]
            print(f"  {name:15s}: {similarity:.3f} similarity")

    except Exception as e:
        print(f"❌ Interactive demo failed: {e}")

if __name__ == "__main__":
    main()
    interactive_similarity_search()