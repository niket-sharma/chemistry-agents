#!/usr/bin/env python3
"""
ChemBERTa Query Testing Script

Test specific queries that showcase ChemBERTa's molecular intelligence.
This script shows FULL responses without truncation.
"""

import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_chemberta_queries():
    """Test ChemBERTa with specific molecular intelligence queries"""

    print("CHEMBERTA MOLECULAR INTELLIGENCE TESTING")
    print("=" * 60)

    try:
        from chemistry_agents.agents.chemberta_conversational_agent import ChemBERTaConversationalAgent
        from chemistry_agents.agents.base_agent import AgentConfig

        # Create agent with minimal logging
        config = AgentConfig(device="cpu", log_level="WARNING")
        agent = ChemBERTaConversationalAgent(config)

        print("Loading ChemBERTa model...")
        agent.load_model()

        if not agent.is_loaded:
            print("ERROR: ChemBERTa model failed to load")
            return

        print("ChemBERTa loaded successfully!")
        print(f"Model parameters: {agent._count_parameters():,}")
        print()

        # Test queries that showcase ChemBERTa's superiority over GPT
        test_queries = [
            {
                "query": "Find molecules similar to aspirin",
                "why_chemberta_wins": "ChemBERTa understands molecular structure and calculates real chemical similarity"
            },
            {
                "query": "Compare the similarity of benzene and toluene",
                "why_chemberta_wins": "ChemBERTa works in 384D chemical space, not just text patterns"
            },
            {
                "query": "Cluster these molecules: CCO, CCN, c1ccccc1",
                "why_chemberta_wins": "ChemBERTa groups by learned chemical features, not structural patterns"
            },
            {
                "query": "How diverse are ethanol, benzene, and caffeine?",
                "why_chemberta_wins": "ChemBERTa calculates true chemical space diversity metrics"
            }
        ]

        for i, test in enumerate(test_queries, 1):
            print(f"TEST {i}: {test['query']}")
            print("-" * 50)
            print(f"WHY CHEMBERTA BEATS GPT: {test['why_chemberta_wins']}")
            print()

            try:
                # Get full response
                response = agent.chat(test['query'])
                print("CHEMBERTA RESPONSE:")
                print(response)

                # Show reasoning
                trace = agent.get_reasoning_trace()
                print(f"\nREASONING STEPS ({len(trace)}):")
                for step in trace:
                    print(f"  {step.step_number}. {step.description}")

            except Exception as e:
                print(f"ERROR: {e}")

            print("\n" + "=" * 60 + "\n")

        print("CHEMBERTA MOLECULAR INTELLIGENCE TESTING COMPLETE!")

    except Exception as e:
        print(f"Failed to run ChemBERTa tests: {e}")

def test_individual_capabilities():
    """Test individual ChemBERTa capabilities"""

    print("TESTING INDIVIDUAL CHEMBERTA CAPABILITIES")
    print("=" * 50)

    try:
        from chemistry_agents.agents.chemberta_agent import ChemBERTaAgent
        from chemistry_agents.agents.base_agent import AgentConfig

        agent = ChemBERTaAgent(AgentConfig(device="cpu", log_level="WARNING"))
        agent.load_model()

        if not agent.is_loaded:
            print("ChemBERTa model not available")
            return

        # Test 1: Molecular embeddings
        print("1. MOLECULAR EMBEDDINGS:")
        molecules = ["CCO", "c1ccccc1", "CC(=O)O"]
        names = ["ethanol", "benzene", "acetic acid"]

        for mol, name in zip(molecules, names):
            embedding = agent.get_molecular_embedding(mol)
            magnitude = (embedding ** 2).sum() ** 0.5
            print(f"   {name} ({mol}): 384D vector, magnitude: {magnitude:.3f}")

        print()

        # Test 2: Molecular similarity
        print("2. MOLECULAR SIMILARITY:")
        similarity_pairs = [
            ("CCO", "CCN", "ethanol vs ethylamine"),
            ("c1ccccc1", "Cc1ccccc1", "benzene vs toluene"),
            ("CCO", "c1ccccc1", "ethanol vs benzene")
        ]

        for mol1, mol2, description in similarity_pairs:
            sim = agent.calculate_molecular_similarity(mol1, mol2)
            print(f"   {description}: {sim:.3f}")

        print()

        # Test 3: Chemical clustering
        print("3. MOLECULAR CLUSTERING:")
        test_molecules = ["CCO", "CCN", "CCC", "c1ccccc1", "Cc1ccccc1"]
        mol_names = ["ethanol", "ethylamine", "propane", "benzene", "toluene"]

        clusters = agent.cluster_molecules(test_molecules, n_clusters=2)
        print("   Clusters found:")
        for cluster_id, cluster_mols in clusters.items():
            # Map back to names
            cluster_names = []
            for mol in cluster_mols:
                if mol in test_molecules:
                    idx = test_molecules.index(mol)
                    cluster_names.append(mol_names[idx])
            print(f"     Cluster {cluster_id}: {cluster_names}")

        print("\nINDIVIDUAL CAPABILITY TESTING COMPLETE!")

    except Exception as e:
        print(f"Individual capability testing failed: {e}")

if __name__ == "__main__":
    test_chemberta_queries()
    print("\n" + "=" * 80 + "\n")
    test_individual_capabilities()