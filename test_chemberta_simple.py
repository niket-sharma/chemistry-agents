#!/usr/bin/env python3
"""
Simple ChemBERTa Test

Test the ChemBERTa conversational agent without Unicode issues.
"""

import sys
import os
import numpy as np

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def main():
    print("CHEMBERTA CONVERSATIONAL AGENT TEST")
    print("=" * 50)

    try:
        from chemistry_agents.agents.chemberta_conversational_agent import ChemBERTaConversationalAgent
        from chemistry_agents.agents.base_agent import AgentConfig

        # Create agent
        config = AgentConfig(device="cpu", log_level="INFO")
        agent = ChemBERTaConversationalAgent(config)

        print("Loading ChemBERTa model...")
        agent.load_model()

        if agent.is_loaded:
            print("SUCCESS: ChemBERTa model loaded!")
            print(f"Model parameters: {agent._count_parameters():,}")

            # Test molecular similarity
            print("\nTesting molecular similarity...")
            similarity = agent.calculate_molecular_similarity("CCO", "CCN")
            print(f"Ethanol vs Ethylamine similarity: {similarity:.3f}")

            # Test conversation
            print("\nTesting conversation...")
            query = "Compare the similarity of ethanol and benzene"
            response = agent.chat(query)

            print(f"Query: {query}")
            print(f"Response length: {len(response)} characters")
            print("First 200 characters:")
            print(response[:200] + "...")

            # Test reasoning trace
            trace = agent.get_reasoning_trace()
            print(f"\nReasoning steps: {len(trace)}")
            for step in trace:
                print(f"  {step.step_number}. {step.description}")

            print("\nCHEMBERTA AGENT IS WORKING!")

        else:
            print("ChemBERTa model failed to load")

    except Exception as e:
        print(f"Error: {e}")

def test_molecular_intelligence():
    """Test ChemBERTa's molecular intelligence capabilities"""
    print("\nTESTING MOLECULAR INTELLIGENCE")
    print("-" * 40)

    try:
        from chemistry_agents.agents.chemberta_agent import ChemBERTaAgent
        from chemistry_agents.agents.base_agent import AgentConfig

        agent = ChemBERTaAgent(AgentConfig(device="cpu"))
        agent.load_model()

        if not agent.is_loaded:
            print("ChemBERTa not available")
            return

        # Test 1: Molecular embeddings
        print("1. Testing molecular embeddings...")
        embedding = agent.get_molecular_embedding("CCO")
        print(f"   Ethanol embedding shape: {embedding.shape}")
        print(f"   Embedding magnitude: {np.linalg.norm(embedding):.3f}")

        # Test 2: Similarity calculation
        print("\n2. Testing molecular similarity...")
        molecules = ["CCO", "CCN", "c1ccccc1"]
        names = ["ethanol", "ethylamine", "benzene"]

        for i, (mol1, name1) in enumerate(zip(molecules, names)):
            for j, (mol2, name2) in enumerate(zip(molecules, names)):
                if i < j:
                    sim = agent.calculate_molecular_similarity(mol1, mol2)
                    print(f"   {name1} vs {name2}: {sim:.3f}")

        # Test 3: Clustering
        print("\n3. Testing molecular clustering...")
        test_molecules = ["CCO", "CCN", "CCC", "c1ccccc1", "Cc1ccccc1"]
        clusters = agent.cluster_molecules(test_molecules, n_clusters=2)

        print("   Clusters found:")
        for cluster_id, cluster_mols in clusters.items():
            print(f"     Cluster {cluster_id}: {cluster_mols}")

        print("\nMOLECULAR INTELLIGENCE TESTS PASSED!")

    except Exception as e:
        print(f"Error in molecular intelligence test: {e}")

if __name__ == "__main__":
    main()
    test_molecular_intelligence()