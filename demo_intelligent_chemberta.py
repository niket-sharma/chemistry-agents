#!/usr/bin/env python3
"""
Intelligent ChemBERTa Agent Demo

Demonstrates the intelligent task routing capabilities where the agent
automatically detects query types and routes to appropriate analysis methods.

This shows how the agent can determine what type of task is being requested:
- Solubility queries → Would use solubility-trained model
- Toxicity queries → Would use toxicity-trained model
- Bioactivity queries → Would use bioactivity-trained model
- General queries → Uses base ChemBERTa for similarity/clustering
"""

import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def main():
    """Main demo of intelligent ChemBERTa agent"""

    print("INTELLIGENT CHEMBERTA AGENT DEMO")
    print("=" * 60)
    print("Showcasing automatic task detection and intelligent routing!")
    print()

    try:
        from chemistry_agents.agents.intelligent_chemberta_agent import create_intelligent_chemberta_agent
        from chemistry_agents.agents.base_agent import AgentConfig

        # Create intelligent agent
        config = AgentConfig(device="cpu", log_level="WARNING")
        agent = create_intelligent_chemberta_agent(config)

        print(f"Base ChemBERTa Model: {'Loaded' if agent.is_loaded else 'Not Loaded'}")
        print(f"Specialized Models: {len(agent.specialized_models)} available")
        print(f"Model Parameters: {agent._count_parameters():,}")
        print()

        # Demo queries that show intelligent task routing
        demo_queries = [
            {
                "category": "TOXICITY ANALYSIS",
                "queries": [
                    "Is benzene toxic to humans?",
                    "What's the toxicity risk of this PCB compound?",
                    "How dangerous is ethanol for consumption?"
                ],
                "explanation": "These queries trigger toxicity detection and would use toxicity-specialized models"
            },
            {
                "category": "SOLUBILITY PREDICTION",
                "queries": [
                    "How soluble is aspirin in water?",
                    "What's the aqueous solubility of caffeine?",
                    "Can this compound dissolve in water?"
                ],
                "explanation": "These queries trigger solubility detection and would use solubility-specialized models"
            },
            {
                "category": "BIOACTIVITY ASSESSMENT",
                "queries": [
                    "What's the bioactivity of ibuprofen?",
                    "How pharmacologically active is this compound?",
                    "Analyze the therapeutic efficacy of aspirin"
                ],
                "explanation": "These queries trigger bioactivity detection and would use bioactivity-specialized models"
            },
            {
                "category": "MOLECULAR SIMILARITY",
                "queries": [
                    "Find molecules similar to caffeine",
                    "Compare benzene and toluene similarity",
                    "Cluster these molecules: CCO, CCN, c1ccccc1"
                ],
                "explanation": "These queries use general ChemBERTa for molecular intelligence tasks"
            }
        ]

        for category_info in demo_queries:
            print(f">> {category_info['category']}")
            print("-" * 50)
            print(f"Purpose: {category_info['explanation']}")
            print()

            for i, query in enumerate(category_info['queries'], 1):
                print(f"Query {i}: {query}")

                # Show task detection
                task_type, confidence = agent.detect_task_type(query)
                print(f"Detected Task: {task_type} (confidence: {confidence:.2f})")

                # Show what would happen
                if task_type in ['toxicity', 'solubility', 'bioactivity']:
                    if task_type in agent.specialized_models:
                        print(f"Routing: Using specialized {task_type} ChemBERTa model")
                    else:
                        print(f"Routing: Would use specialized {task_type} model (not trained yet)")
                        print(f"Fallback: Using general ChemBERTa analysis")
                else:
                    print(f"Routing: Using general ChemBERTa for molecular analysis")

                # Extract molecules
                molecules = agent._extract_molecules_from_query(query)
                if molecules:
                    print(f"Molecules Found: {', '.join(molecules)}")

                print()

            print("=" * 60 + "\n")

        # Show one complete example
        print(">> COMPLETE ANALYSIS EXAMPLE")
        print("=" * 60)

        example_query = "How toxic is benzene?"
        print(f"Query: {example_query}")
        print()

        print("Full ChemBERTa Analysis:")
        response = agent.chat(example_query)

        # Show key parts of the response
        lines = response.split('\n')
        for line in lines[:20]:  # First 20 lines
            print(line)

        if len(lines) > 20:
            print("... (analysis continues)")

        # Show reasoning process
        trace = agent.get_reasoning_trace()
        print(f"\nIntelligent Reasoning Process ({len(trace)} steps):")
        for step in trace:
            print(f"  {step.step_number}. {step.description}")

        print()
        print("=" * 60)

        # Summary of capabilities
        print("\n>> INTELLIGENT CHEMBERTA CAPABILITIES SUMMARY")
        print("=" * 60)
        print("- Automatic task type detection from natural language")
        print("- Intelligent routing to specialized models")
        print("- Molecular structure recognition and analysis")
        print("- Fallback to general ChemBERTa when needed")
        print("- Conversational interface with reasoning traces")
        print("- Superior molecular intelligence vs GPT")

        print(f"\n>> NEXT STEPS TO ENHANCE:")
        print("1. Train specialized models: python train_task_specific_chemberta.py")
        print("2. Models will be automatically loaded and used for routing")
        print("3. Improved accuracy on task-specific predictions")
        print("4. Larger datasets for better model performance")

    except Exception as e:
        print(f"Demo failed: {e}")
        import traceback
        traceback.print_exc()

def test_specific_molecules():
    """Test with specific molecules and tasks"""

    print("\n" + "=" * 80 + "\n")
    print("MOLECULE-SPECIFIC TESTING")
    print("=" * 40)

    try:
        from chemistry_agents.agents.intelligent_chemberta_agent import create_intelligent_chemberta_agent
        from chemistry_agents.agents.base_agent import AgentConfig

        agent = create_intelligent_chemberta_agent(AgentConfig(device="cpu", log_level="WARNING"))

        # Test specific molecules with different tasks
        molecule_tests = [
            {
                "molecule": "CCO",
                "name": "ethanol",
                "queries": [
                    "Is ethanol toxic?",
                    "How soluble is ethanol?",
                    "What's the bioactivity of ethanol?"
                ]
            },
            {
                "molecule": "c1ccccc1",
                "name": "benzene",
                "queries": [
                    "Is benzene carcinogenic?",
                    "Can benzene dissolve in water?",
                    "Find molecules similar to benzene"
                ]
            },
            {
                "molecule": "CC(=O)OC1=CC=CC=C1C(=O)O",
                "name": "aspirin",
                "queries": [
                    "What's aspirin's toxicity profile?",
                    "How water-soluble is aspirin?",
                    "What's the anti-inflammatory activity of aspirin?"
                ]
            }
        ]

        for mol_test in molecule_tests:
            print(f"\n>> TESTING {mol_test['name'].upper()} ({mol_test['molecule']})")
            print("-" * 50)

            for query in mol_test['queries']:
                task_type, confidence = agent.detect_task_type(query)
                molecules = agent._extract_molecules_from_query(query)

                print(f"Query: {query}")
                print(f"Task: {task_type} (confidence: {confidence:.2f})")
                print(f"Molecules: {molecules}")

                if mol_test['molecule'] in ' '.join(molecules):
                    print("Correctly identified target molecule")
                else:
                    print("Target molecule not explicitly found in query")

                print()

    except Exception as e:
        print(f"Molecule testing failed: {e}")

if __name__ == "__main__":
    main()
    test_specific_molecules()