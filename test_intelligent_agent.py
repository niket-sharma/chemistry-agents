#!/usr/bin/env python3
"""
Test Intelligent ChemBERTa Agent

Demonstrates intelligent task routing and specialized model usage.
Even without trained models, shows the task detection capabilities.
"""

import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_task_detection():
    """Test the task detection capabilities"""

    print("INTELLIGENT CHEMBERTA AGENT - TASK DETECTION TEST")
    print("=" * 60)

    try:
        from chemistry_agents.agents.intelligent_chemberta_agent import IntelligentChemBERTaAgent
        from chemistry_agents.agents.base_agent import AgentConfig

        # Create intelligent agent
        config = AgentConfig(device="cpu", log_level="WARNING")
        agent = IntelligentChemBERTaAgent(config)

        print("Intelligent ChemBERTa Agent initialized")
        print(f"Available specialized models: {list(agent.specialized_models.keys())}")
        print()

        # Test task detection on various queries
        test_queries = [
            {
                "query": "Is aspirin toxic to humans?",
                "expected_task": "toxicity"
            },
            {
                "query": "How soluble is caffeine in water?",
                "expected_task": "solubility"
            },
            {
                "query": "What's the bioactivity of ibuprofen?",
                "expected_task": "bioactivity"
            },
            {
                "query": "Find molecules similar to benzene",
                "expected_task": "general"
            },
            {
                "query": "Predict the aqueous solubility of ethanol",
                "expected_task": "solubility"
            },
            {
                "query": "Is this compound mutagenic or carcinogenic?",
                "expected_task": "toxicity"
            },
            {
                "query": "Analyze the therapeutic efficacy of this drug",
                "expected_task": "bioactivity"
            }
        ]

        print("TASK DETECTION RESULTS:")
        print("-" * 40)

        correct_detections = 0
        for i, test in enumerate(test_queries, 1):
            detected_task, confidence = agent.detect_task_type(test["query"])
            expected = test["expected_task"]
            correct = "✓" if detected_task == expected else "✗"

            if detected_task == expected:
                correct_detections += 1

            print(f"{i}. Query: {test['query']}")
            print(f"   Expected: {expected} | Detected: {detected_task} | Confidence: {confidence:.2f} {correct}")
            print()

        accuracy = correct_detections / len(test_queries) * 100
        print(f"Task Detection Accuracy: {accuracy:.1f}% ({correct_detections}/{len(test_queries)})")

    except Exception as e:
        print(f"Task detection test failed: {e}")

def test_intelligent_routing():
    """Test intelligent routing with conversational interface"""

    print("\n" + "=" * 60)
    print("INTELLIGENT ROUTING TEST")
    print("=" * 60)

    try:
        from chemistry_agents.agents.intelligent_chemberta_agent import create_intelligent_chemberta_agent
        from chemistry_agents.agents.base_agent import AgentConfig

        # Create agent
        config = AgentConfig(device="cpu", log_level="WARNING")
        agent = create_intelligent_chemberta_agent(config)

        print("Intelligent agent created and models loaded")
        print(f"Base ChemBERTa loaded: {agent.is_loaded}")
        print(f"Specialized models: {list(agent.specialized_models.keys())}")
        print()

        # Test queries that should route to different models
        routing_tests = [
            {
                "query": "How toxic is benzene?",
                "description": "Should route to toxicity model (if available)"
            },
            {
                "query": "What's the water solubility of aspirin?",
                "description": "Should route to solubility model (if available)"
            },
            {
                "query": "Find molecules similar to caffeine",
                "description": "Should use general ChemBERTa for similarity"
            }
        ]

        for i, test in enumerate(routing_tests, 1):
            print(f"TEST {i}: {test['query']}")
            print(f"Expected: {test['description']}")
            print("-" * 40)

            try:
                # This will show the reasoning trace
                response = agent.chat(test['query'])

                # Show first part of response
                lines = response.split('\n')
                for line in lines[:15]:  # Show first 15 lines
                    print(line)

                if len(lines) > 15:
                    print("... (response continues)")

                # Show reasoning
                trace = agent.get_reasoning_trace()
                print(f"\nReasoning Steps ({len(trace)}):")
                for step in trace:
                    print(f"  {step.step_number}. {step.description}")

            except Exception as e:
                print(f"Error: {e}")

            print("\n" + "=" * 60 + "\n")

    except Exception as e:
        print(f"Intelligent routing test failed: {e}")

def test_capabilities():
    """Test the explain capabilities function"""

    print("AGENT CAPABILITIES TEST")
    print("=" * 40)

    try:
        from chemistry_agents.agents.intelligent_chemberta_agent import create_intelligent_chemberta_agent
        from chemistry_agents.agents.base_agent import AgentConfig

        agent = create_intelligent_chemberta_agent(AgentConfig(device="cpu", log_level="WARNING"))

        capabilities = agent.explain_capabilities()
        print(capabilities)

    except Exception as e:
        print(f"Capabilities test failed: {e}")

if __name__ == "__main__":
    test_task_detection()
    test_intelligent_routing()
    print("\n" + "=" * 80 + "\n")
    test_capabilities()