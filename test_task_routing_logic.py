#!/usr/bin/env python3
"""
Test Task Routing Logic

Shows exactly how the intelligent agent decides which task type based on keywords.
"""

import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_routing_logic():
    """Test the task routing decision logic"""

    print("TASK ROUTING DECISION LOGIC TEST")
    print("=" * 50)
    print("This shows HOW the agent decides which specialized model to use")
    print()

    # The keyword patterns (copied from the agent)
    task_patterns = {
        "solubility": [
            "solubility", "soluble", "dissolution", "dissolve", "aqueous",
            "water soluble", "logS", "hydrophilic", "hydrophobic"
        ],
        "toxicity": [
            "toxic", "toxicity", "poison", "harmful", "dangerous", "LD50",
            "cytotoxic", "mutagenic", "carcinogenic", "safety", "adverse"
        ],
        "bioactivity": [
            "bioactivity", "bioactive", "activity", "active", "potency",
            "efficacy", "therapeutic", "pharmacological", "biological activity",
            "IC50", "EC50", "binding", "receptor"
        ]
    }

    def analyze_query(query):
        """Simulate the exact logic from the agent"""
        query_lower = query.lower()
        task_scores = {}

        print(f"Query: '{query}'")
        print(f"Lowercase: '{query_lower}'")
        print("Keyword matching:")

        # Score each task type based on keyword matches
        for task_type, patterns in task_patterns.items():
            score = 0
            matches = []
            for pattern in patterns:
                if pattern in query_lower:
                    # Weight longer patterns more heavily
                    points = len(pattern.split())
                    score += points
                    matches.append(f"'{pattern}' (+{points})")

            task_scores[task_type] = score
            if matches:
                print(f"  {task_type}: {score} points - {', '.join(matches)}")
            else:
                print(f"  {task_type}: {score} points - no matches")

        # Find highest scoring task
        if not any(task_scores.values()):
            result_task = "general"
            confidence = 0.0
        else:
            result_task = max(task_scores, key=task_scores.get)
            max_score = task_scores[result_task]
            confidence = min(max_score / 3.0, 1.0)  # Normalize confidence

        print(f"DECISION: {result_task} (confidence: {confidence:.2f})")
        print()

        return result_task, confidence

    # Test different types of queries
    test_queries = [
        "How toxic is benzene?",
        "What's the water solubility of aspirin?",
        "Is this compound bioactive?",
        "Find molecules similar to caffeine",
        "Can this drug dissolve in aqueous solution?",
        "What's the LD50 of this compound?",
        "Analyze the therapeutic efficacy of ibuprofen",
        "Calculate molecular similarity between benzene and toluene"
    ]

    print("TESTING DIFFERENT QUERY TYPES:")
    print("-" * 50)

    for query in test_queries:
        analyze_query(query)

    print("KEY INSIGHTS:")
    print("-" * 30)
    print("1. KEYWORD MATCHING: Agent uses simple string matching on predefined keywords")
    print("2. SCORING: Longer keyword phrases get more points (e.g., 'water soluble' = 2 points)")
    print("3. CONFIDENCE: Score divided by 3, capped at 1.0")
    print("4. FALLBACK: If no keywords match, defaults to 'general' ChemBERTa")
    print()
    print("ChemBERTa MODEL ROLE:")
    print("- ChemBERTa is NOT used for task detection")
    print("- ChemBERTa is used for actual molecular analysis AFTER task is determined")
    print("- Base ChemBERTa: similarity, clustering, embeddings")
    print("- Specialized ChemBERTa: task-specific predictions (when trained)")

if __name__ == "__main__":
    test_routing_logic()