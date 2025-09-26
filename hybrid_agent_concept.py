#!/usr/bin/env python3
"""
Hybrid LLM + ChemBERT Agent Concept

Combines the power of:
- Large LLM (OpenAI GPT): Natural language, reasoning, general chemistry knowledge
- Specialized ChemBERT: Molecular intelligence, property prediction
"""

import asyncio
import os
import sys
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import json
from dotenv import load_dotenv
from pathlib import Path

# Add src to path for proper imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Load environment variables from .env file with override
load_dotenv(override=True)

@dataclass
class HybridAnalysisResult:
    """Result from hybrid LLM + ChemBERT analysis"""
    query: str
    llm_reasoning: str
    chemberta_predictions: Dict[str, Any]
    synthesis: str
    confidence: float
    reasoning_steps: List[str]

class HybridChemistryAgent:
    """
    Intelligent agent that combines LLM reasoning with ChemBERT molecular intelligence
    """

    def __init__(self, openai_api_key: str = None):
        self.openai_api_key = openai_api_key or os.getenv('OPENAI_API_KEY')
        self.openai_model = os.getenv('OPENAI_MODEL', 'gpt-4')
        self.openai_temperature = float(os.getenv('OPENAI_TEMPERATURE', '0.1'))
        self.openai_max_tokens = int(os.getenv('OPENAI_MAX_TOKENS', '2000'))
        self.chemberta_agent = None  # Our existing intelligent ChemBERT agent
        self.llm_client = None  # OpenAI client

    async def initialize(self):
        """Initialize both LLM and ChemBERT components"""

        # Initialize ChemBERT agent
        from chemistry_agents.agents.intelligent_chemberta_agent import create_intelligent_chemberta_agent
        from chemistry_agents.agents.base_agent import AgentConfig

        config = AgentConfig(device="cpu", log_level="WARNING")
        self.chemberta_agent = create_intelligent_chemberta_agent(config)

        # Initialize OpenAI client
        if self.openai_api_key:
            import openai
            self.llm_client = openai.OpenAI(api_key=self.openai_api_key)

    async def analyze_complex_query(self, user_query: str) -> HybridAnalysisResult:
        """
        Main hybrid analysis pipeline combining LLM and ChemBERT
        """

        reasoning_steps = []

        # Step 1: LLM analyzes query and plans approach
        llm_analysis = await self._llm_query_analysis(user_query)
        reasoning_steps.append(f"LLM Analysis: {llm_analysis['approach']}")

        # Step 2: Extract molecules and chemistry tasks from LLM analysis
        molecules = llm_analysis.get('molecules', [])
        tasks = llm_analysis.get('chemistry_tasks', [])

        # Step 3: Use ChemBERT for molecular predictions
        chemberta_results = {}
        if molecules:
            for molecule in molecules:
                result = self.chemberta_agent.chat(f"Analyze {molecule} for {', '.join(tasks)}")
                chemberta_results[molecule] = result
                reasoning_steps.append(f"ChemBERT analyzed {molecule}")

        # Step 4: LLM synthesizes results with domain knowledge
        synthesis_prompt = self._build_synthesis_prompt(
            user_query, llm_analysis, chemberta_results
        )

        synthesis = await self._llm_synthesis(synthesis_prompt)
        reasoning_steps.append("LLM synthesized final answer")

        return HybridAnalysisResult(
            query=user_query,
            llm_reasoning=llm_analysis['reasoning'],
            chemberta_predictions=chemberta_results,
            synthesis=synthesis,
            confidence=0.85,  # Calculate based on both components
            reasoning_steps=reasoning_steps
        )

    async def _llm_query_analysis(self, query: str) -> Dict[str, Any]:
        """LLM analyzes the query and plans the approach"""

        system_prompt = """
        You are a chemistry expert AI. Analyze chemistry queries and plan how to approach them.

        For each query, identify:
        1. The main chemistry question
        2. What molecules are mentioned
        3. What chemistry tasks are needed (toxicity, solubility, bioactivity, etc.)
        4. What domain knowledge is required
        5. Your reasoning approach

        Return ONLY a valid JSON object with these exact keys:
        {
            "approach": "brief description of approach",
            "molecules": ["list", "of", "molecules"],
            "chemistry_tasks": ["list", "of", "tasks"],
            "reasoning": "detailed reasoning about the query"
        }
        """

        response = await self._call_llm(system_prompt, query)
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            # Fallback if JSON parsing fails
            return {
                "approach": "Direct analysis needed",
                "molecules": [],
                "chemistry_tasks": ["general"],
                "reasoning": response
            }

    async def _llm_synthesis(self, synthesis_prompt: str) -> str:
        """LLM synthesizes ChemBERT results with domain knowledge"""

        system_prompt = """
        You are a chemistry expert. You have computational predictions from specialized models.
        Synthesize these results with your chemistry knowledge to provide comprehensive answers.

        Explain:
        - What the computational predictions mean
        - How they relate to chemistry principles
        - Practical implications
        - Any limitations or considerations
        """

        return await self._call_llm(system_prompt, synthesis_prompt)

    async def _call_llm(self, system_prompt: str, user_prompt: str) -> str:
        """Call OpenAI API"""

        if not self.llm_client:
            return "LLM not available - provide OpenAI API key"

        response = self.llm_client.chat.completions.create(
            model=self.openai_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=self.openai_temperature,
            max_tokens=self.openai_max_tokens
        )

        return response.choices[0].message.content

    def _build_synthesis_prompt(self, original_query: str,
                               llm_analysis: Dict,
                               chemberta_results: Dict) -> str:
        """Build prompt for LLM to synthesize results"""

        prompt = f"""
        Original Question: {original_query}

        My Initial Analysis: {llm_analysis.get('reasoning', '')}

        Computational Predictions from ChemBERT:
        """

        for molecule, prediction in chemberta_results.items():
            prompt += f"\n{molecule}: {prediction}"

        prompt += f"""

        Please provide a comprehensive answer that:
        1. Directly answers the original question
        2. Explains what the computational predictions mean
        3. Relates predictions to chemistry principles
        4. Provides practical implications
        5. Notes any limitations or considerations
        """

        return prompt

# Example usage patterns for different hybrid tasks

class DrugDesignAssistant(HybridChemistryAgent):
    """Specialized hybrid agent for drug design tasks"""

    async def design_drug_variant(self, base_drug: str, requirements: str) -> HybridAnalysisResult:
        """Design drug variants based on requirements"""

        query = f"Design a variant of {base_drug} that meets these requirements: {requirements}"
        return await self.analyze_complex_query(query)

class ChemistryTutor(HybridChemistryAgent):
    """Specialized hybrid agent for chemistry education"""

    async def explain_with_examples(self, concept: str) -> HybridAnalysisResult:
        """Explain chemistry concepts with molecular examples"""

        query = f"Explain {concept} with specific molecular examples and predictions"
        return await self.analyze_complex_query(query)

class SafetyAssessmentAgent(HybridChemistryAgent):
    """Specialized hybrid agent for safety assessment"""

    async def assess_safety(self, compound: str, use_case: str) -> HybridAnalysisResult:
        """Comprehensive safety assessment with regulatory context"""

        query = f"Assess the safety of {compound} for use in {use_case}, including regulatory considerations"
        return await self.analyze_complex_query(query)

# Example test function
async def test_hybrid_agent():
    """Test the hybrid agent concept"""

    agent = HybridChemistryAgent()
    await agent.initialize()

    # Test complex drug design query
    query = "I need a painkiller similar to ibuprofen but with better water solubility. What are my options?"

    result = await agent.analyze_complex_query(query)

    print("Hybrid Analysis Result:")
    print(f"Query: {result.query}")
    print(f"LLM Reasoning: {result.llm_reasoning}")
    print(f"ChemBERT Predictions: {result.chemberta_predictions}")
    print(f"Final Synthesis: {result.synthesis}")
    print(f"Reasoning Steps: {result.reasoning_steps}")

if __name__ == "__main__":
    # Check if API key is available
    if os.getenv('OPENAI_API_KEY'):
        print("Hybrid LLM + ChemBERT Agent Ready!")
        print(f"OpenAI Model: {os.getenv('OPENAI_MODEL', 'gpt-4')}")
        print("Run: asyncio.run(test_hybrid_agent()) to test")
    else:
        print("Hybrid LLM + ChemBERT Agent Concept Ready!")
        print("⚠️  Add your OpenAI API key to .env file to test:")
        print("   OPENAI_API_KEY=your_key_here")