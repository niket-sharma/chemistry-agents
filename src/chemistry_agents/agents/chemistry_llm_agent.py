"""
True LLM Chemistry Agent with Reasoning and Conversational Capabilities

This agent provides natural language interaction, multi-step reasoning,
and tool use for comprehensive chemistry analysis.
"""

import json
import re
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
import logging

from .base_agent import BaseChemistryAgent, AgentConfig, PredictionResult
from .property_prediction_agent import PropertyPredictionAgent
from .solubility_agent import SolubilityAgent
from .toxicity_agent import ToxicityAgent
from .unit_operations_agent import UnitOperationsAgent

@dataclass
class ConversationMessage:
    """Message in conversation history"""
    role: str  # 'user', 'assistant', 'system', 'tool'
    content: str
    tool_calls: Optional[List[Dict]] = None
    tool_results: Optional[List[Dict]] = None
    timestamp: Optional[str] = None

@dataclass
class ReasoningStep:
    """Step in multi-step reasoning process"""
    step_number: int
    description: str
    action: str
    tool_used: Optional[str] = None
    result: Optional[Any] = None
    confidence: Optional[float] = None

class ChemistryLLMAgent(BaseChemistryAgent):
    """
    Advanced LLM-powered chemistry agent with reasoning capabilities

    Features:
    - Natural language conversation about chemistry
    - Multi-step problem solving and planning
    - Tool use (property prediction, unit operations, etc.)
    - Reasoning and explanation of results
    - Chemistry knowledge integration
    """

    def __init__(self, config: Optional[AgentConfig] = None, llm_provider: str = "openai"):
        super().__init__(config)
        self.llm_provider = llm_provider
        self.conversation_history: List[ConversationMessage] = []
        self.reasoning_trace: List[ReasoningStep] = []

        # Initialize specialized agents as tools
        self.tools = {
            "property_predictor": PropertyPredictionAgent(config),
            "solubility_agent": SolubilityAgent(config),
            "toxicity_agent": ToxicityAgent(config),
            "unit_operations": UnitOperationsAgent(config)
        }

        # Chemistry knowledge base
        self.chemistry_knowledge = self._load_chemistry_knowledge()

        self.logger.info("Initialized Chemistry LLM Agent with reasoning capabilities")

    def _load_chemistry_knowledge(self) -> Dict[str, Any]:
        """Load chemistry knowledge base for reasoning"""
        return {
            "functional_groups": {
                "alcohol": {"smarts": "[OH]", "properties": ["polar", "hydrogen_bonding"]},
                "carboxylic_acid": {"smarts": "[CX3](=O)[OX2H1]", "properties": ["acidic", "polar"]},
                "amine": {"smarts": "[NX3;H2,H1;!$(NC=O)]", "properties": ["basic", "hydrogen_bonding"]},
                "aromatic": {"smarts": "c", "properties": ["hydrophobic", "pi_stacking"]}
            },
            "solubility_rules": [
                "Alcohols with fewer than 4 carbons are generally water soluble",
                "Aromatic compounds are typically hydrophobic",
                "Carboxylic acids are polar and often water soluble",
                "Long alkyl chains reduce water solubility"
            ],
            "toxicity_patterns": [
                "Benzene rings can indicate potential toxicity",
                "Heavy metals are typically toxic",
                "Small polar molecules often have lower toxicity",
                "Reactive functional groups may increase toxicity"
            ]
        }

    def chat(self, user_message: str) -> str:
        """
        Main conversational interface for chemistry queries
        """
        # Add user message to history
        self.conversation_history.append(
            ConversationMessage(role="user", content=user_message)
        )

        # Analyze the query and plan response
        response = self._process_chemistry_query(user_message)

        # Add assistant response to history
        self.conversation_history.append(
            ConversationMessage(role="assistant", content=response)
        )

        return response

    def _process_chemistry_query(self, query: str) -> str:
        """
        Process chemistry query with multi-step reasoning
        """
        self.reasoning_trace = []

        # Step 1: Understand the query
        query_analysis = self._analyze_query(query)
        self._add_reasoning_step(
            "Analyzing user query",
            "query_analysis",
            result=query_analysis
        )

        # Step 2: Extract molecules if present
        molecules = self._extract_molecules(query)
        if molecules:
            self._add_reasoning_step(
                f"Identified {len(molecules)} molecules: {', '.join(molecules)}",
                "molecule_extraction",
                result=molecules
            )

        # Step 3: Determine what analysis to perform
        analysis_plan = self._create_analysis_plan(query_analysis, molecules)
        self._add_reasoning_step(
            "Creating analysis plan",
            "planning",
            result=analysis_plan
        )

        # Step 4: Execute analysis
        results = self._execute_analysis_plan(analysis_plan, molecules)

        # Step 5: Generate comprehensive response
        response = self._generate_response(query_analysis, results)

        return response

    def _analyze_query(self, query: str) -> Dict[str, Any]:
        """Analyze what the user is asking for"""
        query_lower = query.lower()

        analysis = {
            "query_type": "general",
            "requested_properties": [],
            "analysis_needed": [],
            "complexity": "simple"
        }

        # Identify query type
        if any(word in query_lower for word in ["solubility", "soluble", "dissolve"]):
            analysis["query_type"] = "solubility"
            analysis["requested_properties"].append("solubility")
            analysis["analysis_needed"].append("solubility_prediction")

        if any(word in query_lower for word in ["toxic", "toxicity", "safe", "harmful"]):
            analysis["query_type"] = "toxicity"
            analysis["requested_properties"].append("toxicity")
            analysis["analysis_needed"].append("toxicity_assessment")

        if any(word in query_lower for word in ["predict", "property", "properties"]):
            analysis["query_type"] = "property_prediction"
            analysis["analysis_needed"].append("property_prediction")

        if any(word in query_lower for word in ["distillation", "reactor", "heat exchanger", "process"]):
            analysis["query_type"] = "unit_operations"
            analysis["analysis_needed"].append("unit_operations")

        if any(word in query_lower for word in ["compare", "difference", "vs", "versus"]):
            analysis["complexity"] = "comparative"

        if any(word in query_lower for word in ["explain", "why", "how", "mechanism"]):
            analysis["complexity"] = "explanatory"
            analysis["analysis_needed"].append("explanation")

        return analysis

    def _extract_molecules(self, query: str) -> List[str]:
        """Extract SMILES or chemical names from query"""
        molecules = []

        # Look for common chemical names first (to avoid parsing errors)
        chemical_names = {
            "ethanol": "CCO",
            "methanol": "CO",
            "benzene": "c1ccccc1",
            "water": "O",
            "acetic acid": "CC(=O)O",
            "acetone": "CC(=O)C",
            "ibuprofen": "CC(C)Cc1ccc(C(C)C(=O)O)cc1",
            "aspirin": "CC(=O)OC1=CC=CC=C1C(=O)O",
            "caffeine": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
            "phenol": "c1ccc(cc1)O",
            "toluene": "Cc1ccccc1"
        }

        query_lower = query.lower()
        for name, smiles in chemical_names.items():
            if name in query_lower:
                molecules.append(smiles)

        # Look for likely SMILES patterns (more selective)
        # Only check tokens that look like SMILES
        tokens = query.split()
        for token in tokens:
            # Only check tokens that could be SMILES
            if self._looks_like_smiles(token):
                if self._is_valid_smiles(token):
                    molecules.append(token)

        return list(set(molecules))  # Remove duplicates

    def _looks_like_smiles(self, token: str) -> bool:
        """Check if a token looks like it could be SMILES before validating"""
        # Quick heuristics to avoid validating obviously non-SMILES strings
        if len(token) < 2:
            return False

        # Must contain typical SMILES characters
        smiles_chars = set('CONSPBFIcnops123456789()[]=#@+-.')
        if not any(c in smiles_chars for c in token):
            return False

        # Avoid common English words that might have SMILES-like patterns
        common_words = {
            'what', 'is', 'the', 'of', 'and', 'or', 'in', 'on', 'at', 'to', 'for',
            'with', 'by', 'from', 'are', 'can', 'has', 'have', 'does', 'will',
            'toxic', 'toxicity', 'soluble', 'solubility', 'polar', 'molecule',
            'compound', 'chemical', 'structure', 'property', 'analysis', 'compare',
            'why', 'how', 'makes', 'between', 'difference', 'explain', 'about'
        }

        if token.lower() in common_words:
            return False

        # If it passes these checks, it might be SMILES
        return True

    def _is_valid_smiles(self, smiles: str) -> bool:
        """Basic SMILES validation with suppressed error messages"""
        try:
            from rdkit import Chem
            from rdkit import RDLogger

            # Suppress RDKit error messages temporarily
            lg = RDLogger.logger()
            lg.setLevel(RDLogger.CRITICAL)

            mol = Chem.MolFromSmiles(smiles)

            # Restore normal logging
            lg.setLevel(RDLogger.WARNING)

            return mol is not None
        except:
            # Simple heuristic if RDKit not available
            return len(smiles) > 1 and any(c in smiles for c in "CON()=[]")

    def _create_analysis_plan(self, query_analysis: Dict, molecules: List[str]) -> Dict[str, Any]:
        """Create step-by-step analysis plan"""
        plan = {
            "steps": [],
            "tools_needed": [],
            "expected_outputs": []
        }

        if molecules:
            if "solubility_prediction" in query_analysis["analysis_needed"]:
                plan["steps"].append("predict_solubility")
                plan["tools_needed"].append("solubility_agent")
                plan["expected_outputs"].append("solubility_values")

            if "toxicity_assessment" in query_analysis["analysis_needed"]:
                plan["steps"].append("assess_toxicity")
                plan["tools_needed"].append("toxicity_agent")
                plan["expected_outputs"].append("toxicity_scores")

            if "property_prediction" in query_analysis["analysis_needed"]:
                plan["steps"].append("predict_properties")
                plan["tools_needed"].append("property_predictor")
                plan["expected_outputs"].append("property_values")

        if "unit_operations" in query_analysis["analysis_needed"]:
            plan["steps"].append("unit_operations_analysis")
            plan["tools_needed"].append("unit_operations")
            plan["expected_outputs"].append("engineering_calculations")

        if "explanation" in query_analysis["analysis_needed"]:
            plan["steps"].append("provide_explanation")
            plan["tools_needed"].append("chemistry_knowledge")
            plan["expected_outputs"].append("detailed_explanation")

        return plan

    def _execute_analysis_plan(self, plan: Dict, molecules: List[str]) -> Dict[str, Any]:
        """Execute the analysis plan using available tools"""
        results = {}

        for step in plan["steps"]:
            try:
                if step == "predict_solubility" and molecules:
                    self._add_reasoning_step(
                        "Predicting solubility using ChemBERTa model",
                        "tool_use",
                        tool_used="solubility_agent"
                    )

                    if not self.tools["solubility_agent"].is_loaded:
                        self.tools["solubility_agent"].load_model()

                    solubility_results = []
                    for mol in molecules:
                        result = self.tools["solubility_agent"].predict_single(mol)
                        solubility_results.append(result)

                    results["solubility"] = solubility_results

                elif step == "assess_toxicity" and molecules:
                    self._add_reasoning_step(
                        "Assessing toxicity using AI models",
                        "tool_use",
                        tool_used="toxicity_agent"
                    )

                    if not self.tools["toxicity_agent"].is_loaded:
                        self.tools["toxicity_agent"].load_model()

                    toxicity_results = []
                    for mol in molecules:
                        result = self.tools["toxicity_agent"].predict_single(mol)
                        toxicity_results.append(result)

                    results["toxicity"] = toxicity_results

                elif step == "predict_properties" and molecules:
                    self._add_reasoning_step(
                        "Predicting molecular properties",
                        "tool_use",
                        tool_used="property_predictor"
                    )

                    if not self.tools["property_predictor"].is_loaded:
                        self.tools["property_predictor"].load_model()

                    property_results = []
                    for mol in molecules:
                        result = self.tools["property_predictor"].predict_single(mol)
                        property_results.append(result)

                    results["properties"] = property_results

                elif step == "provide_explanation":
                    explanation = self._generate_chemistry_explanation(molecules, results)
                    results["explanation"] = explanation

            except Exception as e:
                self.logger.error(f"Error in step {step}: {e}")
                results[f"{step}_error"] = str(e)

        return results

    def _generate_chemistry_explanation(self, molecules: List[str], results: Dict) -> str:
        """Generate chemistry-based explanation of results"""
        explanations = []

        for i, mol in enumerate(molecules):
            mol_explanation = f"\n**Analysis of molecule {mol}:**\n"

            # Structural analysis
            mol_explanation += self._analyze_molecular_structure(mol)

            # Results interpretation
            if "solubility" in results and i < len(results["solubility"]):
                sol_result = results["solubility"][i]
                mol_explanation += f"\n- **Solubility**: {sol_result.prediction:.2f} logS"
                mol_explanation += f"\n  - {self._interpret_solubility(sol_result.prediction)}"

            if "toxicity" in results and i < len(results["toxicity"]):
                tox_result = results["toxicity"][i]
                mol_explanation += f"\n- **Toxicity**: {tox_result.prediction:.2f}"
                mol_explanation += f"\n  - {self._interpret_toxicity(tox_result.prediction)}"

            explanations.append(mol_explanation)

        return "\n".join(explanations)

    def _analyze_molecular_structure(self, smiles: str) -> str:
        """Analyze molecular structure for explanation"""
        analysis = f"- **Structure**: {smiles}\n"

        # Simple structural analysis
        if "c1ccccc1" in smiles or "c" in smiles:
            analysis += "- Contains aromatic ring(s) - typically hydrophobic\n"

        if "O" in smiles and "H" in smiles:
            analysis += "- Contains hydroxyl group(s) - increases polarity\n"

        if "N" in smiles:
            analysis += "- Contains nitrogen - may be basic\n"

        if "C(=O)" in smiles:
            analysis += "- Contains carbonyl group - polar functional group\n"

        return analysis

    def _interpret_solubility(self, logS: float) -> str:
        """Interpret solubility prediction"""
        if logS > -1:
            return "Highly soluble in water"
        elif logS > -3:
            return "Moderately soluble in water"
        elif logS > -5:
            return "Poorly soluble in water"
        else:
            return "Very poorly soluble in water"

    def _interpret_toxicity(self, toxicity_score: float) -> str:
        """Interpret toxicity prediction"""
        if toxicity_score < 0.3:
            return "Low toxicity expected"
        elif toxicity_score < 0.7:
            return "Moderate toxicity - caution advised"
        else:
            return "High toxicity - significant safety concerns"

    def _add_reasoning_step(self, description: str, action: str,
                          tool_used: Optional[str] = None,
                          result: Optional[Any] = None,
                          confidence: Optional[float] = None):
        """Add step to reasoning trace"""
        step = ReasoningStep(
            step_number=len(self.reasoning_trace) + 1,
            description=description,
            action=action,
            tool_used=tool_used,
            result=result,
            confidence=confidence
        )
        self.reasoning_trace.append(step)
        self.logger.info(f"Reasoning Step {step.step_number}: {description}")

    def _generate_response(self, query_analysis: Dict, results: Dict) -> str:
        """Generate comprehensive response based on analysis"""
        response = ""

        # Add greeting and understanding
        response += "I'll analyze this chemistry question step by step.\n\n"

        # Add reasoning trace
        if self.reasoning_trace:
            response += "**My reasoning process:**\n"
            for step in self.reasoning_trace:
                response += f"{step.step_number}. {step.description}\n"
            response += "\n"

        # Add results
        if "solubility" in results:
            response += "**Solubility Analysis:**\n"
            for result in results["solubility"]:
                response += f"- {result.smiles}: {result.prediction:.2f} logS ({self._interpret_solubility(result.prediction)})\n"
            response += "\n"

        if "toxicity" in results:
            response += "**Toxicity Assessment:**\n"
            for result in results["toxicity"]:
                response += f"- {result.smiles}: {result.prediction:.2f} ({self._interpret_toxicity(result.prediction)})\n"
            response += "\n"

        if "explanation" in results:
            response += "**Detailed Chemical Analysis:**\n"
            response += results["explanation"]
            response += "\n"

        # Add chemistry insights
        response += self._add_chemistry_insights(query_analysis, results)

        return response

    def _add_chemistry_insights(self, query_analysis: Dict, results: Dict) -> str:
        """Add chemistry-specific insights and recommendations"""
        insights = "\n**Chemistry Insights:**\n"

        if query_analysis["query_type"] == "solubility":
            insights += "- Solubility depends on molecular polarity and hydrogen bonding\n"
            insights += "- Aromatic rings generally decrease water solubility\n"
            insights += "- Hydroxyl and carboxyl groups increase water solubility\n"

        elif query_analysis["query_type"] == "toxicity":
            insights += "- Toxicity predictions should be validated experimentally\n"
            insights += "- Consider dose-response relationships\n"
            insights += "- Structural alerts can indicate potential toxicity mechanisms\n"

        insights += "\n**Recommendations:**\n"
        insights += "- Always validate computational predictions with experimental data\n"
        insights += "- Consider multiple properties when designing molecules\n"
        insights += "- Use these predictions as starting points for further investigation\n"

        return insights

    def get_reasoning_trace(self) -> List[ReasoningStep]:
        """Get the reasoning trace for the last query"""
        return self.reasoning_trace

    def get_conversation_history(self) -> List[ConversationMessage]:
        """Get full conversation history"""
        return self.conversation_history

    def reset_conversation(self):
        """Reset conversation history"""
        self.conversation_history = []
        self.reasoning_trace = []
        self.logger.info("Conversation reset")

    def load_model(self, model_path: Optional[str] = None) -> None:
        """
        Load the LLM agent (no specific model to load, just mark as ready)
        """
        self.is_loaded = True
        self.logger.info("Chemistry LLM Agent loaded and ready for conversation")

    def predict_single(self, smiles: str) -> PredictionResult:
        """
        LLM agent doesn't do direct predictions - it orchestrates other agents
        This method provides a simple interface for basic queries
        """
        query = f"Analyze the molecule {smiles} and provide properties"
        response = self.chat(query)

        return PredictionResult(
            smiles=smiles,
            prediction=0.0,  # LLM agent doesn't return numeric predictions
            confidence=1.0,
            additional_info={"llm_response": response}
        )

    def explain_predictions(self, molecules: List[str]) -> str:
        """Provide detailed explanations for predictions"""
        explanations = []

        for mol in molecules:
            explanation = f"\n**Molecular Analysis for {mol}:**\n"

            # Get predictions from all agents
            try:
                if not self.tools["solubility_agent"].is_loaded:
                    self.tools["solubility_agent"].load_model()
                sol_result = self.tools["solubility_agent"].predict_single(mol)

                explanation += f"**Solubility Prediction:** {sol_result.prediction:.2f} logS\n"
                explanation += f"- {self._interpret_solubility(sol_result.prediction)}\n"
                explanation += f"- Confidence: {sol_result.confidence:.2f}\n\n"

            except Exception as e:
                explanation += f"Solubility prediction failed: {e}\n\n"

            # Add structural analysis
            explanation += self._analyze_molecular_structure(mol)

            # Add chemistry insights
            explanation += "\n**Chemical Reasoning:**\n"
            explanation += self._get_structure_property_relationships(mol)

            explanations.append(explanation)

        return "\n".join(explanations)

    def _get_structure_property_relationships(self, smiles: str) -> str:
        """Explain structure-property relationships"""
        relationships = ""

        # Analyze functional groups and their effects
        if "OH" in smiles:
            relationships += "- Hydroxyl groups increase polarity and hydrogen bonding potential\n"
            relationships += "- This typically increases water solubility\n"

        if "c1ccccc1" in smiles:
            relationships += "- Aromatic rings contribute to hydrophobic character\n"
            relationships += "- May reduce water solubility but increase membrane permeability\n"

        if "C(=O)O" in smiles:
            relationships += "- Carboxylic acid groups are highly polar and ionizable\n"
            relationships += "- Typically increase water solubility at physiological pH\n"

        if "N" in smiles:
            relationships += "- Nitrogen atoms can form hydrogen bonds\n"
            relationships += "- May be protonated at physiological pH, affecting properties\n"

        return relationships