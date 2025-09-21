"""
ChemBERTa Conversational Agent

Combines ChemBERTa's molecular intelligence with conversational AI capabilities.
This agent uses ChemBERTa for what it does BEST (molecular tasks) and adds
human-friendly conversation on top.

Best of both worlds: ChemBERTa's molecular brain + conversational interface!
"""

from typing import Dict, List, Optional, Any, Tuple
import numpy as np
import logging

from .chemberta_agent import ChemBERTaAgent, ChemBERTaResult
from .base_agent import AgentConfig, PredictionResult
from .chemistry_llm_agent import ConversationMessage, ReasoningStep

class ChemBERTaConversationalAgent(ChemBERTaAgent):
    """
    Conversational agent powered by ChemBERTa's molecular intelligence

    What makes this special:
    - Uses ChemBERTa for molecular understanding (better than GPT!)
    - Adds conversational interface for human interaction
    - Explains ChemBERTa insights in plain English
    - Combines molecular intelligence with chemical reasoning
    """

    def __init__(self, config: Optional[AgentConfig] = None):
        super().__init__(config)
        self.conversation_history: List[ConversationMessage] = []
        self.reasoning_trace: List[ReasoningStep] = []

        # Conversation patterns for ChemBERTa tasks
        self.conversation_patterns = {
            "similarity": ["similar", "like", "resembles", "compare", "related"],
            "clustering": ["cluster", "group", "categorize", "organize", "classify"],
            "space_analysis": ["diversity", "space", "distribution", "variety", "range"],
            "embedding": ["representation", "encoding", "features", "dimensions"],
            "search": ["find", "search", "discover", "identify", "locate"]
        }

        self.logger.info("Initialized ChemBERTa Conversational Agent")

    def chat(self, user_message: str) -> str:
        """
        Main conversational interface for ChemBERTa-powered chemistry
        """
        # Add to conversation history
        self.conversation_history.append(
            ConversationMessage(role="user", content=user_message)
        )

        # Reset reasoning trace
        self.reasoning_trace = []

        # Process the query
        response = self._process_chemberta_query(user_message)

        # Add response to history
        self.conversation_history.append(
            ConversationMessage(role="assistant", content=response)
        )

        return response

    def _process_chemberta_query(self, query: str) -> str:
        """Process query using ChemBERTa intelligence"""

        # Step 1: Analyze what the user wants
        query_analysis = self._analyze_chemberta_query(query)
        self._add_reasoning_step(
            f"Analyzing query: identified {query_analysis['task_type']} task",
            "query_analysis",
            result=query_analysis
        )

        # Step 2: Extract molecules
        molecules = self._extract_molecules_from_query(query)
        if molecules:
            self._add_reasoning_step(
                f"Found {len(molecules)} molecules: {', '.join(molecules)}",
                "molecule_extraction",
                result=molecules
            )

        # Step 3: Execute ChemBERTa analysis
        chemberta_results = self._execute_chemberta_analysis(query_analysis, molecules)
        if chemberta_results:
            self._add_reasoning_step(
                f"ChemBERTa analysis completed",
                "chemberta_analysis",
                tool_used="ChemBERTa",
                result="Molecular intelligence applied"
            )

        # Step 4: Generate conversational response
        response = self._generate_conversational_response(query_analysis, molecules, chemberta_results)

        return response

    def _analyze_chemberta_query(self, query: str) -> Dict[str, Any]:
        """Analyze what ChemBERTa task the user wants"""
        query_lower = query.lower()

        analysis = {
            "task_type": "general",
            "chemberta_task": None,
            "requires_model": False,
            "molecules_needed": True
        }

        # Similarity tasks
        if any(word in query_lower for word in self.conversation_patterns["similarity"]):
            analysis["task_type"] = "similarity"
            analysis["chemberta_task"] = "molecular_similarity"
            analysis["requires_model"] = True

        # Clustering tasks
        elif any(word in query_lower for word in self.conversation_patterns["clustering"]):
            analysis["task_type"] = "clustering"
            analysis["chemberta_task"] = "molecular_clustering"
            analysis["requires_model"] = True

        # Chemical space analysis
        elif any(word in query_lower for word in self.conversation_patterns["space_analysis"]):
            analysis["task_type"] = "space_analysis"
            analysis["chemberta_task"] = "chemical_space_analysis"
            analysis["requires_model"] = True

        # Search tasks
        elif any(word in query_lower for word in self.conversation_patterns["search"]):
            analysis["task_type"] = "search"
            analysis["chemberta_task"] = "similarity_search"
            analysis["requires_model"] = True

        # Embedding/representation tasks
        elif any(word in query_lower for word in self.conversation_patterns["embedding"]):
            analysis["task_type"] = "embedding"
            analysis["chemberta_task"] = "molecular_embedding"
            analysis["requires_model"] = True

        # General molecular questions
        elif any(mol_word in query_lower for mol_word in ["molecule", "compound", "chemical", "smiles"]):
            analysis["task_type"] = "molecular_analysis"
            analysis["chemberta_task"] = "general_analysis"
            analysis["requires_model"] = True

        return analysis

    def _extract_molecules_from_query(self, query: str) -> List[str]:
        """Extract molecules from query (reuse from parent)"""
        # Use the same molecule extraction as ChemBERTa agent
        return self._extract_molecules_simple(query)

    def _extract_molecules_simple(self, query: str) -> List[str]:
        """Simple molecule extraction"""
        molecules = []

        # Common molecules
        common_molecules = {
            "ethanol": "CCO",
            "methanol": "CO",
            "benzene": "c1ccccc1",
            "toluene": "Cc1ccccc1",
            "phenol": "c1ccc(cc1)O",
            "aspirin": "CC(=O)OC1=CC=CC=C1C(=O)O",
            "ibuprofen": "CC(C)Cc1ccc(C(C)C(=O)O)cc1",
            "caffeine": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"
        }

        query_lower = query.lower()
        for name, smiles in common_molecules.items():
            if name in query_lower:
                molecules.append(smiles)

        # Look for SMILES-like patterns
        import re
        tokens = query.split()
        for token in tokens:
            if self._looks_like_smiles_simple(token):
                molecules.append(token)

        return list(set(molecules))

    def _looks_like_smiles_simple(self, token: str) -> bool:
        """Simple SMILES detection"""
        if len(token) < 3:
            return False
        return any(c in token for c in "CONSPc123456789()[]=#")

    def _execute_chemberta_analysis(self, query_analysis: Dict, molecules: List[str]) -> Dict[str, Any]:
        """Execute the appropriate ChemBERTa analysis"""
        if not self.is_loaded:
            return {"error": "ChemBERTa model not loaded"}

        if not molecules:
            return {"error": "No molecules found in query"}

        task = query_analysis.get("chemberta_task")
        results = {}

        try:
            if task == "molecular_similarity" and len(molecules) >= 2:
                # Calculate similarities between all pairs
                similarities = {}
                for i, mol1 in enumerate(molecules):
                    for j, mol2 in enumerate(molecules[i+1:], i+1):
                        sim = self.calculate_molecular_similarity(mol1, mol2)
                        similarities[f"{mol1}_vs_{mol2}"] = sim

                results["similarities"] = similarities

            elif task == "molecular_clustering":
                if len(molecules) >= 3:
                    clusters = self.cluster_molecules(molecules)
                    results["clusters"] = clusters
                else:
                    results["error"] = "Need at least 3 molecules for clustering"

            elif task == "chemical_space_analysis":
                space_analysis = self.analyze_chemical_space(molecules)
                results["space_analysis"] = space_analysis

            elif task == "similarity_search":
                if len(molecules) >= 2:
                    target = molecules[0]
                    candidates = molecules[1:]
                    similar = self.find_similar_molecules(target, candidates)
                    results["similar_molecules"] = similar

            elif task == "molecular_embedding":
                embeddings = {}
                for mol in molecules:
                    emb = self.get_molecular_embedding(mol)
                    embeddings[mol] = {
                        "dimension": len(emb),
                        "magnitude": float(np.linalg.norm(emb)),
                        "sample_values": emb[:5].tolist()
                    }
                results["embeddings"] = embeddings

            elif task == "general_analysis":
                # General ChemBERTa analysis
                for mol in molecules:
                    emb = self.get_molecular_embedding(mol)
                    results[mol] = {
                        "embedding_dimension": len(emb),
                        "embedding_magnitude": float(np.linalg.norm(emb)),
                        "chemberta_encoded": True
                    }

        except Exception as e:
            results["error"] = str(e)

        return results

    def _generate_conversational_response(self, query_analysis: Dict,
                                        molecules: List[str],
                                        chemberta_results: Dict) -> str:
        """Generate human-friendly response based on ChemBERTa analysis"""

        response = "I'll analyze this using ChemBERTa's molecular intelligence.\n\n"

        # Add reasoning process
        response += "**My ChemBERTa Analysis Process:**\n"
        for step in self.reasoning_trace:
            response += f"{step.step_number}. {step.description}\n"
        response += "\n"

        # Add ChemBERTa results interpretation
        if "error" in chemberta_results:
            response += f"Analysis failed: {chemberta_results['error']}\n"
            response += "Make sure ChemBERTa model is loaded and molecules are valid.\n"
            return response

        task_type = query_analysis["task_type"]

        if task_type == "similarity" and "similarities" in chemberta_results:
            response += "**ChemBERTa Molecular Similarity Analysis:**\n"
            similarities = chemberta_results["similarities"]

            if similarities:
                response += "ChemBERTa found these molecular similarities:\n"
                for pair, sim in similarities.items():
                    mol1, mol2 = pair.split("_vs_")
                    response += f"- {mol1} vs {mol2}: {sim:.3f} similarity\n"

                # Interpretation
                avg_sim = np.mean(list(similarities.values()))
                if avg_sim > 0.8:
                    response += "\n**ChemBERTa Insight**: These molecules are very similar chemically!\n"
                elif avg_sim > 0.5:
                    response += "\n**ChemBERTa Insight**: Moderate chemical similarity detected.\n"
                else:
                    response += "\n**ChemBERTa Insight**: These molecules are quite different chemically.\n"

        elif task_type == "clustering" and "clusters" in chemberta_results:
            response += "**ChemBERTa Molecular Clustering:**\n"
            clusters = chemberta_results["clusters"]

            response += f"ChemBERTa grouped your molecules into {len(clusters)} clusters:\n"
            for cluster_id, cluster_mols in clusters.items():
                response += f"- Cluster {cluster_id}: {cluster_mols}\n"

            response += "\n**ChemBERTa Insight**: Molecules in the same cluster share similar chemical features!\n"

        elif task_type == "space_analysis" and "space_analysis" in chemberta_results:
            response += "**ChemBERTa Chemical Space Analysis:**\n"
            analysis = chemberta_results["space_analysis"]

            if "diversity_metrics" in analysis:
                diversity = analysis["diversity_metrics"]
                response += f"- Chemical diversity score: {diversity['diversity_score']:.3f}\n"
                response += f"- Average molecular similarity: {diversity['mean_similarity']:.3f}\n"

                if diversity['diversity_score'] > 0.7:
                    response += "\n**ChemBERTa Insight**: Highly diverse chemical set!\n"
                else:
                    response += "\n**ChemBERTa Insight**: Molecules share similar chemical features.\n"

        elif "embeddings" in chemberta_results:
            response += "**ChemBERTa Molecular Embeddings:**\n"
            embeddings = chemberta_results["embeddings"]

            for mol, emb_info in embeddings.items():
                response += f"- {mol}: {emb_info['dimension']}D vector (magnitude: {emb_info['magnitude']:.3f})\n"

            response += "\n**ChemBERTa Insight**: Each molecule is encoded as a 384-dimensional vector capturing its chemical essence!\n"

        # Add ChemBERTa advantages
        response += "\n**Why ChemBERTa Excels at This:**\n"
        response += "- Trained on 77 MILLION molecules (not just text like GPT)\n"
        response += "- Understands molecular structure and chemical patterns\n"
        response += "- Calculates true chemical similarity, not just structural\n"
        response += "- Discovers hidden relationships humans might miss\n"

        return response

    def _add_reasoning_step(self, description: str, action: str,
                          tool_used: Optional[str] = None,
                          result: Optional[Any] = None):
        """Add reasoning step"""
        step = ReasoningStep(
            step_number=len(self.reasoning_trace) + 1,
            description=description,
            action=action,
            tool_used=tool_used,
            result=result
        )
        self.reasoning_trace.append(step)

    def get_reasoning_trace(self) -> List[ReasoningStep]:
        """Get reasoning trace"""
        return self.reasoning_trace

    def get_conversation_history(self) -> List[ConversationMessage]:
        """Get conversation history"""
        return self.conversation_history

    def reset_conversation(self):
        """Reset conversation"""
        self.conversation_history = []
        self.reasoning_trace = []

    def explain_capabilities(self) -> str:
        """Explain what this agent can do"""
        return """
**ChemBERTa Conversational Agent Capabilities:**

**What I Excel At (Better than GPT!):**
- "Find molecules similar to aspirin" - Uses ChemBERTa molecular similarity
- "Cluster these molecules: CCO, c1ccccc1, CCN" - ChemBERTa clustering
- "How diverse are these compounds?" - Chemical space analysis
- "What's the molecular representation of benzene?" - ChemBERTa embeddings

**Example Conversations:**
- "Which drugs are most similar to ibuprofen?"
- "Group these molecules by chemical similarity"
- "How chemically diverse is this compound library?"
- "Find the molecular embedding for caffeine"
- "Compare the chemical similarity of benzene and toluene"

**Why I'm Special:**
- ChemBERTa molecular brain (77M molecules trained)
- Conversational interface (human-friendly)
- Explains complex molecular insights in plain English
- Combines AI molecular intelligence with chemical reasoning

**Perfect For:**
- Drug discovery and similarity searching
- Chemical library analysis and clustering
- Molecular dataset exploration
- Structure-activity relationship discovery
        """

def create_chemberta_examples() -> List[str]:
    """Create example queries for ChemBERTa agent"""
    return [
        "Find molecules similar to aspirin",
        "Cluster these molecules: CCO, c1ccccc1, CC(=O)O, CCN",
        "How diverse are benzene, toluene, and phenol?",
        "What's the molecular representation of caffeine?",
        "Compare the similarity between ethanol and methanol",
        "Group these drugs by chemical similarity: aspirin, ibuprofen, acetaminophen",
        "Analyze the chemical space of these compounds",
        "Find the ChemBERTa embedding for morphine"
    ]