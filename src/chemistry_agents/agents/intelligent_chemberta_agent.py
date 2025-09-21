"""
Intelligent ChemBERTa Agent with Task-Specific Model Routing

This agent automatically determines the task type from queries and uses
the appropriate specialized ChemBERTa model trained on specific datasets:
- Solubility queries → Solubility-trained ChemBERTa
- Toxicity queries → Toxicity-trained ChemBERTa
- Bioactivity queries → Bioactivity-trained ChemBERTa
- General queries → Base ChemBERTa
"""

import os
import json
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
import logging
from transformers import AutoTokenizer, AutoModel

from .chemberta_conversational_agent import ChemBERTaConversationalAgent
from .base_agent import AgentConfig, PredictionResult

class ChemBERTaRegressor(nn.Module):
    """ChemBERTa with regression head for property prediction"""

    def __init__(self, model_name: str = "DeepChem/ChemBERTa-77M-MLM", dropout: float = 0.3):
        super().__init__()
        self.chemberta = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)
        self.regressor = nn.Linear(self.chemberta.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.chemberta(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        output = self.dropout(pooled_output)
        return self.regressor(output)

class IntelligentChemBERTaAgent(ChemBERTaConversationalAgent):
    """
    Intelligent ChemBERTa agent that routes queries to task-specific models

    Features:
    - Automatic task detection from natural language queries
    - Specialized models for solubility, toxicity, bioactivity
    - Improved accuracy on specific prediction tasks
    - Seamless integration with conversational interface
    """

    def __init__(self, config: Optional[AgentConfig] = None):
        super().__init__(config)

        # Task-specific models and metadata
        self.specialized_models = {}
        self.task_metadata = {}
        self.specialized_tokenizer = None

        # Task detection patterns
        self.task_patterns = {
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

        self.models_dir = "models/task_specific_chemberta"
        self.logger.info("Initialized Intelligent ChemBERTa Agent with task routing")

    def load_specialized_models(self) -> bool:
        """Load all available task-specific models"""

        if not os.path.exists(self.models_dir):
            self.logger.warning(f"Specialized models directory not found: {self.models_dir}")
            self.logger.info("Run train_task_specific_chemberta.py to create specialized models")
            return False

        try:
            # Load tokenizer
            tokenizer_path = os.path.join(self.models_dir, "tokenizer")
            if os.path.exists(tokenizer_path):
                self.specialized_tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
                self.logger.info("Loaded specialized tokenizer")

            # Load each task-specific model
            for task_type in ["solubility", "toxicity", "bioactivity"]:
                model_path = os.path.join(self.models_dir, f"chemberta_{task_type}.pt")
                info_path = os.path.join(self.models_dir, f"chemberta_{task_type}_info.json")

                if os.path.exists(model_path) and os.path.exists(info_path):
                    # Load model
                    model = ChemBERTaRegressor()
                    model.load_state_dict(torch.load(model_path, map_location=self.config.device))
                    model.eval()
                    model.to(self.config.device)
                    self.specialized_models[task_type] = model

                    # Load metadata
                    with open(info_path, 'r') as f:
                        self.task_metadata[task_type] = json.load(f)

                    self.logger.info(f"✓ Loaded {task_type} specialized model")
                else:
                    self.logger.warning(f"Missing files for {task_type} model")

            if self.specialized_models:
                self.logger.info(f"Loaded {len(self.specialized_models)} specialized models: {list(self.specialized_models.keys())}")
                return True
            else:
                self.logger.warning("No specialized models loaded")
                return False

        except Exception as e:
            self.logger.error(f"Failed to load specialized models: {e}")
            return False

    def detect_task_type(self, query: str) -> Tuple[str, float]:
        """
        Detect the most likely task type from the query

        Returns:
            (task_type, confidence_score)
        """

        query_lower = query.lower()
        task_scores = {}

        # Score each task type based on keyword matches
        for task_type, patterns in self.task_patterns.items():
            score = 0
            for pattern in patterns:
                if pattern in query_lower:
                    # Weight longer patterns more heavily
                    score += len(pattern.split())

            task_scores[task_type] = score

        # Find highest scoring task
        if not any(task_scores.values()):
            return "general", 0.0

        best_task = max(task_scores, key=task_scores.get)
        max_score = task_scores[best_task]
        confidence = min(max_score / 3.0, 1.0)  # Normalize confidence

        return best_task, confidence

    def predict_with_specialized_model(self, task_type: str, smiles: str) -> Dict[str, Any]:
        """Make prediction using task-specific model"""

        if task_type not in self.specialized_models:
            raise ValueError(f"No specialized model available for task: {task_type}")

        if not self.specialized_tokenizer:
            raise ValueError("Specialized tokenizer not loaded")

        try:
            model = self.specialized_models[task_type]

            # Tokenize SMILES
            encoding = self.specialized_tokenizer(
                smiles,
                truncation=True,
                padding='max_length',
                max_length=512,
                return_tensors='pt'
            )

            input_ids = encoding['input_ids'].to(self.config.device)
            attention_mask = encoding['attention_mask'].to(self.config.device)

            # Make prediction
            with torch.no_grad():
                output = model(input_ids, attention_mask)
                prediction = output.item()

            # Get task metadata for interpretation
            metadata = self.task_metadata.get(task_type, {})

            # Interpret prediction based on task
            interpretation = self._interpret_prediction(task_type, prediction, metadata)

            return {
                "prediction": prediction,
                "task_type": task_type,
                "model_used": f"specialized_{task_type}",
                "interpretation": interpretation,
                "metadata": metadata
            }

        except Exception as e:
            self.logger.error(f"Prediction failed for {task_type}: {e}")
            raise

    def _interpret_prediction(self, task_type: str, prediction: float, metadata: Dict) -> str:
        """Interpret numerical prediction based on task type"""

        if task_type == "solubility":
            if prediction > -1:
                return "Highly soluble in water"
            elif prediction > -3:
                return "Moderately soluble in water"
            elif prediction > -5:
                return "Poorly soluble in water"
            else:
                return "Practically insoluble in water"

        elif task_type == "toxicity":
            if prediction > 0.8:
                return "High toxicity risk"
            elif prediction > 0.5:
                return "Moderate toxicity risk"
            elif prediction > 0.3:
                return "Low toxicity risk"
            else:
                return "Very low toxicity risk"

        elif task_type == "bioactivity":
            if prediction > 0.8:
                return "High bioactivity"
            elif prediction > 0.6:
                return "Moderate bioactivity"
            elif prediction > 0.4:
                return "Low bioactivity"
            else:
                return "Minimal bioactivity"

        else:
            return f"Predicted value: {prediction:.3f}"

    def chat(self, user_message: str) -> str:
        """Enhanced chat with intelligent task routing"""

        # Store original query
        original_query = user_message

        # Add to conversation history
        from .chemistry_llm_agent import ConversationMessage
        self.conversation_history.append(
            ConversationMessage(role="user", content=user_message)
        )

        # Reset reasoning trace
        self.reasoning_trace = []

        # Step 1: Detect task type
        task_type, confidence = self.detect_task_type(user_message)
        self._add_reasoning_step(
            f"Detected task type: {task_type} (confidence: {confidence:.2f})",
            "task_detection",
            result={"task_type": task_type, "confidence": confidence}
        )

        # Step 2: Extract molecules
        molecules = self._extract_molecules_from_query(user_message)
        if molecules:
            self._add_reasoning_step(
                f"Found {len(molecules)} molecules: {', '.join(molecules)}",
                "molecule_extraction",
                result=molecules
            )

        # Step 3: Route to appropriate analysis
        if (task_type in self.specialized_models and
            confidence > 0.3 and
            molecules and
            self.specialized_models):

            # Use specialized model
            response = self._process_with_specialized_model(
                task_type, user_message, molecules, confidence
            )
        else:
            # Fall back to general ChemBERTa analysis
            self._add_reasoning_step(
                f"Using general ChemBERTa analysis (no specialized model or low confidence)",
                "fallback_to_general"
            )
            response = self._process_chemberta_query(user_message)

        # Add response to history
        self.conversation_history.append(
            ConversationMessage(role="assistant", content=response)
        )

        return response

    def _process_with_specialized_model(self, task_type: str, query: str,
                                      molecules: List[str], confidence: float) -> str:
        """Process query using specialized model"""

        response = f"I'll analyze this using my specialized {task_type} ChemBERTa model.\n\n"

        # Add reasoning process
        response += "**My Intelligent Analysis Process:**\n"
        for step in self.reasoning_trace:
            response += f"{step.step_number}. {step.description}\n"
        response += "\n"

        # Make predictions with specialized model
        specialized_predictions = {}

        for molecule in molecules:
            try:
                result = self.predict_with_specialized_model(task_type, molecule)
                specialized_predictions[molecule] = result

                self._add_reasoning_step(
                    f"Specialized {task_type} prediction for {molecule}: {result['prediction']:.3f}",
                    f"specialized_prediction",
                    tool_used=f"ChemBERTa-{task_type}",
                    result=result
                )
            except Exception as e:
                self.logger.warning(f"Specialized prediction failed for {molecule}: {e}")

        # Generate response based on specialized predictions
        if specialized_predictions:
            response += f"**Specialized {task_type.title()} Analysis Results:**\n"

            for molecule, result in specialized_predictions.items():
                prediction = result['prediction']
                interpretation = result['interpretation']
                response += f"- {molecule}: {prediction:.3f} ({interpretation})\n"

            # Add insights based on task type
            response += f"\n**Specialized Model Insights:**\n"
            response += f"- Using ChemBERTa fine-tuned specifically for {task_type} prediction\n"
            response += f"- Model trained on {task_type}-specific dataset\n"
            response += f"- Task detection confidence: {confidence:.2f}\n"

            if task_type in self.task_metadata:
                metadata = self.task_metadata[task_type]
                response += f"- Training samples: {metadata.get('num_samples', 'N/A')}\n"
                response += f"- Target range: {metadata.get('target_range', 'N/A')}\n"

        else:
            response += f"No valid predictions could be made.\n"

        # Add comparison with general models
        response += f"\n**Why Specialized Models Excel:**\n"
        response += f"- Fine-tuned on {task_type}-specific data\n"
        response += f"- Better accuracy for {task_type} predictions\n"
        response += f"- Task-optimized feature learning\n"
        response += f"- Domain-specific knowledge integration\n"

        return response

    def explain_capabilities(self) -> str:
        """Explain intelligent agent capabilities"""

        base_capabilities = super().explain_capabilities()

        specialized_info = "\n**Intelligent Task Routing:**\n"
        if self.specialized_models:
            specialized_info += f"Available specialized models: {', '.join(self.specialized_models.keys())}\n"
            specialized_info += "\n**Task Detection Examples:**\n"
            specialized_info += "- 'Is aspirin toxic?' → Toxicity model\n"
            specialized_info += "- 'How soluble is caffeine?' → Solubility model\n"
            specialized_info += "- 'What's the bioactivity of ibuprofen?' → Bioactivity model\n"
            specialized_info += "- 'Find similar molecules' → General ChemBERTa\n"
        else:
            specialized_info += "No specialized models loaded. Run training pipeline to enable task-specific routing.\n"

        return base_capabilities + specialized_info

def create_intelligent_chemberta_agent(config: Optional[AgentConfig] = None) -> IntelligentChemBERTaAgent:
    """Create and initialize intelligent ChemBERTa agent"""

    agent = IntelligentChemBERTaAgent(config)

    # Load base ChemBERTa model
    agent.load_model()

    # Load specialized models if available
    agent.load_specialized_models()

    return agent