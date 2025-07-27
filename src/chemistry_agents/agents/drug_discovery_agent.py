import numpy as np
from typing import List, Optional, Dict, Any, Tuple
from .base_agent import BaseChemistryAgent, PredictionResult, AgentConfig
from .solubility_agent import SolubilityAgent
from .toxicity_agent import ToxicityAgent
from .property_prediction_agent import PropertyPredictionAgent

class DrugDiscoveryAgent(BaseChemistryAgent):
    """
    Comprehensive agent for drug discovery tasks combining multiple prediction models
    """
    
    def __init__(self, config: Optional[AgentConfig] = None):
        super().__init__(config)
        
        # Initialize specialized agents
        self.solubility_agent = SolubilityAgent(config)
        self.toxicity_agent = ToxicityAgent(config)
        self.bioactivity_agent = PropertyPredictionAgent(
            config, 
            property_name="bioactivity",
            model_type="transformer"
        )
        
        # Drug-likeness criteria
        self.drug_likeness_criteria = {
            "lipinski_rule_of_five": True,
            "veber_rules": True,
            "ghose_filter": True,
            "muegge_rules": True
        }
        
        # Target profile weights for multi-parameter optimization
        self.target_profile_weights = {
            "potency": 0.25,
            "selectivity": 0.20,
            "solubility": 0.20,
            "toxicity": 0.20,
            "permeability": 0.15
        }
    
    def load_model(self, model_path: Optional[str] = None) -> None:
        """Load all models for the drug discovery pipeline"""
        try:
            # Load individual agent models
            if model_path:
                solubility_model = f"{model_path}/solubility_model.pt"
                toxicity_model = f"{model_path}/toxicity_model.pt"
                bioactivity_model = f"{model_path}/bioactivity_model.pt"
                
                self.solubility_agent.load_model(solubility_model)
                self.toxicity_agent.load_model(toxicity_model)
                self.bioactivity_agent.load_model(bioactivity_model)
            else:
                # Load default models
                self.solubility_agent.load_model()
                self.toxicity_agent.load_model()
                self.bioactivity_agent.load_model()
            
            self.is_loaded = True
            self.logger.info("All drug discovery models loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to load drug discovery models: {e}")
            raise
    
    def predict_single(self, smiles: str) -> PredictionResult:
        """Comprehensive drug discovery prediction for a single molecule"""
        if not self.is_loaded:
            raise RuntimeError("Models not loaded. Call load_model() first.")
        
        # Get predictions from all agents
        solubility_result = self.solubility_agent.predict_single(smiles)
        toxicity_result = self.toxicity_agent.predict_single(smiles)
        bioactivity_result = self.bioactivity_agent.predict_single(smiles)
        
        # Calculate drug-likeness scores
        drug_likeness = self._assess_drug_likeness(smiles)
        
        # Calculate overall drug discovery score
        discovery_score = self._calculate_discovery_score(
            solubility_result, toxicity_result, bioactivity_result, drug_likeness
        )
        
        # Compile comprehensive result
        result = PredictionResult(
            smiles=smiles,
            prediction=discovery_score,
            confidence=self._calculate_overall_confidence([
                solubility_result, toxicity_result, bioactivity_result
            ]),
            additional_info={
                "discovery_score": discovery_score,
                "solubility": {
                    "prediction": solubility_result.prediction,
                    "class": solubility_result.additional_info.get("solubility_class") if solubility_result.additional_info else None
                },
                "toxicity": {
                    "prediction": toxicity_result.prediction,
                    "class": toxicity_result.additional_info.get("toxicity_class") if toxicity_result.additional_info else None
                },
                "bioactivity": {
                    "prediction": bioactivity_result.prediction
                },
                "drug_likeness": drug_likeness,
                "development_recommendation": self._get_development_recommendation(discovery_score, drug_likeness)
            }
        )
        
        return result
    
    def _assess_drug_likeness(self, smiles: str) -> Dict[str, Any]:
        """Assess drug-likeness using multiple rule sets"""
        try:
            from rdkit import Chem
            from rdkit.Chem import Descriptors, rdMolDescriptors
            
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return {"error": "Invalid SMILES"}
            
            # Calculate descriptors
            mw = Descriptors.MolWt(mol)
            logp = Descriptors.MolLogP(mol)
            hbd = Descriptors.NumHDonors(mol)
            hba = Descriptors.NumHAcceptors(mol)
            tpsa = Descriptors.TPSA(mol)
            rotatable_bonds = Descriptors.NumRotatableBonds(mol)
            aromatic_rings = rdMolDescriptors.CalcNumAromaticRings(mol)
            
            # Lipinski Rule of Five
            lipinski_violations = 0
            lipinski_details = {}
            
            if mw > 500:
                lipinski_violations += 1
                lipinski_details["molecular_weight"] = "violation"
            if logp > 5:
                lipinski_violations += 1
                lipinski_details["logp"] = "violation"
            if hbd > 5:
                lipinski_violations += 1
                lipinski_details["hbd"] = "violation"
            if hba > 10:
                lipinski_violations += 1
                lipinski_details["hba"] = "violation"
            
            # Veber Rules (oral bioavailability)
            veber_compliant = tpsa <= 140 and rotatable_bonds <= 10
            
            # Ghose Filter
            ghose_compliant = (160 <= mw <= 480 and 
                             -0.4 <= logp <= 5.6 and 
                             20 <= mw/hba <= 70 and
                             aromatic_rings >= 1)
            
            # Muegge Rules
            muegge_compliant = (200 <= mw <= 600 and
                              -2 <= logp <= 5 and
                              tpsa <= 150 and
                              aromatic_rings <= 7 and
                              rotatable_bonds <= 15 and
                              hbd <= 5 and
                              hba <= 10)
            
            # Calculate overall drug-likeness score
            rule_scores = {
                "lipinski": max(0, 1 - lipinski_violations / 4),
                "veber": 1.0 if veber_compliant else 0.0,
                "ghose": 1.0 if ghose_compliant else 0.0,
                "muegge": 1.0 if muegge_compliant else 0.0
            }
            
            overall_score = np.mean(list(rule_scores.values()))
            
            return {
                "overall_score": overall_score,
                "lipinski": {
                    "compliant": lipinski_violations <= 1,
                    "violations": lipinski_violations,
                    "details": lipinski_details
                },
                "veber": {"compliant": veber_compliant},
                "ghose": {"compliant": ghose_compliant},
                "muegge": {"compliant": muegge_compliant},
                "descriptors": {
                    "molecular_weight": mw,
                    "logp": logp,
                    "hbd": hbd,
                    "hba": hba,
                    "tpsa": tpsa,
                    "rotatable_bonds": rotatable_bonds,
                    "aromatic_rings": aromatic_rings
                }
            }
            
        except Exception as e:
            return {"error": f"Drug-likeness assessment failed: {e}"}
    
    def _calculate_discovery_score(self, 
                                 solubility_result: PredictionResult,
                                 toxicity_result: PredictionResult,
                                 bioactivity_result: PredictionResult,
                                 drug_likeness: Dict[str, Any]) -> float:
        """Calculate overall drug discovery score (0-100)"""
        scores = {}
        
        # Solubility score (0-100, higher is better)
        if solubility_result.additional_info and "error" not in solubility_result.additional_info:
            # Convert log S to score (assuming log S range from -10 to 0)
            solubility_score = min(100, max(0, (solubility_result.prediction + 10) * 10))
            scores["solubility"] = solubility_score
        
        # Toxicity score (0-100, higher is safer)
        if toxicity_result.additional_info and "error" not in toxicity_result.additional_info:
            safety_assessment = toxicity_result.additional_info.get("safety_assessment", {})
            toxicity_score = safety_assessment.get("safety_score", 50)
            scores["toxicity"] = toxicity_score
        
        # Bioactivity score (0-100, higher activity is better)
        if bioactivity_result.additional_info and "error" not in bioactivity_result.additional_info:
            # Assuming bioactivity prediction is probability (0-1)
            bioactivity_score = bioactivity_result.prediction * 100
            scores["bioactivity"] = bioactivity_score
        
        # Drug-likeness score (0-100)
        if "error" not in drug_likeness:
            drug_likeness_score = drug_likeness.get("overall_score", 0.5) * 100
            scores["drug_likeness"] = drug_likeness_score
        
        # Calculate weighted average
        if scores:
            weights = {
                "solubility": 0.25,
                "toxicity": 0.30,
                "bioactivity": 0.25,
                "drug_likeness": 0.20
            }
            
            weighted_score = sum(scores[key] * weights.get(key, 0.25) 
                               for key in scores) / sum(weights[key] for key in scores)
            return min(100, max(0, weighted_score))
        
        return 0.0
    
    def _calculate_overall_confidence(self, results: List[PredictionResult]) -> float:
        """Calculate overall confidence from individual predictions"""
        confidences = [r.confidence for r in results if r.confidence is not None]
        return np.mean(confidences) if confidences else 0.5
    
    def _get_development_recommendation(self, discovery_score: float, drug_likeness: Dict[str, Any]) -> str:
        """Get development recommendation based on discovery score and drug-likeness"""
        if discovery_score >= 80 and drug_likeness.get("overall_score", 0) >= 0.7:
            return "Highly recommended for development"
        elif discovery_score >= 60 and drug_likeness.get("overall_score", 0) >= 0.5:
            return "Recommended for development with optimization"
        elif discovery_score >= 40:
            return "Consider for development after significant optimization"
        else:
            return "Not recommended for development"
    
    def screen_compound_library(self, smiles_list: List[str], 
                              criteria: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Screen a compound library for drug discovery potential"""
        criteria = criteria or {
            "min_discovery_score": 50,
            "require_lipinski_compliance": True,
            "max_toxicity_risk": "moderate"
        }
        
        results = self.predict_batch(smiles_list)
        
        # Filter compounds based on criteria
        promising_compounds = []
        for result in results:
            if self._meets_screening_criteria(result, criteria):
                promising_compounds.append(result)
        
        # Rank compounds by discovery score
        promising_compounds.sort(key=lambda x: x.prediction, reverse=True)
        
        # Generate screening report
        report = {
            "total_compounds": len(smiles_list),
            "promising_compounds": len(promising_compounds),
            "hit_rate": len(promising_compounds) / len(smiles_list) if smiles_list else 0,
            "top_candidates": [self._format_candidate_info(r) for r in promising_compounds[:10]],
            "screening_criteria": criteria,
            "statistics": self._calculate_screening_statistics(results)
        }
        
        return report
    
    def _meets_screening_criteria(self, result: PredictionResult, criteria: Dict[str, Any]) -> bool:
        """Check if compound meets screening criteria"""
        if not result.additional_info:
            return False
        
        # Check discovery score
        min_score = criteria.get("min_discovery_score", 0)
        if result.prediction < min_score:
            return False
        
        # Check Lipinski compliance
        if criteria.get("require_lipinski_compliance", False):
            drug_likeness = result.additional_info.get("drug_likeness", {})
            lipinski_info = drug_likeness.get("lipinski", {})
            if not lipinski_info.get("compliant", False):
                return False
        
        # Check toxicity risk
        max_toxicity = criteria.get("max_toxicity_risk", "high")
        toxicity_info = result.additional_info.get("toxicity", {})
        toxicity_class = toxicity_info.get("class", "unknown")
        
        risk_hierarchy = ["low", "moderate", "high", "very_high"]
        if max_toxicity in risk_hierarchy and toxicity_class in risk_hierarchy:
            if risk_hierarchy.index(toxicity_class) > risk_hierarchy.index(max_toxicity):
                return False
        
        return True
    
    def _format_candidate_info(self, result: PredictionResult) -> Dict[str, Any]:
        """Format candidate information for reporting"""
        additional_info = result.additional_info or {}
        
        return {
            "smiles": result.smiles,
            "discovery_score": round(result.prediction, 2),
            "confidence": round(result.confidence or 0, 2),
            "solubility_class": additional_info.get("solubility", {}).get("class"),
            "toxicity_class": additional_info.get("toxicity", {}).get("class"),
            "drug_likeness_score": round(additional_info.get("drug_likeness", {}).get("overall_score", 0) * 100, 1),
            "recommendation": additional_info.get("development_recommendation")
        }
    
    def _calculate_screening_statistics(self, results: List[PredictionResult]) -> Dict[str, Any]:
        """Calculate screening statistics"""
        valid_results = [r for r in results if r.additional_info and "error" not in r.additional_info]
        
        if not valid_results:
            return {"error": "No valid results"}
        
        discovery_scores = [r.prediction for r in valid_results]
        
        # Drug-likeness statistics
        drug_likeness_scores = []
        lipinski_compliant = 0
        
        for result in valid_results:
            drug_likeness = result.additional_info.get("drug_likeness", {})
            if "error" not in drug_likeness:
                drug_likeness_scores.append(drug_likeness.get("overall_score", 0))
                if drug_likeness.get("lipinski", {}).get("compliant", False):
                    lipinski_compliant += 1
        
        return {
            "discovery_score": {
                "mean": np.mean(discovery_scores),
                "median": np.median(discovery_scores),
                "std": np.std(discovery_scores),
                "min": np.min(discovery_scores),
                "max": np.max(discovery_scores)
            },
            "drug_likeness": {
                "mean_score": np.mean(drug_likeness_scores) if drug_likeness_scores else 0,
                "lipinski_compliance_rate": lipinski_compliant / len(valid_results)
            }
        }
    
    def optimize_lead_compound(self, smiles: str, 
                             optimization_targets: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """Analyze lead compound and suggest optimization strategies"""
        optimization_targets = optimization_targets or {
            "min_discovery_score": 70,
            "min_solubility": -3,  # log S
            "max_toxicity_score": 30,  # safety score
            "min_drug_likeness": 0.7
        }
        
        # Get current profile
        current_result = self.predict_single(smiles)
        
        if not current_result.additional_info or "error" in current_result.additional_info:
            return {"error": "Failed to analyze lead compound"}
        
        # Identify optimization opportunities
        optimization_needs = self._identify_optimization_needs(current_result, optimization_targets)
        
        # Generate optimization strategies
        strategies = self._generate_optimization_strategies(current_result, optimization_needs)
        
        return {
            "lead_compound": smiles,
            "current_profile": self._format_candidate_info(current_result),
            "optimization_targets": optimization_targets,
            "optimization_needs": optimization_needs,
            "recommended_strategies": strategies,
            "priority_areas": self._prioritize_optimization_areas(optimization_needs)
        }
    
    def _identify_optimization_needs(self, result: PredictionResult, 
                                   targets: Dict[str, float]) -> Dict[str, Any]:
        """Identify areas needing optimization"""
        needs = {}
        additional_info = result.additional_info or {}
        
        # Discovery score
        if result.prediction < targets.get("min_discovery_score", 70):
            needs["discovery_score"] = {
                "current": result.prediction,
                "target": targets["min_discovery_score"],
                "gap": targets["min_discovery_score"] - result.prediction
            }
        
        # Solubility
        solubility_pred = additional_info.get("solubility", {}).get("prediction")
        if solubility_pred and solubility_pred < targets.get("min_solubility", -3):
            needs["solubility"] = {
                "current": solubility_pred,
                "target": targets["min_solubility"],
                "gap": targets["min_solubility"] - solubility_pred
            }
        
        # Drug-likeness
        drug_likeness_score = additional_info.get("drug_likeness", {}).get("overall_score", 0)
        if drug_likeness_score < targets.get("min_drug_likeness", 0.7):
            needs["drug_likeness"] = {
                "current": drug_likeness_score,
                "target": targets["min_drug_likeness"],
                "gap": targets["min_drug_likeness"] - drug_likeness_score
            }
        
        return needs
    
    def _generate_optimization_strategies(self, result: PredictionResult, 
                                        needs: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate optimization strategies based on identified needs"""
        strategies = []
        
        if "solubility" in needs:
            strategies.append({
                "area": "solubility",
                "strategy": "Improve aqueous solubility",
                "approaches": [
                    "Add polar functional groups (OH, NH2, COOH)",
                    "Reduce lipophilicity by removing hydrophobic substituents",
                    "Consider salt formation",
                    "Introduce ionizable groups at physiological pH"
                ]
            })
        
        if "drug_likeness" in needs:
            strategies.append({
                "area": "drug_likeness",
                "strategy": "Improve drug-like properties",
                "approaches": [
                    "Reduce molecular weight if > 500 Da",
                    "Optimize LogP to 1-3 range",
                    "Minimize rotatable bonds",
                    "Reduce polar surface area if > 140 Ų"
                ]
            })
        
        # Add more strategies based on other needs...
        
        return strategies
    
    def _prioritize_optimization_areas(self, needs: Dict[str, Any]) -> List[str]:
        """Prioritize optimization areas based on gap size and importance"""
        priorities = []
        
        # Sort by gap size (normalized)
        area_gaps = [(area, info.get("gap", 0)) for area, info in needs.items()]
        area_gaps.sort(key=lambda x: x[1], reverse=True)
        
        for area, gap in area_gaps:
            priorities.append(area)
        
        return priorities