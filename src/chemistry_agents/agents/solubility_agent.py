import numpy as np
from typing import List, Optional, Dict, Any
from .property_prediction_agent import PropertyPredictionAgent
from .base_agent import PredictionResult, AgentConfig

class SolubilityAgent(PropertyPredictionAgent):
    """
    Specialized agent for predicting aqueous solubility of molecules
    """
    
    def __init__(self, 
                 config: Optional[AgentConfig] = None,
                 model_type: str = "neural_network",
                 transformer_model: str = "DeepChem/ChemBERTa-77M-MLM"):
        super().__init__(
            config=config,
            property_name="solubility",
            model_type=model_type,
            transformer_model=transformer_model
        )
        
        # Solubility-specific thresholds (log S values)
        self.solubility_classes = {
            "highly_soluble": (-1, float('inf')),      # > -1 log S
            "soluble": (-3, -1),                       # -3 to -1 log S
            "moderately_soluble": (-5, -3),           # -5 to -3 log S
            "poorly_soluble": (-7, -5),               # -7 to -5 log S
            "insoluble": (float('-inf'), -7)          # < -7 log S
        }
    
    def predict_single(self, smiles: str) -> PredictionResult:
        """Make solubility prediction for a single molecule"""
        result = super().predict_single(smiles)
        
        if result.additional_info and "error" not in result.additional_info:
            # Add solubility-specific information
            solubility_class = self._classify_solubility(result.prediction)
            lipinski_compliance = self._check_lipinski_rule(smiles)
            
            result.additional_info.update({
                "solubility_class": solubility_class,
                "lipinski_compliant": lipinski_compliance,
                "unit": "log S (mol/L)",
                "interpretation": self._interpret_solubility(result.prediction)
            })
        
        return result
    
    def _classify_solubility(self, log_s_value: float) -> str:
        """Classify solubility based on log S value"""
        for class_name, (min_val, max_val) in self.solubility_classes.items():
            if min_val <= log_s_value < max_val:
                return class_name
        return "unknown"
    
    def _interpret_solubility(self, log_s_value: float) -> str:
        """Provide interpretation of solubility value"""
        solubility_class = self._classify_solubility(log_s_value)
        
        interpretations = {
            "highly_soluble": "Excellent aqueous solubility, suitable for most formulations",
            "soluble": "Good aqueous solubility, generally drug-like",
            "moderately_soluble": "Moderate solubility, may require formulation optimization",
            "poorly_soluble": "Poor solubility, significant formulation challenges",
            "insoluble": "Very poor solubility, major development hurdle"
        }
        
        return interpretations.get(solubility_class, "Unknown solubility class")
    
    def _check_lipinski_rule(self, smiles: str) -> Dict[str, Any]:
        """Check Lipinski's Rule of Five compliance"""
        try:
            from rdkit import Chem
            from rdkit.Chem import Descriptors
            
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return {"compliant": False, "reason": "Invalid SMILES"}
            
            # Calculate Lipinski descriptors
            mw = Descriptors.MolWt(mol)
            logp = Descriptors.MolLogP(mol)
            hbd = Descriptors.NumHDonors(mol)
            hba = Descriptors.NumHAcceptors(mol)
            
            # Check compliance
            violations = []
            if mw > 500:
                violations.append("Molecular weight > 500 Da")
            if logp > 5:
                violations.append("LogP > 5")
            if hbd > 5:
                violations.append("H-bond donors > 5")
            if hba > 10:
                violations.append("H-bond acceptors > 10")
            
            return {
                "compliant": len(violations) <= 1,  # Allow one violation
                "violations": violations,
                "descriptors": {
                    "molecular_weight": mw,
                    "logp": logp,
                    "hbd": hbd,
                    "hba": hba
                }
            }
            
        except Exception as e:
            return {"compliant": False, "reason": f"Calculation error: {e}"}
    
    def predict_formulation_feasibility(self, smiles_list: List[str]) -> List[Dict[str, Any]]:
        """Predict formulation feasibility based on solubility"""
        results = self.predict_batch(smiles_list)
        
        feasibility_assessments = []
        
        for result in results:
            if result.additional_info and "error" not in result.additional_info:
                solubility_class = result.additional_info.get("solubility_class", "unknown")
                lipinski_compliant = result.additional_info.get("lipinski_compliant", {}).get("compliant", False)
                
                # Assess formulation feasibility
                if solubility_class in ["highly_soluble", "soluble"] and lipinski_compliant:
                    feasibility = "high"
                    recommendations = ["Standard oral formulation feasible"]
                elif solubility_class == "moderately_soluble":
                    feasibility = "medium"
                    recommendations = [
                        "Consider salt forms",
                        "Evaluate particle size reduction",
                        "Solid dispersion formulation"
                    ]
                elif solubility_class == "poorly_soluble":
                    feasibility = "low"
                    recommendations = [
                        "Nanotechnology approaches",
                        "Lipid-based formulations",
                        "Amorphous solid dispersions",
                        "Cyclodextrin complexation"
                    ]
                else:
                    feasibility = "very_low"
                    recommendations = [
                        "Structural modification required",
                        "Prodrug approach",
                        "Alternative delivery routes"
                    ]
                
                assessment = {
                    "smiles": result.smiles,
                    "solubility_prediction": result.prediction,
                    "formulation_feasibility": feasibility,
                    "recommendations": recommendations,
                    "risk_factors": self._identify_risk_factors(result)
                }
            else:
                assessment = {
                    "smiles": result.smiles,
                    "formulation_feasibility": "unknown",
                    "error": result.additional_info.get("error", "Unknown error")
                }
            
            feasibility_assessments.append(assessment)
        
        return feasibility_assessments
    
    def _identify_risk_factors(self, result: PredictionResult) -> List[str]:
        """Identify potential risk factors for formulation"""
        risk_factors = []
        
        if result.additional_info:
            lipinski_info = result.additional_info.get("lipinski_compliant", {})
            violations = lipinski_info.get("violations", [])
            
            if violations:
                risk_factors.extend([f"Lipinski violation: {v}" for v in violations])
            
            descriptors = lipinski_info.get("descriptors", {})
            
            # Additional risk factors
            if descriptors.get("molecular_weight", 0) > 600:
                risk_factors.append("Very high molecular weight")
            
            if descriptors.get("logp", 0) > 6:
                risk_factors.append("Very high lipophilicity")
            
            if descriptors.get("logp", 0) < -2:
                risk_factors.append("Very high hydrophilicity")
        
        return risk_factors
    
    def compare_solubility(self, smiles_list: List[str]) -> Dict[str, Any]:
        """Compare solubility predictions for multiple molecules"""
        results = self.predict_batch(smiles_list)
        
        valid_results = [r for r in results if r.additional_info and "error" not in r.additional_info]
        
        if not valid_results:
            return {"error": "No valid predictions available"}
        
        predictions = [r.prediction for r in valid_results]
        
        comparison = {
            "molecules": len(valid_results),
            "best_solubility": {
                "smiles": valid_results[np.argmax(predictions)].smiles,
                "log_s": max(predictions),
                "class": self._classify_solubility(max(predictions))
            },
            "worst_solubility": {
                "smiles": valid_results[np.argmin(predictions)].smiles,
                "log_s": min(predictions),
                "class": self._classify_solubility(min(predictions))
            },
            "statistics": {
                "mean_log_s": np.mean(predictions),
                "std_log_s": np.std(predictions),
                "median_log_s": np.median(predictions)
            },
            "class_distribution": self._get_class_distribution(valid_results)
        }
        
        return comparison
    
    def _get_class_distribution(self, results: List[PredictionResult]) -> Dict[str, int]:
        """Get distribution of solubility classes"""
        distribution = {class_name: 0 for class_name in self.solubility_classes.keys()}
        
        for result in results:
            if result.additional_info:
                solubility_class = result.additional_info.get("solubility_class", "unknown")
                if solubility_class in distribution:
                    distribution[solubility_class] += 1
        
        return distribution