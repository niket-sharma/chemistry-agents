import numpy as np
from typing import List, Optional, Dict, Any, Tuple
from .property_prediction_agent import PropertyPredictionAgent
from .base_agent import PredictionResult, AgentConfig

class ToxicityAgent(PropertyPredictionAgent):
    """
    Specialized agent for predicting various toxicity endpoints of molecules
    """
    
    def __init__(self, 
                 config: Optional[AgentConfig] = None,
                 toxicity_endpoint: str = "acute_toxicity",
                 model_type: str = "neural_network",
                 transformer_model: str = "DeepChem/ChemBERTa-77M-MLM"):
        super().__init__(
            config=config,
            property_name=f"{toxicity_endpoint}_prediction",
            model_type=model_type,
            transformer_model=transformer_model
        )
        
        self.toxicity_endpoint = toxicity_endpoint
        
        # Define toxicity thresholds and classes
        self.toxicity_endpoints = {
            "acute_toxicity": {
                "unit": "LD50 (mg/kg)",
                "classes": {
                    "non_toxic": (2000, float('inf')),
                    "low_toxicity": (300, 2000),
                    "moderate_toxicity": (50, 300),
                    "high_toxicity": (5, 50),
                    "extremely_toxic": (0, 5)
                }
            },
            "hepatotoxicity": {
                "unit": "probability",
                "classes": {
                    "non_hepatotoxic": (0, 0.3),
                    "low_risk": (0.3, 0.5),
                    "moderate_risk": (0.5, 0.7),
                    "high_risk": (0.7, 1.0)
                }
            },
            "cardiotoxicity": {
                "unit": "probability",
                "classes": {
                    "non_cardiotoxic": (0, 0.3),
                    "low_risk": (0.3, 0.5),
                    "moderate_risk": (0.5, 0.7),
                    "high_risk": (0.7, 1.0)
                }
            },
            "mutagenicity": {
                "unit": "probability",
                "classes": {
                    "non_mutagenic": (0, 0.3),
                    "low_risk": (0.3, 0.5),
                    "moderate_risk": (0.5, 0.7),
                    "high_risk": (0.7, 1.0)
                }
            },
            "skin_sensitization": {
                "unit": "probability",
                "classes": {
                    "non_sensitizing": (0, 0.3),
                    "weak_sensitizer": (0.3, 0.5),
                    "moderate_sensitizer": (0.5, 0.7),
                    "strong_sensitizer": (0.7, 1.0)
                }
            }
        }
    
    def predict_single(self, smiles: str) -> PredictionResult:
        """Make toxicity prediction for a single molecule"""
        result = super().predict_single(smiles)
        
        if result.additional_info and "error" not in result.additional_info:
            # Add toxicity-specific information
            toxicity_class = self._classify_toxicity(result.prediction)
            structural_alerts = self._check_structural_alerts(smiles)
            safety_assessment = self._assess_safety_profile(result.prediction, structural_alerts)
            
            result.additional_info.update({
                "toxicity_endpoint": self.toxicity_endpoint,
                "toxicity_class": toxicity_class,
                "unit": self.toxicity_endpoints[self.toxicity_endpoint]["unit"],
                "structural_alerts": structural_alerts,
                "safety_assessment": safety_assessment,
                "interpretation": self._interpret_toxicity(result.prediction)
            })
        
        return result
    
    def _classify_toxicity(self, prediction_value: float) -> str:
        """Classify toxicity based on prediction value"""
        endpoint_info = self.toxicity_endpoints.get(self.toxicity_endpoint, {})
        classes = endpoint_info.get("classes", {})
        
        for class_name, (min_val, max_val) in classes.items():
            if min_val <= prediction_value < max_val:
                return class_name
        return "unknown"
    
    def _interpret_toxicity(self, prediction_value: float) -> str:
        """Provide interpretation of toxicity prediction"""
        toxicity_class = self._classify_toxicity(prediction_value)
        
        interpretations = {
            "acute_toxicity": {
                "non_toxic": "Low acute toxicity risk, generally safe",
                "low_toxicity": "Low to moderate acute toxicity, use with caution",
                "moderate_toxicity": "Moderate acute toxicity, requires safety measures",
                "high_toxicity": "High acute toxicity, significant safety concern",
                "extremely_toxic": "Extremely toxic, avoid human exposure"
            },
            "hepatotoxicity": {
                "non_hepatotoxic": "Low risk of liver toxicity",
                "low_risk": "Low risk of hepatotoxicity, monitor liver function",
                "moderate_risk": "Moderate hepatotoxicity risk, careful monitoring required",
                "high_risk": "High hepatotoxicity risk, consider alternatives"
            },
            "cardiotoxicity": {
                "non_cardiotoxic": "Low risk of cardiac toxicity",
                "low_risk": "Low cardiotoxicity risk, monitor cardiac function",
                "moderate_risk": "Moderate cardiac risk, ECG monitoring recommended",
                "high_risk": "High cardiotoxicity risk, consider alternatives"
            },
            "mutagenicity": {
                "non_mutagenic": "Low mutagenic potential",
                "low_risk": "Low mutagenic risk, additional testing may be needed",
                "moderate_risk": "Moderate mutagenic concern, genotoxicity studies required",
                "high_risk": "High mutagenic risk, avoid development"
            },
            "skin_sensitization": {
                "non_sensitizing": "Low skin sensitization potential",
                "weak_sensitizer": "Weak skin sensitizer, patch testing recommended",
                "moderate_sensitizer": "Moderate sensitization risk, dermal safety evaluation needed",
                "strong_sensitizer": "Strong skin sensitizer, avoid dermal exposure"
            }
        }
        
        endpoint_interpretations = interpretations.get(self.toxicity_endpoint, {})
        return endpoint_interpretations.get(toxicity_class, "Unknown toxicity class")
    
    def _check_structural_alerts(self, smiles: str) -> Dict[str, Any]:
        """Check for structural alerts (toxicophores)"""
        try:
            from rdkit import Chem
            from rdkit.Chem import rdMolDescriptors
            
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return {"alerts": [], "error": "Invalid SMILES"}
            
            # Define common toxicophore patterns (SMARTS)
            toxicophore_patterns = {
                "aromatic_amine": "[c:1][NH2,NH]",
                "nitro_group": "[N+](=O)[O-]",
                "aldehyde": "[CH]=O",
                "epoxide": "C1OC1",
                "quinone": "O=C1C=CC(=O)C=C1",
                "michael_acceptor": "C=CC(=O)",
                "aromatic_nitro": "c[N+](=O)[O-]",
                "halogenated_aromatic": "c[F,Cl,Br,I]",
                "thiocarbonyl": "C=S",
                "phosphate_ester": "P(=O)(O)(O)O"
            }
            
            alerts = []
            for alert_name, pattern in toxicophore_patterns.items():
                try:
                    pattern_mol = Chem.MolFromSmarts(pattern)
                    if pattern_mol and mol.HasSubstructMatch(pattern_mol):
                        matches = mol.GetSubstructMatches(pattern_mol)
                        alerts.append({
                            "alert_type": alert_name,
                            "pattern": pattern,
                            "matches": len(matches)
                        })
                except:
                    continue
            
            return {
                "alerts": alerts,
                "alert_count": len(alerts),
                "high_priority_alerts": [a for a in alerts if a["alert_type"] in 
                                       ["aromatic_amine", "nitro_group", "quinone"]]
            }
            
        except Exception as e:
            return {"alerts": [], "error": f"Structural alert analysis failed: {e}"}
    
    def _assess_safety_profile(self, prediction_value: float, structural_alerts: Dict[str, Any]) -> Dict[str, Any]:
        """Assess overall safety profile"""
        toxicity_class = self._classify_toxicity(prediction_value)
        alert_count = structural_alerts.get("alert_count", 0)
        high_priority_alerts = len(structural_alerts.get("high_priority_alerts", []))
        
        # Calculate safety score (0-100, higher is safer)
        if self.toxicity_endpoint == "acute_toxicity":
            # For LD50, higher values are safer
            base_score = min(100, (prediction_value / 2000) * 80)
        else:
            # For probability-based endpoints, lower values are safer
            base_score = (1 - prediction_value) * 80
        
        # Penalize for structural alerts
        alert_penalty = min(30, alert_count * 5 + high_priority_alerts * 10)
        safety_score = max(0, base_score - alert_penalty)
        
        # Determine overall risk level
        if safety_score >= 70:
            risk_level = "low"
        elif safety_score >= 50:
            risk_level = "moderate"
        elif safety_score >= 30:
            risk_level = "high"
        else:
            risk_level = "very_high"
        
        return {
            "safety_score": round(safety_score, 1),
            "risk_level": risk_level,
            "primary_concerns": self._identify_primary_concerns(toxicity_class, structural_alerts),
            "recommendations": self._generate_safety_recommendations(risk_level, structural_alerts)
        }
    
    def _identify_primary_concerns(self, toxicity_class: str, structural_alerts: Dict[str, Any]) -> List[str]:
        """Identify primary safety concerns"""
        concerns = []
        
        # Toxicity-based concerns
        if toxicity_class in ["high_toxicity", "extremely_toxic", "high_risk"]:
            concerns.append(f"High {self.toxicity_endpoint} risk")
        
        # Structural alert concerns
        high_priority_alerts = structural_alerts.get("high_priority_alerts", [])
        for alert in high_priority_alerts:
            concerns.append(f"Contains {alert['alert_type']} toxicophore")
        
        if structural_alerts.get("alert_count", 0) > 3:
            concerns.append("Multiple structural alerts present")
        
        return concerns
    
    def _generate_safety_recommendations(self, risk_level: str, structural_alerts: Dict[str, Any]) -> List[str]:
        """Generate safety recommendations"""
        recommendations = []
        
        if risk_level == "low":
            recommendations.append("Proceed with standard safety protocols")
        elif risk_level == "moderate":
            recommendations.extend([
                "Conduct additional safety studies",
                "Implement enhanced safety monitoring"
            ])
        elif risk_level == "high":
            recommendations.extend([
                "Extensive safety evaluation required",
                "Consider structural modifications",
                "Implement strict exposure controls"
            ])
        else:  # very_high
            recommendations.extend([
                "Significant safety concerns identified",
                "Strong recommendation for structural modification",
                "Consider discontinuing development"
            ])
        
        # Alert-specific recommendations
        if structural_alerts.get("alert_count", 0) > 0:
            recommendations.append("Address structural alerts through medicinal chemistry")
        
        return recommendations
    
    def predict_multi_endpoint_toxicity(self, smiles_list: List[str], 
                                       endpoints: List[str]) -> Dict[str, List[PredictionResult]]:
        """Predict multiple toxicity endpoints for molecules"""
        results = {}
        
        for endpoint in endpoints:
            if endpoint in self.toxicity_endpoints:
                # Temporarily change endpoint
                original_endpoint = self.toxicity_endpoint
                self.toxicity_endpoint = endpoint
                self.property_name = f"{endpoint}_prediction"
                
                # Make predictions
                endpoint_results = self.predict_batch(smiles_list)
                results[endpoint] = endpoint_results
                
                # Restore original endpoint
                self.toxicity_endpoint = original_endpoint
                self.property_name = f"{original_endpoint}_prediction"
        
        return results
    
    def generate_toxicity_report(self, smiles: str) -> Dict[str, Any]:
        """Generate comprehensive toxicity report for a molecule"""
        # Predict multiple endpoints
        endpoints = ["acute_toxicity", "hepatotoxicity", "cardiotoxicity", "mutagenicity"]
        multi_results = self.predict_multi_endpoint_toxicity([smiles], endpoints)
        
        # Compile comprehensive report
        report = {
            "smiles": smiles,
            "overall_assessment": {},
            "endpoint_predictions": {},
            "structural_analysis": {},
            "recommendations": []
        }
        
        # Process each endpoint
        overall_risk_scores = []
        for endpoint, results in multi_results.items():
            if results and results[0].additional_info and "error" not in results[0].additional_info:
                result = results[0]
                safety_assessment = result.additional_info.get("safety_assessment", {})
                safety_score = safety_assessment.get("safety_score", 50)
                overall_risk_scores.append(safety_score)
                
                report["endpoint_predictions"][endpoint] = {
                    "prediction": result.prediction,
                    "class": result.additional_info.get("toxicity_class"),
                    "safety_score": safety_score,
                    "interpretation": result.additional_info.get("interpretation")
                }
        
        # Calculate overall assessment
        if overall_risk_scores:
            avg_safety_score = np.mean(overall_risk_scores)
            min_safety_score = min(overall_risk_scores)
            
            report["overall_assessment"] = {
                "average_safety_score": round(avg_safety_score, 1),
                "worst_case_score": round(min_safety_score, 1),
                "development_recommendation": self._get_development_recommendation(min_safety_score)
            }
        
        return report
    
    def _get_development_recommendation(self, min_safety_score: float) -> str:
        """Get development recommendation based on worst-case safety score"""
        if min_safety_score >= 70:
            return "Proceed with development - favorable safety profile"
        elif min_safety_score >= 50:
            return "Proceed with caution - moderate safety concerns"
        elif min_safety_score >= 30:
            return "High risk - consider structural modifications"
        else:
            return "Not recommended - significant safety concerns"