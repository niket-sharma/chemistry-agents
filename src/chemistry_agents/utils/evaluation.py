import numpy as np
import pandas as pd
import torch
from typing import Dict, List, Any, Optional, Tuple
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt
import seaborn as sns
import logging

class ModelEvaluator:
    """
    Comprehensive model evaluation utilities
    """
    
    def __init__(self, task_type: str = "regression"):
        self.task_type = task_type  # "regression" or "classification"
        self.logger = logging.getLogger(__name__)
    
    def evaluate_predictions(self, 
                           y_true: np.ndarray, 
                           y_pred: np.ndarray,
                           y_prob: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Evaluate predictions based on task type"""
        
        if self.task_type == "regression":
            return self._evaluate_regression(y_true, y_pred)
        elif self.task_type == "classification":
            return self._evaluate_classification(y_true, y_pred, y_prob)
        else:
            raise ValueError(f"Unsupported task type: {self.task_type}")
    
    def _evaluate_regression(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Evaluate regression predictions"""
        metrics = {}
        
        # Basic metrics
        metrics['mse'] = mean_squared_error(y_true, y_pred)
        metrics['rmse'] = np.sqrt(metrics['mse'])
        metrics['mae'] = mean_absolute_error(y_true, y_pred)
        metrics['r2'] = r2_score(y_true, y_pred)
        
        # Correlation metrics
        pearson_corr, pearson_p = pearsonr(y_true, y_pred)
        spearman_corr, spearman_p = spearmanr(y_true, y_pred)
        
        metrics['pearson_r'] = pearson_corr
        metrics['pearson_p_value'] = pearson_p
        metrics['spearman_r'] = spearman_corr
        metrics['spearman_p_value'] = spearman_p
        
        # Relative metrics
        metrics['mape'] = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
        
        # Residual analysis
        residuals = y_true - y_pred
        metrics['mean_residual'] = np.mean(residuals)
        metrics['std_residual'] = np.std(residuals)
        
        return metrics
    
    def _evaluate_classification(self, 
                                y_true: np.ndarray, 
                                y_pred: np.ndarray,
                                y_prob: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Evaluate classification predictions"""
        metrics = {}
        
        # Basic metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['precision'] = precision_score(y_true, y_pred, average='weighted')
        metrics['recall'] = recall_score(y_true, y_pred, average='weighted')
        metrics['f1'] = f1_score(y_true, y_pred, average='weighted')
        
        # ROC AUC if probabilities provided
        if y_prob is not None:
            try:
                if len(np.unique(y_true)) == 2:  # Binary classification
                    metrics['roc_auc'] = roc_auc_score(y_true, y_prob)
                else:  # Multi-class
                    metrics['roc_auc'] = roc_auc_score(y_true, y_prob, multi_class='ovr')
            except:
                self.logger.warning("Could not calculate ROC AUC")
        
        return metrics
    
    def cross_validate_model(self, 
                           model, 
                           X: np.ndarray, 
                           y: np.ndarray,
                           cv_folds: int = 5,
                           random_state: int = 42) -> Dict[str, Any]:
        """Perform cross-validation evaluation"""
        from sklearn.model_selection import KFold, StratifiedKFold
        
        if self.task_type == "classification":
            kfold = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
        else:
            kfold = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
        
        cv_scores = []
        fold_metrics = []
        
        for fold, (train_idx, val_idx) in enumerate(kfold.split(X, y)):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Train model (assuming model has fit and predict methods)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            
            # Evaluate fold
            fold_metrics_dict = self.evaluate_predictions(y_val, y_pred)
            fold_metrics.append(fold_metrics_dict)
            
            # Primary score for CV
            if self.task_type == "regression":
                primary_score = fold_metrics_dict['r2']
            else:
                primary_score = fold_metrics_dict['f1']
            
            cv_scores.append(primary_score)
            
            self.logger.info(f"Fold {fold+1}: {primary_score:.4f}")
        
        # Aggregate results
        cv_results = {
            'cv_scores': cv_scores,
            'mean_score': np.mean(cv_scores),
            'std_score': np.std(cv_scores),
            'fold_metrics': fold_metrics,
            'aggregated_metrics': self._aggregate_fold_metrics(fold_metrics)
        }
        
        return cv_results
    
    def _aggregate_fold_metrics(self, fold_metrics: List[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
        """Aggregate metrics across folds"""
        aggregated = {}
        
        if not fold_metrics:
            return aggregated
        
        # Get all metric names
        metric_names = fold_metrics[0].keys()
        
        for metric in metric_names:
            values = [fold[metric] for fold in fold_metrics if metric in fold]
            aggregated[metric] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values)
            }
        
        return aggregated
    
    def generate_evaluation_report(self, 
                                 y_true: np.ndarray,
                                 y_pred: np.ndarray,
                                 y_prob: Optional[np.ndarray] = None,
                                 model_name: str = "Model") -> Dict[str, Any]:
        """Generate comprehensive evaluation report"""
        
        metrics = self.evaluate_predictions(y_true, y_pred, y_prob)
        
        report = {
            'model_name': model_name,
            'task_type': self.task_type,
            'sample_size': len(y_true),
            'metrics': metrics,
            'performance_summary': self._generate_performance_summary(metrics)
        }
        
        # Add classification-specific information
        if self.task_type == "classification":
            report['confusion_matrix'] = confusion_matrix(y_true, y_pred).tolist()
            report['classification_report'] = classification_report(y_true, y_pred, output_dict=True)
        
        return report
    
    def _generate_performance_summary(self, metrics: Dict[str, float]) -> str:
        """Generate human-readable performance summary"""
        if self.task_type == "regression":
            r2 = metrics.get('r2', 0)
            rmse = metrics.get('rmse', float('inf'))
            
            if r2 > 0.8:
                performance = "Excellent"
            elif r2 > 0.6:
                performance = "Good"
            elif r2 > 0.4:
                performance = "Moderate"
            else:
                performance = "Poor"
            
            return f"{performance} performance (R² = {r2:.3f}, RMSE = {rmse:.3f})"
        
        else:  # classification
            accuracy = metrics.get('accuracy', 0)
            f1 = metrics.get('f1', 0)
            
            if accuracy > 0.9 and f1 > 0.9:
                performance = "Excellent"
            elif accuracy > 0.8 and f1 > 0.8:
                performance = "Good"
            elif accuracy > 0.7 and f1 > 0.7:
                performance = "Moderate"
            else:
                performance = "Poor"
            
            return f"{performance} performance (Accuracy = {accuracy:.3f}, F1 = {f1:.3f})"

class MetricsCalculator:
    """
    Calculate various chemistry-specific metrics
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def calculate_molecular_diversity(self, smiles_list: List[str]) -> Dict[str, float]:
        """Calculate molecular diversity metrics"""
        try:
            from rdkit import Chem
            from rdkit.Chem import DataStructs
            from rdkit import DataStructs
            from rdkit.Chem.Fingerprints import FingerprintMols
            
            # Convert SMILES to molecules
            mols = []
            for smiles in smiles_list:
                mol = Chem.MolFromSmiles(smiles)
                if mol is not None:
                    mols.append(mol)
            
            if len(mols) < 2:
                return {"error": "Need at least 2 valid molecules"}
            
            # Calculate fingerprints
            fps = [FingerprintMols.FingerprintMol(mol) for mol in mols]
            
            # Calculate pairwise similarities
            similarities = []
            for i in range(len(fps)):
                for j in range(i+1, len(fps)):
                    sim = DataStructs.TanimotoSimilarity(fps[i], fps[j])
                    similarities.append(sim)
            
            # Diversity metrics
            similarities = np.array(similarities)
            
            diversity_metrics = {
                'mean_similarity': np.mean(similarities),
                'diversity_index': 1 - np.mean(similarities),  # 1 - mean similarity
                'min_similarity': np.min(similarities),
                'max_similarity': np.max(similarities),
                'similarity_std': np.std(similarities),
                'num_molecules': len(mols),
                'num_comparisons': len(similarities)
            }
            
            return diversity_metrics
            
        except Exception as e:
            self.logger.error(f"Diversity calculation failed: {e}")
            return {"error": str(e)}
    
    def calculate_scaffold_diversity(self, smiles_list: List[str]) -> Dict[str, Any]:
        """Calculate scaffold diversity"""
        try:
            from rdkit import Chem
            from rdkit.Chem.Scaffolds import MurckoScaffold
            
            scaffolds = {}
            valid_molecules = 0
            
            for smiles in smiles_list:
                mol = Chem.MolFromSmiles(smiles)
                if mol is not None:
                    valid_molecules += 1
                    try:
                        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
                        scaffold_smiles = Chem.MolToSmiles(scaffold)
                        
                        if scaffold_smiles in scaffolds:
                            scaffolds[scaffold_smiles] += 1
                        else:
                            scaffolds[scaffold_smiles] = 1
                    except:
                        continue
            
            if not scaffolds:
                return {"error": "No valid scaffolds found"}
            
            # Calculate diversity metrics
            unique_scaffolds = len(scaffolds)
            scaffold_counts = list(scaffolds.values())
            
            diversity_metrics = {
                'unique_scaffolds': unique_scaffolds,
                'total_molecules': valid_molecules,
                'scaffold_diversity_ratio': unique_scaffolds / valid_molecules if valid_molecules > 0 else 0,
                'mean_molecules_per_scaffold': np.mean(scaffold_counts),
                'max_molecules_per_scaffold': max(scaffold_counts),
                'scaffold_distribution': dict(sorted(scaffolds.items(), key=lambda x: x[1], reverse=True)[:10])
            }
            
            return diversity_metrics
            
        except Exception as e:
            self.logger.error(f"Scaffold diversity calculation failed: {e}")
            return {"error": str(e)}
    
    def calculate_property_coverage(self, 
                                  predictions: List[float], 
                                  property_ranges: Dict[str, Tuple[float, float]]) -> Dict[str, float]:
        """Calculate property space coverage"""
        
        coverage_metrics = {}
        
        pred_array = np.array(predictions)
        pred_min, pred_max = np.min(pred_array), np.max(pred_array)
        pred_range = pred_max - pred_min
        
        for prop_name, (target_min, target_max) in property_ranges.items():
            target_range = target_max - target_min
            
            # Calculate overlap
            overlap_min = max(pred_min, target_min)
            overlap_max = min(pred_max, target_max)
            overlap_range = max(0, overlap_max - overlap_min)
            
            # Coverage as fraction of target range covered
            coverage = overlap_range / target_range if target_range > 0 else 0
            
            coverage_metrics[f"{prop_name}_coverage"] = coverage
        
        # Overall metrics
        coverage_metrics["prediction_range"] = pred_range
        coverage_metrics["mean_coverage"] = np.mean([v for k, v in coverage_metrics.items() if k.endswith("_coverage")])
        
        return coverage_metrics
    
    def calculate_enrichment_factor(self, 
                                  y_true: np.ndarray, 
                                  y_pred: np.ndarray,
                                  threshold: float,
                                  top_fraction: float = 0.1) -> float:
        """Calculate enrichment factor for virtual screening"""
        
        # Sort by predicted values (descending)
        sorted_indices = np.argsort(y_pred)[::-1]
        
        # Number of compounds in top fraction
        n_top = int(len(y_pred) * top_fraction)
        
        # Active compounds in top fraction
        top_indices = sorted_indices[:n_top]
        actives_in_top = np.sum(y_true[top_indices] >= threshold)
        
        # Total active compounds
        total_actives = np.sum(y_true >= threshold)
        
        # Calculate enrichment factor
        if total_actives == 0:
            return 0.0
        
        expected_actives = n_top * (total_actives / len(y_true))
        enrichment_factor = actives_in_top / expected_actives if expected_actives > 0 else 0.0
        
        return enrichment_factor

class BenchmarkEvaluator:
    """
    Evaluate models against standard benchmarks
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Standard benchmark datasets
        self.benchmarks = {
            "ESOL": {"task": "regression", "metric": "rmse", "target_range": (-11, 1)},
            "FreeSolv": {"task": "regression", "metric": "rmse", "target_range": (-25, 5)},
            "Lipophilicity": {"task": "regression", "metric": "rmse", "target_range": (-3, 5)},
            "BACE": {"task": "classification", "metric": "roc_auc"},
            "HIV": {"task": "classification", "metric": "roc_auc"},
            "Tox21": {"task": "classification", "metric": "roc_auc"}
        }
    
    def evaluate_against_benchmark(self, 
                                 model,
                                 benchmark_name: str,
                                 test_data: Tuple[np.ndarray, np.ndarray]) -> Dict[str, Any]:
        """Evaluate model against specific benchmark"""
        
        if benchmark_name not in self.benchmarks:
            raise ValueError(f"Unknown benchmark: {benchmark_name}")
        
        benchmark_info = self.benchmarks[benchmark_name]
        X_test, y_test = test_data
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Evaluate based on benchmark type
        evaluator = ModelEvaluator(task_type=benchmark_info["task"])
        metrics = evaluator.evaluate_predictions(y_test, y_pred)
        
        # Get primary metric for benchmark
        primary_metric = benchmark_info["metric"]
        primary_score = metrics.get(primary_metric, 0)
        
        result = {
            "benchmark": benchmark_name,
            "primary_metric": primary_metric,
            "primary_score": primary_score,
            "all_metrics": metrics,
            "benchmark_info": benchmark_info
        }
        
        return result
    
    def compare_with_literature(self, 
                              results: Dict[str, float],
                              benchmark_name: str) -> Dict[str, Any]:
        """Compare results with literature values"""
        
        # Literature baselines (approximate values)
        literature_baselines = {
            "ESOL": {"Random Forest": 0.58, "Graph Neural Network": 0.55, "Transformer": 0.52},
            "FreeSolv": {"Random Forest": 1.15, "Graph Neural Network": 1.05, "Transformer": 0.98},
            "Lipophilicity": {"Random Forest": 0.66, "Graph Neural Network": 0.62, "Transformer": 0.59},
            "BACE": {"Random Forest": 0.85, "Graph Neural Network": 0.87, "Transformer": 0.89},
            "HIV": {"Random Forest": 0.76, "Graph Neural Network": 0.78, "Transformer": 0.80}
        }
        
        if benchmark_name not in literature_baselines:
            return {"error": f"No literature baselines for {benchmark_name}"}
        
        baselines = literature_baselines[benchmark_name]
        primary_metric = self.benchmarks[benchmark_name]["metric"]
        
        if primary_metric not in results:
            return {"error": f"Primary metric {primary_metric} not found in results"}
        
        model_score = results[primary_metric]
        
        comparison = {
            "benchmark": benchmark_name,
            "model_score": model_score,
            "literature_baselines": baselines,
            "rankings": {}
        }
        
        # Calculate rankings
        all_scores = list(baselines.values()) + [model_score]
        
        # For metrics where lower is better (like RMSE)
        if primary_metric in ["rmse", "mae", "mse"]:
            sorted_scores = sorted(all_scores)
            rank = sorted_scores.index(model_score) + 1
        else:  # For metrics where higher is better
            sorted_scores = sorted(all_scores, reverse=True)
            rank = sorted_scores.index(model_score) + 1
        
        comparison["model_rank"] = rank
        comparison["total_methods"] = len(all_scores)
        comparison["percentile"] = (len(all_scores) - rank + 1) / len(all_scores) * 100
        
        return comparison