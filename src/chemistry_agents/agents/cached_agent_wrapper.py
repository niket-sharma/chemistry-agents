"""
Cached Agent Wrapper

Wraps existing chemistry agents with caching capabilities for improved performance
"""

import logging
from typing import Optional, Any, Dict, List
import time

from .base_agent import BaseAgent, AgentConfig, PredictionResult
from ..utils.prediction_cache import PredictionCache, EmbeddingCache
from ..utils.performance_profiler import PerformanceProfiler, profile


class CachedAgentWrapper:
    """
    Wrapper that adds caching to any chemistry agent

    Features:
    - Automatic prediction caching
    - Embedding caching
    - Performance profiling
    - Cache statistics

    Usage:
        base_agent = create_intelligent_chemberta_agent()
        cached_agent = CachedAgentWrapper(base_agent, enable_cache=True)

        # This will use cache if available
        result = cached_agent.predict("c1ccccc1", "toxicity")
    """

    def __init__(
        self,
        agent: BaseAgent,
        enable_cache: bool = True,
        enable_profiling: bool = True,
        cache_dir: str = "cache",
        model_version: str = "1.0",
        logger: Optional[logging.Logger] = None
    ):
        self.agent = agent
        self.enable_cache = enable_cache
        self.enable_profiling = enable_profiling
        self.model_version = model_version
        self.logger = logger or logging.getLogger(__name__)

        # Initialize caches
        if enable_cache:
            self.prediction_cache = PredictionCache(
                cache_dir=f"{cache_dir}/predictions",
                logger=self.logger
            )
            self.embedding_cache = EmbeddingCache(
                cache_dir=f"{cache_dir}/embeddings",
                logger=self.logger
            )
        else:
            self.prediction_cache = None
            self.embedding_cache = None

        # Initialize profiler
        if enable_profiling:
            self.profiler = PerformanceProfiler(logger=self.logger)
        else:
            self.profiler = None

        self.logger.info(
            f"Initialized CachedAgentWrapper (cache={enable_cache}, profiling={enable_profiling})"
        )

    def predict(
        self,
        smiles: str,
        prediction_type: str,
        force_refresh: bool = False
    ) -> Any:
        """
        Make prediction with caching

        Args:
            smiles: SMILES string
            prediction_type: Type of prediction
            force_refresh: If True, bypass cache and recompute

        Returns:
            Prediction result
        """
        model_name = self._get_model_name()

        # Try cache first
        if self.enable_cache and not force_refresh:
            cached_result = self.prediction_cache.get_prediction(
                smiles=smiles,
                prediction_type=prediction_type,
                model_name=model_name,
                model_version=self.model_version
            )

            if cached_result is not None:
                self.logger.debug(f"Cache hit for {prediction_type} prediction of {smiles}")
                return cached_result["prediction"]

        # Cache miss or force refresh - compute prediction
        if self.enable_profiling:
            result = self.profiler.profile_operation(
                operation_name=f"predict_{prediction_type}",
                func=self._compute_prediction,
                smiles=smiles,
                prediction_type=prediction_type,
                metadata={"smiles": smiles, "type": prediction_type}
            )
        else:
            result = self._compute_prediction(smiles, prediction_type)

        # Store in cache
        if self.enable_cache:
            self.prediction_cache.set_prediction(
                smiles=smiles,
                prediction_type=prediction_type,
                model_name=model_name,
                prediction_result=result,
                model_version=self.model_version
            )

        return result

    def predict_batch(
        self,
        smiles_list: List[str],
        prediction_type: str,
        force_refresh: bool = False
    ) -> Dict[str, Any]:
        """
        Batch prediction with caching

        Args:
            smiles_list: List of SMILES strings
            prediction_type: Type of prediction
            force_refresh: If True, bypass cache and recompute

        Returns:
            Dict mapping SMILES to predictions
        """
        model_name = self._get_model_name()
        results = {}

        # Check cache for all molecules
        if self.enable_cache and not force_refresh:
            cached_results, missing_smiles = self.prediction_cache.get_batch_predictions(
                smiles_list=smiles_list,
                prediction_type=prediction_type,
                model_name=model_name,
                model_version=self.model_version
            )

            # Add cached results
            for smiles, cached_result in cached_results.items():
                results[smiles] = cached_result["prediction"]
        else:
            missing_smiles = smiles_list

        # Compute predictions for missing molecules
        if missing_smiles:
            if self.enable_profiling:
                new_results = self.profiler.profile_operation(
                    operation_name=f"predict_batch_{prediction_type}",
                    func=self._compute_batch_predictions,
                    smiles_list=missing_smiles,
                    prediction_type=prediction_type,
                    metadata={"batch_size": len(missing_smiles), "type": prediction_type}
                )
            else:
                new_results = self._compute_batch_predictions(missing_smiles, prediction_type)

            # Add new results
            results.update(new_results)

            # Store in cache
            if self.enable_cache:
                self.prediction_cache.set_batch_predictions(
                    predictions=new_results,
                    prediction_type=prediction_type,
                    model_name=model_name,
                    model_version=self.model_version
                )

        return results

    def _compute_prediction(self, smiles: str, prediction_type: str) -> Any:
        """Compute prediction using wrapped agent"""
        # This is a simplified version - adapt based on your agent's API
        if hasattr(self.agent, 'predict'):
            return self.agent.predict(smiles)
        elif hasattr(self.agent, 'chat'):
            # For conversational agents
            response = self.agent.chat(f"Analyze {smiles} for {prediction_type}")
            return response
        else:
            raise NotImplementedError(f"Agent does not support prediction")

    def _compute_batch_predictions(
        self,
        smiles_list: List[str],
        prediction_type: str
    ) -> Dict[str, Any]:
        """Compute batch predictions"""
        results = {}

        for smiles in smiles_list:
            try:
                result = self._compute_prediction(smiles, prediction_type)
                results[smiles] = result
            except Exception as e:
                self.logger.error(f"Prediction failed for {smiles}: {e}")
                results[smiles] = None

        return results

    def _get_model_name(self) -> str:
        """Get model name from wrapped agent"""
        if hasattr(self.agent, 'model_name'):
            return self.agent.model_name
        elif hasattr(self.agent, '__class__'):
            return self.agent.__class__.__name__
        else:
            return "unknown_model"

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        if not self.enable_cache:
            return {"cache_enabled": False}

        prediction_stats = self.prediction_cache.get_stats()
        embedding_stats = self.embedding_cache.get_stats()

        return {
            "cache_enabled": True,
            "predictions": {
                "total_entries": prediction_stats.total_entries,
                "hit_rate": f"{prediction_stats.hit_rate:.1f}%",
                "total_size_mb": prediction_stats.total_size_bytes / 1024 / 1024,
                "by_type": self.prediction_cache.get_stats_by_type()
            },
            "embeddings": {
                "total_entries": embedding_stats.total_entries,
                "hit_rate": f"{embedding_stats.hit_rate:.1f}%",
                "total_size_mb": embedding_stats.total_size_bytes / 1024 / 1024
            }
        }

    def get_performance_report(self):
        """Get performance profiling report"""
        if not self.enable_profiling:
            return "Profiling not enabled"

        self.profiler.print_report()

    def clear_cache(self, prediction_type: Optional[str] = None):
        """
        Clear cache

        Args:
            prediction_type: If specified, only clear this type
        """
        if self.enable_cache:
            self.prediction_cache.clear(prediction_type=prediction_type)
            if prediction_type is None:
                self.embedding_cache.clear()
            self.logger.info(f"Cache cleared for {prediction_type or 'all types'}")

    def save_performance_report(self, output_file: str):
        """Save performance report to file"""
        if self.enable_profiling:
            self.profiler.save_report(output_file)
            self.logger.info(f"Performance report saved to {output_file}")

    def __getattr__(self, name):
        """
        Delegate attribute access to wrapped agent

        This allows the wrapper to be used as a drop-in replacement
        """
        return getattr(self.agent, name)


def create_cached_agent(
    agent: BaseAgent,
    enable_cache: bool = True,
    enable_profiling: bool = True,
    cache_dir: str = "cache",
    model_version: str = "1.0"
) -> CachedAgentWrapper:
    """
    Convenience function to create a cached agent wrapper

    Args:
        agent: Base agent to wrap
        enable_cache: Enable caching
        enable_profiling: Enable performance profiling
        cache_dir: Cache directory
        model_version: Model version for cache keys

    Returns:
        Cached agent wrapper
    """
    return CachedAgentWrapper(
        agent=agent,
        enable_cache=enable_cache,
        enable_profiling=enable_profiling,
        cache_dir=cache_dir,
        model_version=model_version
    )
