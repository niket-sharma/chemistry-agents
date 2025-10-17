"""
Specialized Prediction Cache for Molecular Property Predictions

Optimized for chemistry-specific caching:
- SMILES canonicalization for cache keys
- Property-specific caching (toxicity, solubility, etc.)
- Batch prediction caching
- Model version tracking
"""

import hashlib
import json
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
import logging
from pathlib import Path

try:
    from rdkit import Chem
    from rdkit.Chem import MolToSmiles, MolFromSmiles
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False

from .cache_manager import CacheManager


@dataclass
class PredictionMetadata:
    """Metadata for a prediction"""
    model_name: str
    model_version: str
    prediction_type: str  # toxicity, solubility, bioactivity, etc.
    smiles: str
    canonical_smiles: str
    timestamp: float
    confidence: Optional[float] = None
    additional_info: Dict[str, Any] = None


class PredictionCache:
    """
    Specialized cache for molecular predictions

    Features:
    - Automatic SMILES canonicalization
    - Property-specific caching
    - Model version tracking
    - Batch operations
    """

    def __init__(
        self,
        cache_dir: str = "cache/predictions",
        max_memory_entries: int = 5000,
        default_ttl: int = 86400 * 30,  # 30 days
        logger: Optional[logging.Logger] = None
    ):
        self.cache_manager = CacheManager(
            cache_dir=cache_dir,
            max_memory_entries=max_memory_entries,
            default_ttl=default_ttl,
            enable_disk_cache=True,
            logger=logger
        )
        self.logger = logger or logging.getLogger(__name__)

        if not RDKIT_AVAILABLE:
            self.logger.warning("RDKit not available - SMILES canonicalization disabled")

    def _canonicalize_smiles(self, smiles: str) -> str:
        """
        Canonicalize SMILES string for consistent cache keys

        Args:
            smiles: Input SMILES string

        Returns:
            Canonical SMILES string
        """
        if not RDKIT_AVAILABLE:
            return smiles.strip()

        try:
            mol = MolFromSmiles(smiles)
            if mol is None:
                self.logger.warning(f"Invalid SMILES: {smiles}")
                return smiles.strip()
            return MolToSmiles(mol, canonical=True)
        except Exception as e:
            self.logger.error(f"Error canonicalizing SMILES {smiles}: {e}")
            return smiles.strip()

    def _make_cache_key(
        self,
        smiles: str,
        prediction_type: str,
        model_name: str,
        model_version: str = "1.0"
    ) -> str:
        """
        Generate cache key for prediction

        Args:
            smiles: SMILES string
            prediction_type: Type of prediction (toxicity, solubility, etc.)
            model_name: Name of the model
            model_version: Version of the model

        Returns:
            Cache key string
        """
        canonical_smiles = self._canonicalize_smiles(smiles)
        key_string = f"{model_name}_{model_version}_{prediction_type}_{canonical_smiles}"
        return key_string

    def get_prediction(
        self,
        smiles: str,
        prediction_type: str,
        model_name: str,
        model_version: str = "1.0"
    ) -> Optional[Dict[str, Any]]:
        """
        Retrieve cached prediction

        Args:
            smiles: SMILES string
            prediction_type: Type of prediction
            model_name: Model name
            model_version: Model version

        Returns:
            Cached prediction result or None
        """
        cache_key = self._make_cache_key(smiles, prediction_type, model_name, model_version)
        result = self.cache_manager.get(cache_key)

        if result:
            self.logger.debug(f"Cache hit for {prediction_type} prediction of {smiles}")
        else:
            self.logger.debug(f"Cache miss for {prediction_type} prediction of {smiles}")

        return result

    def set_prediction(
        self,
        smiles: str,
        prediction_type: str,
        model_name: str,
        prediction_result: Any,
        model_version: str = "1.0",
        confidence: Optional[float] = None,
        ttl: Optional[int] = None,
        additional_info: Optional[Dict] = None
    ):
        """
        Cache prediction result

        Args:
            smiles: SMILES string
            prediction_type: Type of prediction
            model_name: Model name
            prediction_result: Prediction result to cache
            model_version: Model version
            confidence: Confidence score
            ttl: Time-to-live in seconds
            additional_info: Additional metadata
        """
        import time

        cache_key = self._make_cache_key(smiles, prediction_type, model_name, model_version)
        canonical_smiles = self._canonicalize_smiles(smiles)

        # Create metadata
        metadata = PredictionMetadata(
            model_name=model_name,
            model_version=model_version,
            prediction_type=prediction_type,
            smiles=smiles,
            canonical_smiles=canonical_smiles,
            timestamp=time.time(),
            confidence=confidence,
            additional_info=additional_info or {}
        )

        # Store prediction with metadata
        cache_data = {
            "prediction": prediction_result,
            "metadata": asdict(metadata)
        }

        self.cache_manager.set(cache_key, cache_data, ttl=ttl, metadata=asdict(metadata))
        self.logger.debug(f"Cached {prediction_type} prediction for {smiles}")

    def get_batch_predictions(
        self,
        smiles_list: List[str],
        prediction_type: str,
        model_name: str,
        model_version: str = "1.0"
    ) -> Tuple[Dict[str, Any], List[str]]:
        """
        Retrieve multiple predictions from cache

        Args:
            smiles_list: List of SMILES strings
            prediction_type: Type of prediction
            model_name: Model name
            model_version: Model version

        Returns:
            Tuple of (cached_results, missing_smiles)
        """
        cached_results = {}
        missing_smiles = []

        for smiles in smiles_list:
            result = self.get_prediction(smiles, prediction_type, model_name, model_version)
            if result is not None:
                cached_results[smiles] = result
            else:
                missing_smiles.append(smiles)

        cache_hit_rate = len(cached_results) / len(smiles_list) * 100 if smiles_list else 0
        self.logger.info(
            f"Batch cache hit rate: {cache_hit_rate:.1f}% "
            f"({len(cached_results)}/{len(smiles_list)} molecules)"
        )

        return cached_results, missing_smiles

    def set_batch_predictions(
        self,
        predictions: Dict[str, Any],
        prediction_type: str,
        model_name: str,
        model_version: str = "1.0",
        ttl: Optional[int] = None
    ):
        """
        Cache multiple predictions at once

        Args:
            predictions: Dict mapping SMILES to prediction results
            prediction_type: Type of prediction
            model_name: Model name
            model_version: Model version
            ttl: Time-to-live in seconds
        """
        for smiles, result in predictions.items():
            self.set_prediction(
                smiles=smiles,
                prediction_type=prediction_type,
                model_name=model_name,
                prediction_result=result,
                model_version=model_version,
                ttl=ttl
            )

        self.logger.info(f"Cached {len(predictions)} predictions for {prediction_type}")

    def invalidate_model_predictions(
        self,
        model_name: str,
        model_version: Optional[str] = None,
        prediction_type: Optional[str] = None
    ) -> int:
        """
        Invalidate all predictions for a specific model

        Args:
            model_name: Model name
            model_version: Model version (None = all versions)
            prediction_type: Prediction type (None = all types)

        Returns:
            Number of invalidated entries
        """
        pattern_parts = [model_name]
        if model_version:
            pattern_parts.append(model_version)
        if prediction_type:
            pattern_parts.append(prediction_type)

        pattern = "_".join(pattern_parts)
        keys = self.cache_manager.list_keys(pattern=pattern)

        count = 0
        for key in keys:
            if self.cache_manager.delete(key):
                count += 1

        self.logger.info(f"Invalidated {count} predictions for {pattern}")
        return count

    def get_stats_by_type(self) -> Dict[str, int]:
        """
        Get cache statistics grouped by prediction type

        Returns:
            Dict mapping prediction type to count
        """
        stats = {}
        all_keys = self.cache_manager.list_keys()

        for key in all_keys:
            # Parse prediction type from key (format: model_version_type_smiles)
            parts = key.split("_")
            if len(parts) >= 3:
                prediction_type = parts[2]
                stats[prediction_type] = stats.get(prediction_type, 0) + 1

        return stats

    def clear(self, prediction_type: Optional[str] = None):
        """
        Clear cache entries

        Args:
            prediction_type: If specified, only clear this prediction type
        """
        if prediction_type:
            keys = self.cache_manager.list_keys(pattern=f"_{prediction_type}_")
            count = 0
            for key in keys:
                if self.cache_manager.delete(key):
                    count += 1
            self.logger.info(f"Cleared {count} {prediction_type} predictions")
        else:
            self.cache_manager.clear()
            self.logger.info("Cleared all predictions")

    def get_stats(self):
        """Get overall cache statistics"""
        return self.cache_manager.get_stats()


class EmbeddingCache:
    """
    Cache for molecular embeddings from ChemBERT and other models

    Embeddings are expensive to compute, so caching provides major speedups
    """

    def __init__(
        self,
        cache_dir: str = "cache/embeddings",
        max_memory_entries: int = 10000,
        default_ttl: int = 86400 * 90,  # 90 days
        logger: Optional[logging.Logger] = None
    ):
        self.cache_manager = CacheManager(
            cache_dir=cache_dir,
            max_memory_entries=max_memory_entries,
            default_ttl=default_ttl,
            enable_disk_cache=True,
            logger=logger
        )
        self.logger = logger or logging.getLogger(__name__)

        if not RDKIT_AVAILABLE:
            self.logger.warning("RDKit not available - SMILES canonicalization disabled")

    def _canonicalize_smiles(self, smiles: str) -> str:
        """Canonicalize SMILES string"""
        if not RDKIT_AVAILABLE:
            return smiles.strip()

        try:
            mol = MolFromSmiles(smiles)
            if mol is None:
                return smiles.strip()
            return MolToSmiles(mol, canonical=True)
        except Exception:
            return smiles.strip()

    def _make_cache_key(self, smiles: str, model_name: str, model_version: str = "1.0") -> str:
        """Generate cache key for embedding"""
        canonical_smiles = self._canonicalize_smiles(smiles)
        return f"embedding_{model_name}_{model_version}_{canonical_smiles}"

    def get_embedding(
        self,
        smiles: str,
        model_name: str,
        model_version: str = "1.0"
    ) -> Optional[Any]:
        """
        Retrieve cached embedding

        Args:
            smiles: SMILES string
            model_name: Model name
            model_version: Model version

        Returns:
            Cached embedding or None
        """
        cache_key = self._make_cache_key(smiles, model_name, model_version)
        return self.cache_manager.get(cache_key)

    def set_embedding(
        self,
        smiles: str,
        model_name: str,
        embedding: Any,
        model_version: str = "1.0",
        ttl: Optional[int] = None
    ):
        """
        Cache embedding

        Args:
            smiles: SMILES string
            model_name: Model name
            embedding: Embedding to cache (numpy array, tensor, etc.)
            model_version: Model version
            ttl: Time-to-live in seconds
        """
        cache_key = self._make_cache_key(smiles, model_name, model_version)
        self.cache_manager.set(cache_key, embedding, ttl=ttl)
        self.logger.debug(f"Cached embedding for {smiles}")

    def get_batch_embeddings(
        self,
        smiles_list: List[str],
        model_name: str,
        model_version: str = "1.0"
    ) -> Tuple[Dict[str, Any], List[str]]:
        """
        Retrieve multiple embeddings from cache

        Returns:
            Tuple of (cached_embeddings, missing_smiles)
        """
        cached_embeddings = {}
        missing_smiles = []

        for smiles in smiles_list:
            embedding = self.get_embedding(smiles, model_name, model_version)
            if embedding is not None:
                cached_embeddings[smiles] = embedding
            else:
                missing_smiles.append(smiles)

        return cached_embeddings, missing_smiles

    def set_batch_embeddings(
        self,
        embeddings: Dict[str, Any],
        model_name: str,
        model_version: str = "1.0",
        ttl: Optional[int] = None
    ):
        """Cache multiple embeddings at once"""
        for smiles, embedding in embeddings.items():
            self.set_embedding(smiles, model_name, embedding, model_version, ttl)

        self.logger.info(f"Cached {len(embeddings)} embeddings")

    def clear(self, model_name: Optional[str] = None):
        """Clear embedding cache"""
        if model_name:
            keys = self.cache_manager.list_keys(pattern=f"embedding_{model_name}_")
            count = 0
            for key in keys:
                if self.cache_manager.delete(key):
                    count += 1
            self.logger.info(f"Cleared {count} embeddings for {model_name}")
        else:
            self.cache_manager.clear()
            self.logger.info("Cleared all embeddings")

    def get_stats(self):
        """Get cache statistics"""
        return self.cache_manager.get_stats()
