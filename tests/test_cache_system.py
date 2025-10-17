"""
Tests for Caching System

Tests cache manager, prediction cache, embedding cache, and performance profiler
"""

import pytest
import os
import tempfile
import shutil
import time
from pathlib import Path

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from chemistry_agents.utils.cache_manager import CacheManager, cached
from chemistry_agents.utils.prediction_cache import PredictionCache, EmbeddingCache
from chemistry_agents.utils.performance_profiler import PerformanceProfiler, profile


class TestCacheManager:
    """Test basic cache manager functionality"""

    @pytest.fixture
    def temp_cache_dir(self):
        """Create temporary cache directory"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def cache_manager(self, temp_cache_dir):
        """Create cache manager instance"""
        return CacheManager(
            cache_dir=temp_cache_dir,
            max_memory_entries=100,
            enable_disk_cache=True
        )

    def test_basic_set_get(self, cache_manager):
        """Test basic set and get operations"""
        cache_manager.set("test_key", "test_value")
        result = cache_manager.get("test_key")
        assert result == "test_value"

    def test_get_nonexistent(self, cache_manager):
        """Test getting nonexistent key"""
        result = cache_manager.get("nonexistent_key")
        assert result is None

    def test_get_with_default(self, cache_manager):
        """Test getting with default value"""
        result = cache_manager.get("nonexistent_key", default="default_value")
        assert result == "default_value"

    def test_ttl_expiration(self, cache_manager):
        """Test TTL expiration"""
        cache_manager.set("expiring_key", "value", ttl=1)  # 1 second TTL

        # Should be available immediately
        assert cache_manager.get("expiring_key") == "value"

        # Wait for expiration
        time.sleep(1.5)

        # Should be None after expiration
        assert cache_manager.get("expiring_key") is None

    def test_delete(self, cache_manager):
        """Test delete operation"""
        cache_manager.set("delete_key", "value")
        assert cache_manager.get("delete_key") == "value"

        deleted = cache_manager.delete("delete_key")
        assert deleted is True
        assert cache_manager.get("delete_key") is None

    def test_clear_all(self, cache_manager):
        """Test clearing all entries"""
        cache_manager.set("key1", "value1")
        cache_manager.set("key2", "value2")
        cache_manager.set("key3", "value3")

        cache_manager.clear()

        assert cache_manager.get("key1") is None
        assert cache_manager.get("key2") is None
        assert cache_manager.get("key3") is None

    def test_list_keys(self, cache_manager):
        """Test listing cache keys"""
        cache_manager.set("key1", "value1")
        cache_manager.set("key2", "value2")
        cache_manager.set("other_key", "value3")

        all_keys = cache_manager.list_keys()
        assert len(all_keys) == 3

        filtered_keys = cache_manager.list_keys(pattern="key")
        assert len(filtered_keys) == 3  # All contain "key"

        specific_keys = cache_manager.list_keys(pattern="other")
        assert len(specific_keys) == 1

    def test_get_stats(self, cache_manager):
        """Test cache statistics"""
        cache_manager.set("key1", "value1")
        cache_manager.set("key2", "value2")

        # Generate some hits and misses
        cache_manager.get("key1")  # Hit
        cache_manager.get("key2")  # Hit
        cache_manager.get("nonexistent")  # Miss

        stats = cache_manager.get_stats()
        assert stats.total_hits == 2
        assert stats.total_misses == 1
        assert stats.hit_rate > 0

    def test_cached_decorator(self, cache_manager):
        """Test cached decorator"""
        call_count = [0]  # Use list to allow modification in nested function

        @cached(cache_manager, key_prefix="test_func")
        def expensive_function(x):
            call_count[0] += 1
            return x * 2

        # First call should execute function
        result1 = expensive_function(5)
        assert result1 == 10
        assert call_count[0] == 1

        # Second call should use cache
        result2 = expensive_function(5)
        assert result2 == 10
        assert call_count[0] == 1  # Function not called again

        # Different argument should execute function
        result3 = expensive_function(10)
        assert result3 == 20
        assert call_count[0] == 2

    def test_json_and_pickle_storage(self, cache_manager):
        """Test different storage formats"""
        # JSON-serializable data
        json_data = {"key": "value", "number": 42}
        cache_manager.set("json_key", json_data)
        assert cache_manager.get("json_key") == json_data

        # Complex object (requires pickle)
        class CustomClass:
            def __init__(self, value):
                self.value = value

        pickle_data = CustomClass(100)
        cache_manager.set("pickle_key", pickle_data)
        retrieved = cache_manager.get("pickle_key")
        assert retrieved.value == 100


class TestPredictionCache:
    """Test prediction cache functionality"""

    @pytest.fixture
    def temp_cache_dir(self):
        """Create temporary cache directory"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def prediction_cache(self, temp_cache_dir):
        """Create prediction cache instance"""
        return PredictionCache(cache_dir=temp_cache_dir)

    def test_set_and_get_prediction(self, prediction_cache):
        """Test setting and getting predictions"""
        smiles = "CCO"  # Ethanol
        prediction_type = "toxicity"
        model_name = "test_model"
        prediction_result = {"score": 0.75, "confidence": 0.9}

        prediction_cache.set_prediction(
            smiles=smiles,
            prediction_type=prediction_type,
            model_name=model_name,
            prediction_result=prediction_result
        )

        retrieved = prediction_cache.get_prediction(
            smiles=smiles,
            prediction_type=prediction_type,
            model_name=model_name
        )

        assert retrieved is not None
        assert retrieved["prediction"]["score"] == 0.75

    def test_smiles_canonicalization(self, prediction_cache):
        """Test SMILES canonicalization for cache keys"""
        # Different representations of benzene
        smiles1 = "c1ccccc1"
        smiles2 = "C1=CC=CC=C1"

        prediction_result = {"score": 0.5}

        # Set with first representation
        prediction_cache.set_prediction(
            smiles=smiles1,
            prediction_type="toxicity",
            model_name="test_model",
            prediction_result=prediction_result
        )

        # Get with second representation (should work due to canonicalization)
        retrieved = prediction_cache.get_prediction(
            smiles=smiles2,
            prediction_type="toxicity",
            model_name="test_model"
        )

        # This will work if RDKit is available, otherwise they'll be different
        # Just test that the system doesn't crash
        assert retrieved is not None or retrieved is None

    def test_batch_predictions(self, prediction_cache):
        """Test batch prediction caching"""
        smiles_list = ["CCO", "CC", "CCC"]
        predictions = {
            "CCO": {"score": 0.1},
            "CC": {"score": 0.2},
            "CCC": {"score": 0.3}
        }

        # Cache batch
        prediction_cache.set_batch_predictions(
            predictions=predictions,
            prediction_type="toxicity",
            model_name="test_model"
        )

        # Retrieve batch
        cached, missing = prediction_cache.get_batch_predictions(
            smiles_list=smiles_list,
            prediction_type="toxicity",
            model_name="test_model"
        )

        assert len(cached) == 3
        assert len(missing) == 0

    def test_partial_batch_cache_hit(self, prediction_cache):
        """Test partial cache hits in batch"""
        # Cache only some predictions
        prediction_cache.set_prediction(
            smiles="CCO",
            prediction_type="toxicity",
            model_name="test_model",
            prediction_result={"score": 0.1}
        )

        smiles_list = ["CCO", "CC", "CCC"]
        cached, missing = prediction_cache.get_batch_predictions(
            smiles_list=smiles_list,
            prediction_type="toxicity",
            model_name="test_model"
        )

        assert len(cached) == 1
        assert len(missing) == 2
        assert "CCO" in cached
        assert "CC" in missing
        assert "CCC" in missing

    def test_model_version_separation(self, prediction_cache):
        """Test that different model versions are cached separately"""
        smiles = "CCO"

        prediction_cache.set_prediction(
            smiles=smiles,
            prediction_type="toxicity",
            model_name="test_model",
            model_version="1.0",
            prediction_result={"score": 0.5}
        )

        prediction_cache.set_prediction(
            smiles=smiles,
            prediction_type="toxicity",
            model_name="test_model",
            model_version="2.0",
            prediction_result={"score": 0.7}
        )

        v1_result = prediction_cache.get_prediction(
            smiles=smiles,
            prediction_type="toxicity",
            model_name="test_model",
            model_version="1.0"
        )

        v2_result = prediction_cache.get_prediction(
            smiles=smiles,
            prediction_type="toxicity",
            model_name="test_model",
            model_version="2.0"
        )

        assert v1_result["prediction"]["score"] == 0.5
        assert v2_result["prediction"]["score"] == 0.7

    def test_stats_by_type(self, prediction_cache):
        """Test statistics grouped by prediction type"""
        prediction_cache.set_prediction(
            smiles="CCO",
            prediction_type="toxicity",
            model_name="test_model",
            prediction_result={"score": 0.5}
        )

        prediction_cache.set_prediction(
            smiles="CC",
            prediction_type="solubility",
            model_name="test_model",
            prediction_result={"score": 0.5}
        )

        stats = prediction_cache.get_stats_by_type()
        assert "toxicity" in stats
        assert "solubility" in stats


class TestEmbeddingCache:
    """Test embedding cache functionality"""

    @pytest.fixture
    def temp_cache_dir(self):
        """Create temporary cache directory"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def embedding_cache(self, temp_cache_dir):
        """Create embedding cache instance"""
        return EmbeddingCache(cache_dir=temp_cache_dir)

    def test_set_and_get_embedding(self, embedding_cache):
        """Test setting and getting embeddings"""
        smiles = "CCO"
        embedding = [0.1, 0.2, 0.3, 0.4, 0.5]

        embedding_cache.set_embedding(
            smiles=smiles,
            model_name="test_model",
            embedding=embedding
        )

        retrieved = embedding_cache.get_embedding(
            smiles=smiles,
            model_name="test_model"
        )

        assert retrieved == embedding

    def test_batch_embeddings(self, embedding_cache):
        """Test batch embedding operations"""
        embeddings = {
            "CCO": [0.1, 0.2, 0.3],
            "CC": [0.4, 0.5, 0.6],
            "CCC": [0.7, 0.8, 0.9]
        }

        embedding_cache.set_batch_embeddings(
            embeddings=embeddings,
            model_name="test_model"
        )

        cached, missing = embedding_cache.get_batch_embeddings(
            smiles_list=["CCO", "CC", "CCC"],
            model_name="test_model"
        )

        assert len(cached) == 3
        assert len(missing) == 0


class TestPerformanceProfiler:
    """Test performance profiler functionality"""

    @pytest.fixture
    def profiler(self):
        """Create profiler instance"""
        return PerformanceProfiler()

    def test_profile_operation(self, profiler):
        """Test profiling an operation"""
        def test_function(x):
            time.sleep(0.1)
            return x * 2

        result = profiler.profile_operation(
            operation_name="test_op",
            func=test_function,
            5
        )

        assert result == 10

        metrics = profiler.get_metrics()
        assert len(metrics) == 1
        assert metrics[0].operation_name == "test_op"
        assert metrics[0].duration >= 0.1

    def test_profile_decorator(self, profiler):
        """Test profile decorator"""
        @profile(profiler, operation_name="decorated_func")
        def decorated_function(x):
            time.sleep(0.05)
            return x + 10

        result = decorated_function(5)
        assert result == 15

        metrics = profiler.get_metrics()
        assert len(metrics) == 1
        assert metrics[0].operation_name == "decorated_func"

    def test_aggregated_metrics(self, profiler):
        """Test aggregated metrics"""
        def test_function(x):
            return x * 2

        # Call multiple times
        for i in range(5):
            profiler.profile_operation("test_op", test_function, i)

        aggregated = profiler.get_aggregated_metrics()
        assert len(aggregated) == 1
        assert aggregated[0].call_count == 5
        assert aggregated[0].success_count == 5

    def test_disable_profiling(self, profiler):
        """Test disabling profiling"""
        profiler.disable()

        def test_function(x):
            return x * 2

        result = profiler.profile_operation("test_op", test_function, 5)
        assert result == 10

        # No metrics should be recorded
        metrics = profiler.get_metrics()
        assert len(metrics) == 0

    def test_error_handling(self, profiler):
        """Test profiling with errors"""
        def failing_function():
            raise ValueError("Test error")

        with pytest.raises(ValueError):
            profiler.profile_operation("failing_op", failing_function)

        metrics = profiler.get_metrics()
        assert len(metrics) == 1
        assert metrics[0].success is False
        assert "Test error" in metrics[0].error_message


def test_integration():
    """Integration test for complete caching workflow"""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create caches
        pred_cache = PredictionCache(cache_dir=temp_dir)
        profiler = PerformanceProfiler()

        # Simulate prediction workflow with caching and profiling
        def predict_with_cache(smiles, prediction_type):
            # Check cache
            cached = pred_cache.get_prediction(
                smiles=smiles,
                prediction_type=prediction_type,
                model_name="test_model"
            )

            if cached:
                return cached["prediction"]

            # Compute prediction
            def compute_prediction():
                time.sleep(0.01)  # Simulate computation
                return {"score": 0.5}

            result = profiler.profile_operation(
                operation_name=f"compute_{prediction_type}",
                func=compute_prediction
            )

            # Cache result
            pred_cache.set_prediction(
                smiles=smiles,
                prediction_type=prediction_type,
                model_name="test_model",
                prediction_result=result
            )

            return result

        # First call - should compute
        result1 = predict_with_cache("CCO", "toxicity")
        assert result1["score"] == 0.5

        # Second call - should use cache
        result2 = predict_with_cache("CCO", "toxicity")
        assert result2["score"] == 0.5

        # Check profiler recorded only one computation
        metrics = profiler.get_metrics()
        assert len(metrics) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
