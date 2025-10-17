#!/usr/bin/env python3
"""
Caching System Demo

Demonstrates the performance benefits of caching for chemistry predictions
"""

import sys
import os
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from chemistry_agents.utils.prediction_cache import PredictionCache, EmbeddingCache
from chemistry_agents.utils.performance_profiler import PerformanceProfiler, BatchProfiler


def simulate_expensive_prediction(smiles: str, delay: float = 0.1) -> dict:
    """Simulate an expensive prediction computation"""
    time.sleep(delay)  # Simulate computation time
    return {
        "smiles": smiles,
        "score": hash(smiles) % 100 / 100,  # Deterministic "random" score
        "confidence": 0.85
    }


def demo_basic_caching():
    """Demo 1: Basic caching with performance comparison"""
    print("\n" + "=" * 80)
    print("DEMO 1: Basic Caching Performance")
    print("=" * 80)

    molecules = ["CCO", "CC", "CCC", "c1ccccc1", "CC(C)O"]
    cache = PredictionCache(cache_dir="cache/demo")

    print("\n1. WITHOUT CACHE - Computing predictions...")
    start = time.time()
    for smiles in molecules:
        result = simulate_expensive_prediction(smiles)
        print(f"  Predicted {smiles}: {result['score']:.3f}")
    no_cache_time = time.time() - start
    print(f"  Time: {no_cache_time:.2f}s")

    print("\n2. FIRST RUN WITH CACHE - Computing and caching...")
    start = time.time()
    for smiles in molecules:
        cached = cache.get_prediction(smiles, "toxicity", "demo_model")
        if cached:
            result = cached["prediction"]
            print(f"  Cache HIT: {smiles}")
        else:
            result = simulate_expensive_prediction(smiles)
            cache.set_prediction(smiles, "toxicity", "demo_model", result)
            print(f"  Cache MISS: {smiles} - cached for next time")
    first_run_time = time.time() - start
    print(f"  Time: {first_run_time:.2f}s")

    print("\n3. SECOND RUN WITH CACHE - Using cached predictions...")
    start = time.time()
    for smiles in molecules:
        cached = cache.get_prediction(smiles, "toxicity", "demo_model")
        if cached:
            result = cached["prediction"]
            print(f"  Cache HIT: {smiles}: {result['score']:.3f}")
    cached_time = time.time() - start
    print(f"  Time: {cached_time:.2f}s")

    print(f"\n📊 PERFORMANCE IMPROVEMENT:")
    print(f"  Speedup: {no_cache_time/cached_time:.1f}x faster with cache!")
    print(f"  Time saved: {(no_cache_time - cached_time):.2f}s")


def demo_batch_caching():
    """Demo 2: Batch prediction with partial cache hits"""
    print("\n" + "=" * 80)
    print("DEMO 2: Batch Prediction with Partial Cache Hits")
    print("=" * 80)

    cache = PredictionCache(cache_dir="cache/demo")

    # First batch
    batch1 = ["CCO", "CC", "CCC", "c1ccccc1", "CC(C)O"]
    print(f"\n1. Processing first batch ({len(batch1)} molecules)...")

    cached, missing = cache.get_batch_predictions(
        batch1, "toxicity", "demo_model"
    )
    print(f"  Cached: {len(cached)} molecules")
    print(f"  Missing: {len(missing)} molecules")

    # Compute missing
    for smiles in missing:
        result = simulate_expensive_prediction(smiles, delay=0.05)
        cache.set_prediction(smiles, "toxicity", "demo_model", result)
    print(f"  Computed and cached {len(missing)} predictions")

    # Second batch with overlap
    batch2 = ["CCO", "CC", "CCCC", "c1cccnc1", "CCN"]
    print(f"\n2. Processing second batch ({len(batch2)} molecules)...")
    print(f"  (overlaps with {sum(1 for s in batch2 if s in batch1)} from first batch)")

    start = time.time()
    cached, missing = cache.get_batch_predictions(
        batch2, "toxicity", "demo_model"
    )
    print(f"  Cached: {len(cached)} molecules (instant!)")
    print(f"  Missing: {len(missing)} molecules (need to compute)")

    # Compute missing
    for smiles in missing:
        result = simulate_expensive_prediction(smiles, delay=0.05)
        cache.set_prediction(smiles, "toxicity", "demo_model", result)

    batch_time = time.time() - start
    print(f"  Total time: {batch_time:.2f}s")

    cache_hit_rate = len(cached) / len(batch2) * 100
    print(f"\n📊 Cache hit rate: {cache_hit_rate:.1f}%")


def demo_performance_profiling():
    """Demo 3: Performance profiling"""
    print("\n" + "=" * 80)
    print("DEMO 3: Performance Profiling")
    print("=" * 80)

    profiler = PerformanceProfiler()
    molecules = ["CCO", "CC", "CCC", "c1ccccc1", "CC(C)O"]

    print("\nProfiling different operations...")

    # Simulate different operations
    for smiles in molecules:
        profiler.profile_operation(
            operation_name="toxicity_prediction",
            func=simulate_expensive_prediction,
            smiles=smiles,
            delay=0.08
        )

        profiler.profile_operation(
            operation_name="solubility_prediction",
            func=simulate_expensive_prediction,
            smiles=smiles,
            delay=0.05
        )

        profiler.profile_operation(
            operation_name="bioactivity_prediction",
            func=simulate_expensive_prediction,
            smiles=smiles,
            delay=0.12
        )

    print("\n📊 Performance Report:")
    profiler.print_report()


def demo_batch_profiling():
    """Demo 4: Batch processing with profiling"""
    print("\n" + "=" * 80)
    print("DEMO 4: Batch Processing with Profiling")
    print("=" * 80)

    batch_profiler = BatchProfiler()

    def process_batch(molecules, delay):
        results = []
        for smiles in molecules:
            result = simulate_expensive_prediction(smiles, delay=delay)
            results.append(result)
        return results

    # Test different batch sizes
    batch_sizes = [5, 10, 20]

    print("\nTesting different batch sizes...\n")

    for size in batch_sizes:
        molecules = [f"C{i}" for i in range(size)]

        batch_profiler.profile_batch(
            operation_name=f"batch_size_{size}",
            batch_size=size,
            func=process_batch,
            molecules=molecules,
            delay=0.01
        )

    print("\n📊 Batch Processing Summary:")
    batch_profiler.print_summary()


def demo_cache_statistics():
    """Demo 5: Cache statistics and management"""
    print("\n" + "=" * 80)
    print("DEMO 5: Cache Statistics")
    print("=" * 80)

    cache = PredictionCache(cache_dir="cache/demo")

    # Add some predictions
    molecules = [f"C{'C' * i}" for i in range(1, 11)]
    for smiles in molecules:
        for pred_type in ["toxicity", "solubility"]:
            result = simulate_expensive_prediction(smiles, delay=0)
            cache.set_prediction(smiles, pred_type, "demo_model", result)

    print("\nCache contents:")
    print(f"  Total predictions cached: {len(molecules) * 2}")

    # Get statistics
    stats = cache.get_stats()
    print(f"\n📊 Cache Statistics:")
    print(f"  Total entries: {stats.total_entries}")
    print(f"  Hit rate: {stats.hit_rate:.1f}%")
    print(f"  Total hits: {stats.total_hits}")
    print(f"  Total misses: {stats.total_misses}")

    # Stats by type
    type_stats = cache.get_stats_by_type()
    print(f"\n  By prediction type:")
    for pred_type, count in type_stats.items():
        print(f"    {pred_type}: {count} entries")


def demo_embedding_cache():
    """Demo 6: Embedding cache"""
    print("\n" + "=" * 80)
    print("DEMO 6: Embedding Cache")
    print("=" * 80)

    emb_cache = EmbeddingCache(cache_dir="cache/demo_embeddings")

    def simulate_embedding_computation(smiles: str) -> list:
        """Simulate expensive embedding computation"""
        time.sleep(0.1)
        # Generate fake embedding
        return [hash(smiles + str(i)) % 100 / 100 for i in range(768)]

    molecules = ["CCO", "CC", "CCC"]

    print("\n1. Computing embeddings (expensive)...")
    start = time.time()
    for smiles in molecules:
        embedding = simulate_embedding_computation(smiles)
        emb_cache.set_embedding(smiles, "ChemBERTa", embedding)
        print(f"  Computed embedding for {smiles}")
    compute_time = time.time() - start
    print(f"  Time: {compute_time:.2f}s")

    print("\n2. Loading embeddings from cache...")
    start = time.time()
    for smiles in molecules:
        embedding = emb_cache.get_embedding(smiles, "ChemBERTa")
        print(f"  Loaded embedding for {smiles} (768 dimensions)")
    cache_time = time.time() - start
    print(f"  Time: {cache_time:.2f}s")

    print(f"\n📊 Speedup: {compute_time/cache_time:.1f}x faster with cache!")


def main():
    """Run all demos"""
    print("\n" + "=" * 80)
    print("CHEMISTRY AGENTS - CACHING SYSTEM DEMO")
    print("=" * 80)
    print("\nThis demo shows how caching dramatically improves performance")
    print("for molecular property predictions.\n")

    # Run demos
    demo_basic_caching()
    demo_batch_caching()
    demo_performance_profiling()
    demo_batch_profiling()
    demo_cache_statistics()
    demo_embedding_cache()

    print("\n" + "=" * 80)
    print("DEMO COMPLETE")
    print("=" * 80)
    print("\n✅ Key Takeaways:")
    print("  1. Caching provides 10-100x speedup for repeated predictions")
    print("  2. Batch operations maximize cache efficiency")
    print("  3. Performance profiling identifies bottlenecks")
    print("  4. Embedding cache is crucial for ChemBERT models")
    print("\n💡 Next Steps:")
    print("  - Explore: CACHING_GUIDE.md for detailed documentation")
    print("  - Try: python cache_cli.py stats (to view your cache)")
    print("  - Integrate: Wrap your agents with CachedAgentWrapper")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
