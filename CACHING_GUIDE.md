# Caching & Performance Optimization Guide

Comprehensive guide to using the caching and performance optimization features in Chemistry Agents.

## Table of Contents

- [Overview](#overview)
- [Quick Start](#quick-start)
- [Architecture](#architecture)
- [Usage Examples](#usage-examples)
- [Configuration](#configuration)
- [CLI Tools](#cli-tools)
- [Performance Tips](#performance-tips)
- [API Reference](#api-reference)

---

## Overview

The caching system provides significant performance improvements for chemistry agents by:

- **Caching molecular predictions** - Avoid recomputing predictions for the same molecules
- **Caching embeddings** - Store expensive ChemBERT embeddings for reuse
- **Performance profiling** - Identify bottlenecks and optimize workflows
- **Persistent storage** - Cache survives between runs using SQLite + file storage
- **Automatic SMILES canonicalization** - Recognize equivalent molecular representations

### Performance Gains

With caching enabled, you can expect:
- **10-100x speedup** for repeated predictions
- **50-200x speedup** for embedding reuse
- **Reduced memory usage** through intelligent cache eviction
- **Disk persistence** for long-term caching

---

## Quick Start

### 1. Basic Caching

```python
from chemistry_agents.utils.prediction_cache import PredictionCache

# Create cache
cache = PredictionCache(cache_dir="cache/predictions")

# Check cache before prediction
cached_result = cache.get_prediction(
    smiles="c1ccccc1",  # Benzene
    prediction_type="toxicity",
    model_name="ChemBERTa",
    model_version="1.0"
)

if cached_result:
    print("Cache hit!", cached_result)
else:
    # Compute prediction
    result = your_model.predict("c1ccccc1")

    # Store in cache
    cache.set_prediction(
        smiles="c1ccccc1",
        prediction_type="toxicity",
        model_name="ChemBERTa",
        prediction_result=result
    )
```

### 2. Wrap Existing Agent with Caching

```python
from chemistry_agents.agents.intelligent_chemberta_agent import create_intelligent_chemberta_agent
from chemistry_agents.agents.cached_agent_wrapper import create_cached_agent

# Create base agent
base_agent = create_intelligent_chemberta_agent()

# Wrap with caching
cached_agent = create_cached_agent(
    agent=base_agent,
    enable_cache=True,
    enable_profiling=True
)

# Use as normal - caching is automatic!
result = cached_agent.chat("Is benzene toxic?")

# View cache statistics
stats = cached_agent.get_cache_stats()
print(f"Cache hit rate: {stats['predictions']['hit_rate']}")
```

### 3. Performance Profiling

```python
from chemistry_agents.utils.performance_profiler import PerformanceProfiler, profile

profiler = PerformanceProfiler()

@profile(profiler, operation_name="predict_toxicity")
def predict_toxicity(smiles):
    # Your expensive computation
    return model.predict(smiles)

# Make predictions
for smiles in molecule_list:
    result = predict_toxicity(smiles)

# View performance report
profiler.print_report()

# Save report to file
profiler.save_report("performance_report.json")
```

---

## Architecture

### Components

1. **CacheManager** - Low-level cache with memory + disk storage
2. **PredictionCache** - Specialized cache for molecular predictions
3. **EmbeddingCache** - Cache for molecular embeddings
4. **PerformanceProfiler** - Tracks execution time and memory usage
5. **CachedAgentWrapper** - Wraps agents with automatic caching

### Cache Storage

```
cache/
├── predictions/
│   ├── cache.db           # SQLite database with metadata
│   ├── <hash>.json        # JSON-serializable predictions
│   └── <hash>.pickle      # Complex object predictions
├── embeddings/
│   ├── cache.db
│   └── <hash>.pickle      # Embedding vectors
└── reports/
    └── performance_*.json  # Performance reports
```

### Cache Key Format

Predictions: `{model_name}_{model_version}_{prediction_type}_{canonical_smiles}`

Example: `ChemBERTa_1.0_toxicity_c1ccccc1`

---

## Usage Examples

### Example 1: Batch Processing with Cache

```python
from chemistry_agents.utils.prediction_cache import PredictionCache

cache = PredictionCache()

# List of molecules to process
smiles_list = ["CCO", "CC(C)O", "c1ccccc1", "CC", "CCC"]

# Check which are already cached
cached_results, missing_smiles = cache.get_batch_predictions(
    smiles_list=smiles_list,
    prediction_type="toxicity",
    model_name="ChemBERTa"
)

print(f"Found {len(cached_results)} in cache")
print(f"Need to compute {len(missing_smiles)} predictions")

# Compute only missing predictions
new_predictions = {}
for smiles in missing_smiles:
    result = your_model.predict(smiles)
    new_predictions[smiles] = result

# Cache new predictions
cache.set_batch_predictions(
    predictions=new_predictions,
    prediction_type="toxicity",
    model_name="ChemBERTa"
)

# Combine results
all_results = {**cached_results, **new_predictions}
```

### Example 2: Cache with TTL (Time-To-Live)

```python
cache = PredictionCache()

# Cache prediction for 1 hour
cache.set_prediction(
    smiles="CCO",
    prediction_type="toxicity",
    model_name="test_model",
    prediction_result={"score": 0.75},
    ttl=3600  # 3600 seconds = 1 hour
)

# Cache prediction for 30 days
cache.set_prediction(
    smiles="c1ccccc1",
    prediction_type="toxicity",
    model_name="test_model",
    prediction_result={"score": 0.85},
    ttl=2592000  # 30 days
)
```

### Example 3: Embedding Cache

```python
from chemistry_agents.utils.prediction_cache import EmbeddingCache
import numpy as np

emb_cache = EmbeddingCache()

# Cache embeddings
smiles = "c1ccccc1"
embedding = model.get_embedding(smiles)  # numpy array

emb_cache.set_embedding(
    smiles=smiles,
    model_name="ChemBERTa",
    embedding=embedding
)

# Retrieve embedding
cached_embedding = emb_cache.get_embedding(
    smiles=smiles,
    model_name="ChemBERTa"
)

if cached_embedding is not None:
    print("Using cached embedding!")
```

### Example 4: Performance Profiling with Batch Operations

```python
from chemistry_agents.utils.performance_profiler import BatchProfiler

batch_profiler = BatchProfiler()

def process_batch(smiles_batch):
    results = []
    for smiles in smiles_batch:
        result = model.predict(smiles)
        results.append(result)
    return results

# Profile batch processing
smiles_batch = ["CCO", "CC", "CCC", "c1ccccc1"]
results = batch_profiler.profile_batch(
    operation_name="toxicity_prediction",
    batch_size=len(smiles_batch),
    func=process_batch,
    smiles_batch
)

# View throughput statistics
batch_profiler.print_summary()
avg_throughput = batch_profiler.get_average_throughput()
print(f"Average: {avg_throughput:.1f} molecules/second")
```

### Example 5: Cache Decorator

```python
from chemistry_agents.utils.cache_manager import CacheManager, cached

cache_mgr = CacheManager(cache_dir="cache/custom")

@cached(cache_mgr, key_prefix="toxicity", ttl=3600)
def predict_toxicity(smiles: str):
    """This function's results will be automatically cached"""
    # Expensive computation
    time.sleep(1)
    return model.predict(smiles)

# First call - computes and caches
result1 = predict_toxicity("CCO")  # Takes 1 second

# Second call - uses cache
result2 = predict_toxicity("CCO")  # Instant!
```

---

## Configuration

### Configuration File: `configs/cache_config.yaml`

```yaml
cache:
  base_dir: "cache"
  enabled: true
  max_memory_entries: 5000

prediction_cache:
  default_ttl: 2592000  # 30 days
  task_ttl:
    toxicity: 2592000
    solubility: 2592000
    bioactivity: 2592000

profiling:
  enabled: true
  reports_dir: "cache/reports"
```

### Load Configuration

```python
import yaml

with open("configs/cache_config.yaml") as f:
    config = yaml.safe_load(f)

cache = PredictionCache(
    cache_dir=config["cache"]["base_dir"],
    default_ttl=config["prediction_cache"]["default_ttl"]
)
```

---

## CLI Tools

### Cache Management CLI

The `cache_cli.py` tool provides command-line cache management:

```bash
# Show cache statistics
python cache_cli.py stats

# List all cached keys
python cache_cli.py list

# List keys matching pattern
python cache_cli.py list --pattern benzene

# Clear all caches
python cache_cli.py clear --all

# Clear specific prediction type
python cache_cli.py clear --type toxicity

# Optimize cache (remove old entries)
python cache_cli.py optimize --max-age-days 60

# Export cache data
python cache_cli.py export cache_backup.json
```

### Example Output

```
$ python cache_cli.py stats

================================================================================
CACHE STATISTICS
================================================================================

Prediction Cache:
--------------------------------------------------------------------------------
  Total Entries:     1,234
  Memory Entries:    500
  Disk Entries:      1,234
  Hit Rate:          87.3%
  Total Hits:        2,345
  Total Misses:      342
  Cache Size:        45.67 MB
  Oldest Entry:      2024-01-15 10:23:45
  Newest Entry:      2024-01-20 16:42:11

  By Prediction Type:
    toxicity              567 entries
    solubility            423 entries
    bioactivity           244 entries

Embedding Cache:
--------------------------------------------------------------------------------
  Total Entries:     3,456
  Hit Rate:          92.1%
  Cache Size:        234.12 MB
================================================================================
```

---

## Performance Tips

### 1. Enable Caching for Production

Always enable caching in production environments:

```python
cached_agent = create_cached_agent(
    agent=base_agent,
    enable_cache=True,
    enable_profiling=True  # Minimal overhead
)
```

### 2. Use Batch Operations

Batch operations are much more efficient:

```python
# Bad - many individual cache lookups
for smiles in molecule_list:
    result = cache.get_prediction(smiles, "toxicity", "model")

# Good - single batch lookup
cached, missing = cache.get_batch_predictions(
    molecule_list, "toxicity", "model"
)
```

### 3. Configure Appropriate TTL

Balance cache freshness with performance:

```python
# Short TTL for rapidly changing predictions
cache.set_prediction(..., ttl=3600)  # 1 hour

# Long TTL for stable predictions
cache.set_prediction(..., ttl=2592000)  # 30 days

# No expiration for immutable predictions
cache.set_prediction(..., ttl=None)  # Never expires
```

### 4. Regular Cache Maintenance

Schedule periodic cache optimization:

```bash
# Cron job: daily at 2 AM
0 2 * * * python cache_cli.py optimize --max-age-days 90
```

### 5. Monitor Cache Hit Rate

Track cache effectiveness:

```python
stats = cached_agent.get_cache_stats()
hit_rate = float(stats['predictions']['hit_rate'].rstrip('%'))

if hit_rate < 50:
    print("Warning: Low cache hit rate - check caching strategy")
```

### 6. Profile Before Optimizing

Always profile to find real bottlenecks:

```python
profiler = PerformanceProfiler()

# Profile your workflow
with profiler:
    run_your_analysis()

# Identify slowest operations
slowest = profiler.get_slowest_operations(n=10)
for op in slowest:
    print(f"{op.operation_name}: {op.avg_duration:.3f}s")
```

---

## API Reference

### PredictionCache

```python
cache = PredictionCache(
    cache_dir="cache/predictions",
    max_memory_entries=5000,
    default_ttl=2592000
)

# Get single prediction
result = cache.get_prediction(smiles, prediction_type, model_name, model_version)

# Set single prediction
cache.set_prediction(smiles, prediction_type, model_name, prediction_result, ...)

# Batch operations
cached, missing = cache.get_batch_predictions(smiles_list, ...)
cache.set_batch_predictions(predictions_dict, ...)

# Management
cache.clear(prediction_type=None)
stats = cache.get_stats()
type_stats = cache.get_stats_by_type()
```

### EmbeddingCache

```python
emb_cache = EmbeddingCache(cache_dir="cache/embeddings")

# Single operations
embedding = emb_cache.get_embedding(smiles, model_name)
emb_cache.set_embedding(smiles, model_name, embedding)

# Batch operations
cached, missing = emb_cache.get_batch_embeddings(smiles_list, model_name)
emb_cache.set_batch_embeddings(embeddings_dict, model_name)
```

### PerformanceProfiler

```python
profiler = PerformanceProfiler()

# Profile operation
result = profiler.profile_operation(name, func, *args, **kwargs)

# Get metrics
metrics = profiler.get_metrics()
aggregated = profiler.get_aggregated_metrics()
slowest = profiler.get_slowest_operations(n=10)

# Reports
profiler.print_report()
profiler.save_report("report.json")
```

### CachedAgentWrapper

```python
cached_agent = CachedAgentWrapper(
    agent=base_agent,
    enable_cache=True,
    enable_profiling=True
)

# Use like normal agent
result = cached_agent.predict(smiles, prediction_type)
batch_results = cached_agent.predict_batch(smiles_list, prediction_type)

# Management
stats = cached_agent.get_cache_stats()
cached_agent.get_performance_report()
cached_agent.clear_cache(prediction_type=None)
```

---

## Advanced Topics

### Custom Cache Keys

```python
from chemistry_agents.utils.cache_manager import CacheManager

cache = CacheManager()

# Custom key generation
def make_custom_key(params):
    return f"custom_{params['type']}_{params['id']}"

cache.set(make_custom_key(params), value)
```

### Cache Warming

```python
def warm_cache(smiles_list):
    """Pre-compute and cache predictions"""
    for smiles in smiles_list:
        if not cache.get_prediction(smiles, "toxicity", "model"):
            result = model.predict(smiles)
            cache.set_prediction(smiles, "toxicity", "model", result)
```

### Multi-Level Caching

```python
# L1: Memory cache (fast)
# L2: Disk cache (persistent)
# L3: Remote cache (shared across machines)

def get_with_fallback(key):
    # Try memory
    if key in memory_cache:
        return memory_cache[key]

    # Try disk
    result = disk_cache.get(key)
    if result:
        memory_cache[key] = result
        return result

    # Try remote
    result = remote_cache.get(key)
    if result:
        disk_cache.set(key, result)
        memory_cache[key] = result
        return result

    return None
```

---

## Troubleshooting

### Cache Not Persisting

Check disk cache is enabled:
```python
cache = PredictionCache(enable_disk_cache=True)
```

### High Memory Usage

Reduce max memory entries:
```python
cache = PredictionCache(max_memory_entries=1000)
```

### Slow Cache Lookups

The cache may be too large. Optimize:
```bash
python cache_cli.py optimize --max-age-days 30
```

### Cache Statistics Not Updating

Ensure cache is shared across calls:
```python
# Bad - creates new cache each time
def predict(smiles):
    cache = PredictionCache()  # New instance!
    ...

# Good - reuse cache instance
cache = PredictionCache()
def predict(smiles):
    ...
```

---

## Best Practices

1. **Always use caching for production workloads**
2. **Profile first, optimize second**
3. **Set appropriate TTL values**
4. **Monitor cache hit rates**
5. **Regular cache maintenance**
6. **Use batch operations when possible**
7. **Canonicalize SMILES before caching**
8. **Version your models in cache keys**

---

For more information, see the source code in:
- `src/chemistry_agents/utils/cache_manager.py`
- `src/chemistry_agents/utils/prediction_cache.py`
- `src/chemistry_agents/utils/performance_profiler.py`
