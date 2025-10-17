"""
Cache Manager for Chemistry Agents

Provides efficient caching mechanisms for:
- SMILES molecular predictions
- Model embeddings
- Query results
- Performance optimization

Features:
- Persistent storage (SQLite + JSON)
- Configurable TTL (time-to-live)
- Memory and disk caching
- Cache statistics and management
"""

import os
import json
import hashlib
import sqlite3
import time
import pickle
from typing import Any, Dict, Optional, List, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
import logging
from functools import wraps


@dataclass
class CacheEntry:
    """Represents a single cache entry"""
    key: str
    value: Any
    created_at: float
    accessed_at: float
    access_count: int
    ttl: Optional[int] = None  # seconds, None = never expire
    metadata: Dict[str, Any] = None

    def is_expired(self) -> bool:
        """Check if entry has expired"""
        if self.ttl is None:
            return False
        return (time.time() - self.created_at) > self.ttl

    def is_stale(self, max_age: int) -> bool:
        """Check if entry is older than max_age seconds"""
        return (time.time() - self.created_at) > max_age


@dataclass
class CacheStats:
    """Cache statistics"""
    total_entries: int
    memory_entries: int
    disk_entries: int
    total_hits: int
    total_misses: int
    hit_rate: float
    total_size_bytes: int
    oldest_entry: Optional[datetime]
    newest_entry: Optional[datetime]


class CacheManager:
    """
    Main cache manager with memory and disk persistence

    Usage:
        cache = CacheManager(cache_dir="cache/predictions")

        # Store prediction
        cache.set("molecule_benzene_toxicity", {"score": 0.75}, ttl=3600)

        # Retrieve prediction
        result = cache.get("molecule_benzene_toxicity")
    """

    def __init__(
        self,
        cache_dir: str = "cache",
        max_memory_entries: int = 1000,
        default_ttl: Optional[int] = None,
        enable_disk_cache: bool = True,
        logger: Optional[logging.Logger] = None
    ):
        self.cache_dir = Path(cache_dir)
        self.max_memory_entries = max_memory_entries
        self.default_ttl = default_ttl
        self.enable_disk_cache = enable_disk_cache
        self.logger = logger or logging.getLogger(__name__)

        # In-memory cache (LRU-like)
        self._memory_cache: Dict[str, CacheEntry] = {}

        # Statistics
        self._hits = 0
        self._misses = 0

        # Initialize
        self._initialize_cache()

    def _initialize_cache(self):
        """Initialize cache directory and database"""
        if self.enable_disk_cache:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self._db_path = self.cache_dir / "cache.db"
            self._init_database()
            self.logger.info(f"Cache initialized at {self.cache_dir}")

    def _init_database(self):
        """Initialize SQLite database for disk cache"""
        with sqlite3.connect(str(self._db_path)) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS cache_entries (
                    key TEXT PRIMARY KEY,
                    value_type TEXT,
                    value_path TEXT,
                    created_at REAL,
                    accessed_at REAL,
                    access_count INTEGER,
                    ttl INTEGER,
                    metadata TEXT
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_created_at
                ON cache_entries(created_at)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_accessed_at
                ON cache_entries(accessed_at)
            """)
            conn.commit()

    def _hash_key(self, key: str) -> str:
        """Create hash for cache key"""
        return hashlib.sha256(key.encode()).hexdigest()

    def get(self, key: str, default: Any = None) -> Optional[Any]:
        """
        Retrieve value from cache

        Args:
            key: Cache key
            default: Default value if not found

        Returns:
            Cached value or default
        """
        # Check memory cache first
        if key in self._memory_cache:
            entry = self._memory_cache[key]
            if not entry.is_expired():
                entry.accessed_at = time.time()
                entry.access_count += 1
                self._hits += 1
                self.logger.debug(f"Cache HIT (memory): {key}")
                return entry.value
            else:
                # Expired, remove from memory
                del self._memory_cache[key]

        # Check disk cache
        if self.enable_disk_cache:
            value = self._get_from_disk(key)
            if value is not None:
                # Promote to memory cache
                self._memory_cache[key] = CacheEntry(
                    key=key,
                    value=value,
                    created_at=time.time(),
                    accessed_at=time.time(),
                    access_count=1
                )
                self._hits += 1
                self.logger.debug(f"Cache HIT (disk): {key}")
                return value

        self._misses += 1
        self.logger.debug(f"Cache MISS: {key}")
        return default

    def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
        metadata: Optional[Dict] = None
    ):
        """
        Store value in cache

        Args:
            key: Cache key
            value: Value to cache
            ttl: Time-to-live in seconds (None = use default)
            metadata: Additional metadata
        """
        ttl = ttl if ttl is not None else self.default_ttl

        entry = CacheEntry(
            key=key,
            value=value,
            created_at=time.time(),
            accessed_at=time.time(),
            access_count=1,
            ttl=ttl,
            metadata=metadata or {}
        )

        # Store in memory
        self._memory_cache[key] = entry

        # Evict if memory limit exceeded
        if len(self._memory_cache) > self.max_memory_entries:
            self._evict_lru()

        # Store on disk
        if self.enable_disk_cache:
            self._save_to_disk(entry)

        self.logger.debug(f"Cache SET: {key}")

    def _get_from_disk(self, key: str) -> Optional[Any]:
        """Retrieve value from disk cache"""
        try:
            with sqlite3.connect(str(self._db_path)) as conn:
                cursor = conn.execute(
                    "SELECT value_type, value_path, ttl, created_at FROM cache_entries WHERE key = ?",
                    (key,)
                )
                row = cursor.fetchone()

                if row is None:
                    return None

                value_type, value_path, ttl, created_at = row

                # Check expiration
                if ttl is not None and (time.time() - created_at) > ttl:
                    self.delete(key)
                    return None

                # Load value
                full_path = self.cache_dir / value_path
                if not full_path.exists():
                    return None

                if value_type == "json":
                    with open(full_path, 'r') as f:
                        return json.load(f)
                elif value_type == "pickle":
                    with open(full_path, 'rb') as f:
                        return pickle.load(f)

                # Update access info
                conn.execute(
                    """UPDATE cache_entries
                       SET accessed_at = ?, access_count = access_count + 1
                       WHERE key = ?""",
                    (time.time(), key)
                )
                conn.commit()

        except Exception as e:
            self.logger.error(f"Error reading from disk cache: {e}")
            return None

    def _save_to_disk(self, entry: CacheEntry):
        """Save entry to disk cache"""
        try:
            # Determine storage format
            try:
                json.dumps(entry.value)
                value_type = "json"
            except (TypeError, ValueError):
                value_type = "pickle"

            # Create file path
            hashed_key = self._hash_key(entry.key)
            value_filename = f"{hashed_key}.{value_type}"
            value_path = self.cache_dir / value_filename

            # Save value
            if value_type == "json":
                with open(value_path, 'w') as f:
                    json.dump(entry.value, f)
            else:
                with open(value_path, 'wb') as f:
                    pickle.dump(entry.value, f)

            # Save metadata to database
            with sqlite3.connect(str(self._db_path)) as conn:
                conn.execute(
                    """INSERT OR REPLACE INTO cache_entries
                       (key, value_type, value_path, created_at, accessed_at, access_count, ttl, metadata)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        entry.key,
                        value_type,
                        value_filename,
                        entry.created_at,
                        entry.accessed_at,
                        entry.access_count,
                        entry.ttl,
                        json.dumps(entry.metadata or {})
                    )
                )
                conn.commit()

        except Exception as e:
            self.logger.error(f"Error saving to disk cache: {e}")

    def _evict_lru(self):
        """Evict least recently used entries from memory"""
        if not self._memory_cache:
            return

        # Sort by access time
        sorted_entries = sorted(
            self._memory_cache.items(),
            key=lambda x: x[1].accessed_at
        )

        # Remove oldest 10%
        num_to_remove = max(1, len(sorted_entries) // 10)
        for key, _ in sorted_entries[:num_to_remove]:
            del self._memory_cache[key]

        self.logger.debug(f"Evicted {num_to_remove} entries from memory cache")

    def delete(self, key: str) -> bool:
        """Delete entry from cache"""
        deleted = False

        # Remove from memory
        if key in self._memory_cache:
            del self._memory_cache[key]
            deleted = True

        # Remove from disk
        if self.enable_disk_cache:
            try:
                with sqlite3.connect(str(self._db_path)) as conn:
                    cursor = conn.execute(
                        "SELECT value_path FROM cache_entries WHERE key = ?",
                        (key,)
                    )
                    row = cursor.fetchone()

                    if row:
                        value_path = self.cache_dir / row[0]
                        if value_path.exists():
                            value_path.unlink()

                        conn.execute("DELETE FROM cache_entries WHERE key = ?", (key,))
                        conn.commit()
                        deleted = True
            except Exception as e:
                self.logger.error(f"Error deleting from disk cache: {e}")

        return deleted

    def clear(self, max_age: Optional[int] = None):
        """
        Clear cache entries

        Args:
            max_age: If specified, only clear entries older than this (seconds)
        """
        if max_age is None:
            # Clear everything
            self._memory_cache.clear()

            if self.enable_disk_cache:
                try:
                    with sqlite3.connect(str(self._db_path)) as conn:
                        # Get all files
                        cursor = conn.execute("SELECT value_path FROM cache_entries")
                        for (value_path,) in cursor:
                            file_path = self.cache_dir / value_path
                            if file_path.exists():
                                file_path.unlink()

                        # Clear database
                        conn.execute("DELETE FROM cache_entries")
                        conn.commit()
                except Exception as e:
                    self.logger.error(f"Error clearing disk cache: {e}")

            self.logger.info("Cache cleared completely")
        else:
            # Clear old entries
            cutoff_time = time.time() - max_age

            # Clear from memory
            keys_to_delete = [
                k for k, v in self._memory_cache.items()
                if v.created_at < cutoff_time
            ]
            for key in keys_to_delete:
                del self._memory_cache[key]

            # Clear from disk
            if self.enable_disk_cache:
                try:
                    with sqlite3.connect(str(self._db_path)) as conn:
                        cursor = conn.execute(
                            "SELECT key, value_path FROM cache_entries WHERE created_at < ?",
                            (cutoff_time,)
                        )
                        for key, value_path in cursor:
                            file_path = self.cache_dir / value_path
                            if file_path.exists():
                                file_path.unlink()

                        conn.execute("DELETE FROM cache_entries WHERE created_at < ?", (cutoff_time,))
                        conn.commit()
                except Exception as e:
                    self.logger.error(f"Error clearing old entries: {e}")

            self.logger.info(f"Cleared entries older than {max_age} seconds")

    def get_stats(self) -> CacheStats:
        """Get cache statistics"""
        memory_entries = len(self._memory_cache)
        disk_entries = 0
        oldest_entry = None
        newest_entry = None
        total_size = 0

        if self.enable_disk_cache:
            try:
                with sqlite3.connect(str(self._db_path)) as conn:
                    cursor = conn.execute(
                        "SELECT COUNT(*), MIN(created_at), MAX(created_at) FROM cache_entries"
                    )
                    row = cursor.fetchone()
                    if row[0]:
                        disk_entries = row[0]
                        oldest_entry = datetime.fromtimestamp(row[1]) if row[1] else None
                        newest_entry = datetime.fromtimestamp(row[2]) if row[2] else None

                    # Calculate total size
                    for file in self.cache_dir.glob("*.json"):
                        total_size += file.stat().st_size
                    for file in self.cache_dir.glob("*.pickle"):
                        total_size += file.stat().st_size
            except Exception as e:
                self.logger.error(f"Error getting stats: {e}")

        total_requests = self._hits + self._misses
        hit_rate = (self._hits / total_requests * 100) if total_requests > 0 else 0.0

        return CacheStats(
            total_entries=memory_entries + disk_entries,
            memory_entries=memory_entries,
            disk_entries=disk_entries,
            total_hits=self._hits,
            total_misses=self._misses,
            hit_rate=hit_rate,
            total_size_bytes=total_size,
            oldest_entry=oldest_entry,
            newest_entry=newest_entry
        )

    def list_keys(self, pattern: Optional[str] = None) -> List[str]:
        """List all cache keys, optionally filtered by pattern"""
        keys = list(self._memory_cache.keys())

        if self.enable_disk_cache:
            try:
                with sqlite3.connect(str(self._db_path)) as conn:
                    cursor = conn.execute("SELECT key FROM cache_entries")
                    keys.extend([row[0] for row in cursor])
            except Exception as e:
                self.logger.error(f"Error listing keys: {e}")

        # Remove duplicates
        keys = list(set(keys))

        # Filter by pattern
        if pattern:
            keys = [k for k in keys if pattern in k]

        return sorted(keys)


def cached(
    cache_manager: CacheManager,
    key_prefix: str = "",
    ttl: Optional[int] = None,
    include_args: bool = True
):
    """
    Decorator for caching function results

    Usage:
        @cached(cache_manager, key_prefix="prediction", ttl=3600)
        def predict_toxicity(smiles: str):
            # expensive computation
            return result
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key
            if include_args:
                key_parts = [key_prefix, func.__name__] + [str(arg) for arg in args]
                key_parts += [f"{k}={v}" for k, v in sorted(kwargs.items())]
                cache_key = "_".join(key_parts)
            else:
                cache_key = f"{key_prefix}_{func.__name__}"

            # Try to get from cache
            result = cache_manager.get(cache_key)
            if result is not None:
                return result

            # Compute and cache
            result = func(*args, **kwargs)
            cache_manager.set(cache_key, result, ttl=ttl)
            return result

        return wrapper
    return decorator
