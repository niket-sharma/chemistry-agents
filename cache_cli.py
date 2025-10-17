#!/usr/bin/env python3
"""
Cache Management CLI

Command-line interface for managing chemistry agent caches

Usage:
    python cache_cli.py stats                    # Show cache statistics
    python cache_cli.py clear --all              # Clear all caches
    python cache_cli.py clear --type toxicity    # Clear specific prediction type
    python cache_cli.py list --pattern benzene   # List cache keys matching pattern
    python cache_cli.py info <key>               # Show info for specific cache entry
    python cache_cli.py optimize                 # Optimize cache (remove old entries)
    python cache_cli.py export output.json       # Export cache data
"""

import sys
import os
import argparse
import json
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from chemistry_agents.utils.prediction_cache import PredictionCache, EmbeddingCache
from chemistry_agents.utils.cache_manager import CacheManager


def format_size(size_bytes: int) -> str:
    """Format size in bytes to human-readable string"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} TB"


def show_stats(cache_dir: str = "cache"):
    """Show cache statistics"""
    print("\n" + "=" * 80)
    print("CACHE STATISTICS")
    print("=" * 80)

    # Prediction cache stats
    print("\nPrediction Cache:")
    print("-" * 80)
    try:
        pred_cache = PredictionCache(cache_dir=f"{cache_dir}/predictions")
        stats = pred_cache.get_stats()

        print(f"  Total Entries:     {stats.total_entries}")
        print(f"  Memory Entries:    {stats.memory_entries}")
        print(f"  Disk Entries:      {stats.disk_entries}")
        print(f"  Hit Rate:          {stats.hit_rate:.1f}%")
        print(f"  Total Hits:        {stats.total_hits}")
        print(f"  Total Misses:      {stats.total_misses}")
        print(f"  Cache Size:        {format_size(stats.total_size_bytes)}")

        if stats.oldest_entry:
            print(f"  Oldest Entry:      {stats.oldest_entry.strftime('%Y-%m-%d %H:%M:%S')}")
        if stats.newest_entry:
            print(f"  Newest Entry:      {stats.newest_entry.strftime('%Y-%m-%d %H:%M:%S')}")

        # Stats by type
        print("\n  By Prediction Type:")
        type_stats = pred_cache.get_stats_by_type()
        for pred_type, count in sorted(type_stats.items(), key=lambda x: x[1], reverse=True):
            print(f"    {pred_type:<20} {count:>6} entries")

    except Exception as e:
        print(f"  Error: {e}")

    # Embedding cache stats
    print("\nEmbedding Cache:")
    print("-" * 80)
    try:
        emb_cache = EmbeddingCache(cache_dir=f"{cache_dir}/embeddings")
        stats = emb_cache.get_stats()

        print(f"  Total Entries:     {stats.total_entries}")
        print(f"  Memory Entries:    {stats.memory_entries}")
        print(f"  Disk Entries:      {stats.disk_entries}")
        print(f"  Hit Rate:          {stats.hit_rate:.1f}%")
        print(f"  Cache Size:        {format_size(stats.total_size_bytes)}")

    except Exception as e:
        print(f"  Error: {e}")

    print("\n" + "=" * 80 + "\n")


def clear_cache(cache_dir: str = "cache", clear_all: bool = False, prediction_type: str = None):
    """Clear cache entries"""
    if clear_all:
        confirm = input("Are you sure you want to clear ALL cache entries? (yes/no): ")
        if confirm.lower() != 'yes':
            print("Operation cancelled.")
            return

        # Clear prediction cache
        pred_cache = PredictionCache(cache_dir=f"{cache_dir}/predictions")
        pred_cache.clear()

        # Clear embedding cache
        emb_cache = EmbeddingCache(cache_dir=f"{cache_dir}/embeddings")
        emb_cache.clear()

        print("✓ All caches cleared successfully")

    elif prediction_type:
        pred_cache = PredictionCache(cache_dir=f"{cache_dir}/predictions")
        pred_cache.clear(prediction_type=prediction_type)
        print(f"✓ Cleared {prediction_type} predictions")

    else:
        print("Error: Specify --all to clear all caches or --type <type> for specific type")


def list_keys(cache_dir: str = "cache", pattern: str = None, limit: int = 100):
    """List cache keys"""
    print("\nCache Keys:")
    print("-" * 80)

    pred_cache = PredictionCache(cache_dir=f"{cache_dir}/predictions")
    keys = pred_cache.cache_manager.list_keys(pattern=pattern)

    if not keys:
        print("  No keys found")
        return

    total = len(keys)
    displayed = min(total, limit)

    for i, key in enumerate(keys[:limit], 1):
        print(f"  {i}. {key}")

    if total > limit:
        print(f"\n  ... and {total - limit} more keys (use --limit to show more)")

    print(f"\nTotal: {total} keys")
    print("-" * 80)


def optimize_cache(cache_dir: str = "cache", max_age_days: int = 90):
    """Optimize cache by removing old entries"""
    print("\nOptimizing cache...")
    print("-" * 80)

    max_age_seconds = max_age_days * 86400

    # Optimize prediction cache
    pred_cache = PredictionCache(cache_dir=f"{cache_dir}/predictions")
    pred_cache.cache_manager.clear(max_age=max_age_seconds)

    # Optimize embedding cache
    emb_cache = EmbeddingCache(cache_dir=f"{cache_dir}/embeddings")
    emb_cache.cache_manager.clear(max_age=max_age_seconds)

    print(f"✓ Removed entries older than {max_age_days} days")
    print("-" * 80)


def export_cache(cache_dir: str = "cache", output_file: str = "cache_export.json"):
    """Export cache data to file"""
    print(f"\nExporting cache to {output_file}...")

    pred_cache = PredictionCache(cache_dir=f"{cache_dir}/predictions")
    keys = pred_cache.cache_manager.list_keys()

    export_data = {
        "exported_at": datetime.now().isoformat(),
        "total_entries": len(keys),
        "entries": []
    }

    for key in keys:
        entry_data = pred_cache.cache_manager.get(key)
        if entry_data:
            export_data["entries"].append({
                "key": key,
                "data": entry_data
            })

    with open(output_file, 'w') as f:
        json.dump(export_data, f, indent=2, default=str)

    print(f"✓ Exported {len(export_data['entries'])} entries to {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Cache Management CLI for Chemistry Agents",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s stats                          Show cache statistics
  %(prog)s clear --all                    Clear all caches
  %(prog)s clear --type toxicity          Clear specific prediction type
  %(prog)s list --pattern benzene         List keys matching pattern
  %(prog)s optimize --max-age-days 60     Remove entries older than 60 days
  %(prog)s export output.json             Export cache data
        """
    )

    parser.add_argument(
        'command',
        choices=['stats', 'clear', 'list', 'optimize', 'export'],
        help='Command to execute'
    )

    parser.add_argument(
        '--cache-dir',
        default='cache',
        help='Cache directory (default: cache)'
    )

    # Clear command options
    parser.add_argument(
        '--all',
        action='store_true',
        help='Clear all caches'
    )

    parser.add_argument(
        '--type',
        help='Clear specific prediction type'
    )

    # List command options
    parser.add_argument(
        '--pattern',
        help='Filter keys by pattern'
    )

    parser.add_argument(
        '--limit',
        type=int,
        default=100,
        help='Maximum number of keys to display (default: 100)'
    )

    # Optimize command options
    parser.add_argument(
        '--max-age-days',
        type=int,
        default=90,
        help='Maximum age in days (default: 90)'
    )

    # Export command options
    parser.add_argument(
        'output',
        nargs='?',
        default='cache_export.json',
        help='Output file for export (default: cache_export.json)'
    )

    args = parser.parse_args()

    # Execute command
    try:
        if args.command == 'stats':
            show_stats(args.cache_dir)

        elif args.command == 'clear':
            clear_cache(
                cache_dir=args.cache_dir,
                clear_all=args.all,
                prediction_type=args.type
            )

        elif args.command == 'list':
            list_keys(
                cache_dir=args.cache_dir,
                pattern=args.pattern,
                limit=args.limit
            )

        elif args.command == 'optimize':
            optimize_cache(
                cache_dir=args.cache_dir,
                max_age_days=args.max_age_days
            )

        elif args.command == 'export':
            export_cache(
                cache_dir=args.cache_dir,
                output_file=args.output
            )

    except Exception as e:
        print(f"\n❌ Error: {e}", file=sys.stderr)
        return 1

    return 0


if __name__ == '__main__':
    sys.exit(main())
