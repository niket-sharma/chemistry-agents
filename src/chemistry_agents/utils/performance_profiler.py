"""
Performance Profiling Utilities for Chemistry Agents

Tools for measuring and optimizing performance:
- Function execution timing
- Memory usage tracking
- Model inference profiling
- Batch processing optimization
- Performance reports
"""

import time
import functools
import psutil
import logging
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import json
from collections import defaultdict


@dataclass
class PerformanceMetrics:
    """Performance metrics for a single operation"""
    operation_name: str
    start_time: float
    end_time: float
    duration: float
    memory_before_mb: float
    memory_after_mb: float
    memory_delta_mb: float
    success: bool
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            "operation_name": self.operation_name,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_seconds": self.duration,
            "memory_before_mb": self.memory_before_mb,
            "memory_after_mb": self.memory_after_mb,
            "memory_delta_mb": self.memory_delta_mb,
            "success": self.success,
            "error_message": self.error_message,
            "metadata": self.metadata
        }


@dataclass
class AggregatedMetrics:
    """Aggregated performance metrics"""
    operation_name: str
    call_count: int
    total_duration: float
    avg_duration: float
    min_duration: float
    max_duration: float
    avg_memory_delta_mb: float
    success_count: int
    failure_count: int
    success_rate: float


class PerformanceProfiler:
    """
    Performance profiler for chemistry agents

    Features:
    - Automatic timing and memory tracking
    - Operation aggregation
    - Report generation
    - Decorator support
    """

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self._metrics: List[PerformanceMetrics] = []
        self._process = psutil.Process()
        self._enabled = True

    def enable(self):
        """Enable profiling"""
        self._enabled = True
        self.logger.info("Profiling enabled")

    def disable(self):
        """Disable profiling"""
        self._enabled = False
        self.logger.info("Profiling disabled")

    def _get_memory_mb(self) -> float:
        """Get current memory usage in MB"""
        return self._process.memory_info().rss / 1024 / 1024

    def profile_operation(
        self,
        operation_name: str,
        func: Callable,
        *args,
        metadata: Optional[Dict] = None,
        **kwargs
    ) -> Any:
        """
        Profile a function execution

        Args:
            operation_name: Name of the operation
            func: Function to profile
            *args: Function arguments
            metadata: Additional metadata
            **kwargs: Function keyword arguments

        Returns:
            Function result
        """
        if not self._enabled:
            return func(*args, **kwargs)

        memory_before = self._get_memory_mb()
        start_time = time.time()
        success = True
        error_message = None
        result = None

        try:
            result = func(*args, **kwargs)
        except Exception as e:
            success = False
            error_message = str(e)
            raise
        finally:
            end_time = time.time()
            memory_after = self._get_memory_mb()

            metrics = PerformanceMetrics(
                operation_name=operation_name,
                start_time=start_time,
                end_time=end_time,
                duration=end_time - start_time,
                memory_before_mb=memory_before,
                memory_after_mb=memory_after,
                memory_delta_mb=memory_after - memory_before,
                success=success,
                error_message=error_message,
                metadata=metadata or {}
            )

            self._metrics.append(metrics)

            if success:
                self.logger.debug(
                    f"{operation_name}: {metrics.duration:.3f}s, "
                    f"Mem: {metrics.memory_delta_mb:+.2f}MB"
                )
            else:
                self.logger.error(f"{operation_name} failed: {error_message}")

        return result

    def get_metrics(self) -> List[PerformanceMetrics]:
        """Get all recorded metrics"""
        return self._metrics.copy()

    def get_aggregated_metrics(self) -> List[AggregatedMetrics]:
        """
        Get aggregated metrics grouped by operation

        Returns:
            List of aggregated metrics
        """
        operation_groups = defaultdict(list)

        for metric in self._metrics:
            operation_groups[metric.operation_name].append(metric)

        aggregated = []
        for operation_name, metrics in operation_groups.items():
            durations = [m.duration for m in metrics]
            memory_deltas = [m.memory_delta_mb for m in metrics]
            success_count = sum(1 for m in metrics if m.success)
            failure_count = len(metrics) - success_count

            aggregated.append(AggregatedMetrics(
                operation_name=operation_name,
                call_count=len(metrics),
                total_duration=sum(durations),
                avg_duration=sum(durations) / len(durations),
                min_duration=min(durations),
                max_duration=max(durations),
                avg_memory_delta_mb=sum(memory_deltas) / len(memory_deltas),
                success_count=success_count,
                failure_count=failure_count,
                success_rate=success_count / len(metrics) * 100
            ))

        return sorted(aggregated, key=lambda x: x.total_duration, reverse=True)

    def print_report(self):
        """Print performance report to console"""
        aggregated = self.get_aggregated_metrics()

        if not aggregated:
            self.logger.info("No performance metrics recorded")
            return

        print("\n" + "=" * 80)
        print("PERFORMANCE REPORT")
        print("=" * 80)

        total_calls = sum(m.call_count for m in aggregated)
        total_time = sum(m.total_duration for m in aggregated)

        print(f"\nTotal Operations: {len(aggregated)}")
        print(f"Total Calls: {total_calls}")
        print(f"Total Time: {total_time:.2f}s")

        print("\n" + "-" * 80)
        print(f"{'Operation':<40} {'Calls':>8} {'Total(s)':>10} {'Avg(s)':>10} {'Success%':>10}")
        print("-" * 80)

        for metric in aggregated:
            print(
                f"{metric.operation_name:<40} "
                f"{metric.call_count:>8} "
                f"{metric.total_duration:>10.3f} "
                f"{metric.avg_duration:>10.3f} "
                f"{metric.success_rate:>9.1f}%"
            )

        print("-" * 80)

        # Memory statistics
        print("\nMemory Usage:")
        print("-" * 80)
        print(f"{'Operation':<40} {'Avg Delta (MB)':>15}")
        print("-" * 80)

        for metric in aggregated:
            print(f"{metric.operation_name:<40} {metric.avg_memory_delta_mb:>+14.2f}")

        print("=" * 80 + "\n")

    def save_report(self, output_file: str):
        """
        Save performance report to file

        Args:
            output_file: Output file path (JSON or TXT)
        """
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if output_path.suffix == ".json":
            self._save_json_report(output_path)
        else:
            self._save_text_report(output_path)

        self.logger.info(f"Performance report saved to {output_path}")

    def _save_json_report(self, output_path: Path):
        """Save report as JSON"""
        aggregated = self.get_aggregated_metrics()

        report = {
            "generated_at": datetime.now().isoformat(),
            "total_operations": len(aggregated),
            "total_calls": sum(m.call_count for m in aggregated),
            "total_duration": sum(m.total_duration for m in aggregated),
            "operations": [
                {
                    "name": m.operation_name,
                    "call_count": m.call_count,
                    "total_duration": m.total_duration,
                    "avg_duration": m.avg_duration,
                    "min_duration": m.min_duration,
                    "max_duration": m.max_duration,
                    "avg_memory_delta_mb": m.avg_memory_delta_mb,
                    "success_count": m.success_count,
                    "failure_count": m.failure_count,
                    "success_rate": m.success_rate
                }
                for m in aggregated
            ],
            "detailed_metrics": [m.to_dict() for m in self._metrics]
        }

        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)

    def _save_text_report(self, output_path: Path):
        """Save report as text"""
        with open(output_path, 'w') as f:
            aggregated = self.get_aggregated_metrics()

            f.write("=" * 80 + "\n")
            f.write("PERFORMANCE REPORT\n")
            f.write("=" * 80 + "\n")
            f.write(f"Generated: {datetime.now().isoformat()}\n\n")

            total_calls = sum(m.call_count for m in aggregated)
            total_time = sum(m.total_duration for m in aggregated)

            f.write(f"Total Operations: {len(aggregated)}\n")
            f.write(f"Total Calls: {total_calls}\n")
            f.write(f"Total Time: {total_time:.2f}s\n\n")

            f.write("-" * 80 + "\n")
            f.write(f"{'Operation':<40} {'Calls':>8} {'Total(s)':>10} {'Avg(s)':>10} {'Success%':>10}\n")
            f.write("-" * 80 + "\n")

            for metric in aggregated:
                f.write(
                    f"{metric.operation_name:<40} "
                    f"{metric.call_count:>8} "
                    f"{metric.total_duration:>10.3f} "
                    f"{metric.avg_duration:>10.3f} "
                    f"{metric.success_rate:>9.1f}%\n"
                )

            f.write("-" * 80 + "\n")

    def clear(self):
        """Clear all recorded metrics"""
        self._metrics.clear()
        self.logger.info("Performance metrics cleared")

    def get_slowest_operations(self, n: int = 10) -> List[AggregatedMetrics]:
        """
        Get N slowest operations by total duration

        Args:
            n: Number of operations to return

        Returns:
            List of slowest operations
        """
        aggregated = self.get_aggregated_metrics()
        return aggregated[:n]

    def get_memory_intensive_operations(self, n: int = 10) -> List[AggregatedMetrics]:
        """
        Get N most memory-intensive operations

        Args:
            n: Number of operations to return

        Returns:
            List of memory-intensive operations
        """
        aggregated = self.get_aggregated_metrics()
        return sorted(aggregated, key=lambda x: abs(x.avg_memory_delta_mb), reverse=True)[:n]


def profile(
    profiler: PerformanceProfiler,
    operation_name: Optional[str] = None,
    metadata: Optional[Dict] = None
):
    """
    Decorator for profiling functions

    Usage:
        @profile(profiler, operation_name="predict_toxicity")
        def predict_toxicity(smiles):
            # expensive computation
            return result
    """
    def decorator(func):
        nonlocal operation_name
        if operation_name is None:
            operation_name = func.__name__

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return profiler.profile_operation(
                operation_name,
                func,
                *args,
                metadata=metadata,
                **kwargs
            )

        return wrapper
    return decorator


class BatchProfiler:
    """
    Specialized profiler for batch operations

    Tracks per-item and per-batch metrics
    """

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self._batch_metrics: List[Dict[str, Any]] = []

    def profile_batch(
        self,
        operation_name: str,
        batch_size: int,
        func: Callable,
        *args,
        **kwargs
    ) -> Any:
        """
        Profile a batch operation

        Args:
            operation_name: Name of the operation
            batch_size: Number of items in batch
            func: Function to profile
            *args: Function arguments
            **kwargs: Function keyword arguments

        Returns:
            Function result
        """
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()

        duration = end_time - start_time
        per_item_time = duration / batch_size if batch_size > 0 else 0

        metrics = {
            "operation_name": operation_name,
            "batch_size": batch_size,
            "total_duration": duration,
            "per_item_duration": per_item_time,
            "throughput": batch_size / duration if duration > 0 else 0,
            "timestamp": start_time
        }

        self._batch_metrics.append(metrics)

        self.logger.info(
            f"{operation_name}: Processed {batch_size} items in {duration:.2f}s "
            f"({per_item_time*1000:.2f}ms/item, {metrics['throughput']:.1f} items/s)"
        )

        return result

    def get_average_throughput(self, operation_name: Optional[str] = None) -> float:
        """
        Calculate average throughput

        Args:
            operation_name: Filter by operation name (None = all)

        Returns:
            Average throughput (items/second)
        """
        metrics = self._batch_metrics
        if operation_name:
            metrics = [m for m in metrics if m["operation_name"] == operation_name]

        if not metrics:
            return 0.0

        return sum(m["throughput"] for m in metrics) / len(metrics)

    def print_summary(self):
        """Print batch processing summary"""
        if not self._batch_metrics:
            print("No batch metrics recorded")
            return

        print("\n" + "=" * 80)
        print("BATCH PROCESSING SUMMARY")
        print("=" * 80)

        total_items = sum(m["batch_size"] for m in self._batch_metrics)
        total_time = sum(m["total_duration"] for m in self._batch_metrics)
        avg_throughput = sum(m["throughput"] for m in self._batch_metrics) / len(self._batch_metrics)

        print(f"\nTotal Batches: {len(self._batch_metrics)}")
        print(f"Total Items: {total_items}")
        print(f"Total Time: {total_time:.2f}s")
        print(f"Average Throughput: {avg_throughput:.1f} items/s")

        print("\n" + "-" * 80)
        print(f"{'Operation':<40} {'Batches':>8} {'Items':>10} {'Avg Throughput':>15}")
        print("-" * 80)

        # Group by operation
        operations = defaultdict(list)
        for m in self._batch_metrics:
            operations[m["operation_name"]].append(m)

        for op_name, op_metrics in operations.items():
            batch_count = len(op_metrics)
            total_items = sum(m["batch_size"] for m in op_metrics)
            avg_throughput = sum(m["throughput"] for m in op_metrics) / len(op_metrics)

            print(
                f"{op_name:<40} "
                f"{batch_count:>8} "
                f"{total_items:>10} "
                f"{avg_throughput:>13.1f}/s"
            )

        print("=" * 80 + "\n")


# Global profiler instance
_global_profiler = PerformanceProfiler()


def get_global_profiler() -> PerformanceProfiler:
    """Get the global profiler instance"""
    return _global_profiler
