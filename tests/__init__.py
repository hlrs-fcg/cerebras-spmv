"""
Benchmarking and Testing Infrastructure

This package provides utilities for saving and analyzing SpMV benchmark results.
"""

from .benchmark import BenchmarkResult, save_benchmark, load_benchmarks

__all__ = ['BenchmarkResult', 'save_benchmark', 'load_benchmarks']
