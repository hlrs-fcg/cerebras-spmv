#!/usr/bin/env python3
"""
Comprehensive Sweep Configuration Generator for Cluster Execution

Generates sweep configurations for systematic benchmarking on Cerebras WSE-3.
Based on analysis from benchmark-analysis.ipynb.
"""

import argparse
import csv
import numpy as np
from pathlib import Path
from typing import List, Dict


# WSE-3 Hardware Constraints
WSE3_MAX_WIDTH = 758  # 762 - 4 (fabric_offsets)
WSE3_MAX_HEIGHT = 1171  # 1172 - 1 (fabric_offsets)
WSE3_MAX_PES = WSE3_MAX_WIDTH * WSE3_MAX_HEIGHT  # ~888,000 PEs
PE_MEMORY_LIMIT = 48000  # bytes per PE


def generate_density_sweep(
    rows: int,
    cols: int,
    num_pes: int,
    density_min: float = 0.001,
    density_max: float = 0.1,
    num_points: int = 100,
    log_scale: bool = True
) -> List[Dict]:
    """Generate density sweep with fixed matrix size and PE count.

    Useful for understanding: density vs compute time, memory impact, load balancing.
    """
    if log_scale:
        densities = np.logspace(np.log10(density_min), np.log10(density_max), num_points)
    else:
        densities = np.linspace(density_min, density_max, num_points)

    configs = []
    for density in densities:
        configs.append({
            'rows': rows,
            'cols': cols,
            'density': float(density),
            'num_pes': num_pes,
            'seed': 42,
            'arch': 'wse3',
            'comment': f'Density sweep: {rows}x{cols}, {num_pes} PEs'
        })

    return configs


def generate_pe_scaling_sweep(
    rows: int,
    cols: int,
    density: float,
    pe_start: int = 16,
    pe_end: int = 16384,
    scaling: str = 'power2'
) -> List[Dict]:
    """Generate PE scaling sweep (strong scaling).

    Fixed problem size, increasing PEs. Shows parallel efficiency and overhead.
    """
    if scaling == 'power2':
        # Powers of 2: 16, 32, 64, 128, ...
        pe_counts = [2**i for i in range(int(np.log2(pe_start)), int(np.log2(pe_end)) + 1)]
    elif scaling == 'linear':
        # Linear spacing
        pe_counts = list(range(pe_start, pe_end + 1, (pe_end - pe_start) // 20))
    else:
        raise ValueError(f"Unknown scaling: {scaling}")

    configs = []
    for num_pes in pe_counts:
        configs.append({
            'rows': rows,
            'cols': cols,
            'density': density,
            'num_pes': num_pes,
            'seed': 42,
            'arch': 'wse3',
            'comment': f'PE scaling: {int(rows*cols*density)} NNZ, {num_pes} PEs'
        })

    return configs


def generate_constant_nnz_sweep(
    target_nnz: int,
    num_pes: int,
    num_points: int = 20
) -> List[Dict]:
    """Generate constant NNZ sweep with varying matrix dimensions.

    Keeps total NNZ constant while varying matrix aspect ratio.
    Useful for: arithmetic intensity analysis, memory access patterns.
    """
    configs = []

    # Vary matrix size from square to increasingly rectangular
    # Keep N*M*density = constant
    for i in range(num_points):
        # Start with square matrix, gradually increase aspect ratio
        ratio = 1.0 + i * 0.5  # 1.0, 1.5, 2.0, 2.5, ...
        total_elements = target_nnz / 0.01  # Assume base density of 0.01

        # N * M = total_elements, M = ratio * N
        # N^2 * ratio = total_elements
        rows = int(np.sqrt(total_elements / ratio))
        cols = int(rows * ratio)
        density = target_nnz / (rows * cols)

        if density > 1.0 or density < 0.0001:
            continue  # Skip invalid configurations

        configs.append({
            'rows': rows,
            'cols': cols,
            'density': float(density),
            'num_pes': num_pes,
            'seed': 42,
            'arch': 'wse3',
            'comment': f'Constant NNZ: {target_nnz}, aspect {ratio:.1f}'
        })

    return configs


def generate_weak_scaling_sweep(
    base_nnz_per_pe: int = 1000,
    pe_start: int = 16,
    pe_end: int = 16384,
    scaling: str = 'power2'
) -> List[Dict]:
    """Generate weak scaling sweep.

    Problem size scales proportionally with PE count.
    Ideal weak scaling: constant time per PE as both problem and PEs grow.
    """
    if scaling == 'power2':
        pe_counts = [2**i for i in range(int(np.log2(pe_start)), int(np.log2(pe_end)) + 1)]
    else:
        pe_counts = list(range(pe_start, pe_end + 1, (pe_end - pe_start) // 20))

    configs = []
    for num_pes in pe_counts:
        target_nnz = base_nnz_per_pe * num_pes

        # Choose matrix size based on target NNZ
        # Assume square matrix with density 0.01
        n = int(np.sqrt(target_nnz / 0.01))
        density = target_nnz / (n * n)

        configs.append({
            'rows': n,
            'cols': n,
            'density': float(density),
            'num_pes': num_pes,
            'seed': 42,
            'arch': 'wse3',
            'comment': f'Weak scaling: {base_nnz_per_pe} NNZ/PE, {num_pes} PEs'
        })

    return configs


def generate_memory_utilization_sweep(
    rows: int,
    cols: int,
    target_utilizations: List[float] = [0.2, 0.4, 0.6, 0.8, 0.95],
    num_pes: int = 512
) -> List[Dict]:
    """Generate sweep targeting specific memory utilization levels.

    Useful for: understanding memory pressure effects, finding optimal utilization.
    Memory per PE ≈ 8 * nnz_per_pe + 4 * rows_per_pe
    """
    configs = []

    for util in target_utilizations:
        target_memory = PE_MEMORY_LIMIT * util

        # Estimate nnz_per_pe for target memory
        # Assume rows distributed evenly: rows_per_pe = rows / num_pes
        rows_per_pe = rows / num_pes
        # target_memory = 8 * nnz_per_pe + 4 * rows_per_pe
        nnz_per_pe = (target_memory - 4 * rows_per_pe) / 8

        total_nnz = int(nnz_per_pe * num_pes)
        density = total_nnz / (rows * cols)

        if density > 1.0 or density < 0.0001:
            continue

        configs.append({
            'rows': rows,
            'cols': cols,
            'density': float(density),
            'num_pes': num_pes,
            'seed': 42,
            'arch': 'wse3',
            'comment': f'Memory util target: {util*100:.0f}%, {num_pes} PEs'
        })

    return configs


def generate_load_balance_study(
    rows: int,
    num_pes_list: List[int] = [16, 32, 64, 128, 256, 512, 1024],
    densities: List[float] = [0.001, 0.01, 0.05, 0.1]
) -> List[Dict]:
    """Generate configurations for load balancing analysis.

    Varies both PE count and density to study load imbalance effects.
    """
    configs = []
    cols = rows  # Square matrices

    for num_pes in num_pes_list:
        for density in densities:
            configs.append({
                'rows': rows,
                'cols': cols,
                'density': density,
                'num_pes': num_pes,
                'seed': 42,
                'arch': 'wse3',
                'comment': f'Load balance: {rows}x{cols}, d={density}, {num_pes} PEs'
            })

    return configs


def generate_large_scale_sweep(
    sizes: List[int] = [10000, 50000, 100000, 200000],
    densities: List[float] = [0.0001, 0.001, 0.01],
    pe_strategy: str = 'auto-min'
) -> List[Dict]:
    """Generate large-scale problem sweep.

    Tests realistic large problems with automatic PE selection.
    """
    configs = []

    for size in sizes:
        for density in densities:
            configs.append({
                'rows': size,
                'cols': size,
                'density': density,
                'num_pes': pe_strategy,  # Let cluster_compile.py decide
                'seed': 42,
                'arch': 'wse3',
                'comment': f'Large scale: {size}x{size}, d={density}, {pe_strategy}'
            })

    return configs


def save_sweep_config(configs: List[Dict], output_path: str):
    """Save sweep configuration to CSV."""
    if not configs:
        print(f"Warning: No configurations to save to {output_path}")
        return

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = ['rows', 'cols', 'density', 'num_pes', 'seed', 'arch', 'comment']

    with open(output_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(configs)

    print(f"✓ Saved {len(configs)} configurations to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Generate comprehensive sweep configurations for cluster benchmarking',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Sweep Types:
  density      - Fixed matrix size and PEs, sweep density (log scale)
  pe-scaling   - Fixed problem size, sweep PE count (strong scaling)
  constant-nnz - Fixed total NNZ, varying matrix dimensions (arithmetic intensity)
  weak-scaling - Problem size scales with PE count (weak scaling)
  memory-util  - Target specific memory utilization levels
  load-balance - Load balancing analysis across PE counts and densities
  large-scale  - Realistic large problems with auto PE selection
  comprehensive - All of the above combined

Examples:
  # Generate density sweep for 10K x 10K matrix with 512 PEs
  python generate_cluster_sweeps.py density --rows 10000 --cols 10000 --num-pes 512

  # Generate PE scaling sweep
  python generate_cluster_sweeps.py pe-scaling --rows 10000 --cols 10000 --density 0.01

  # Generate comprehensive suite for publication
  python generate_cluster_sweeps.py comprehensive
        """
    )

    parser.add_argument('sweep_type',
                       choices=['density', 'pe-scaling', 'constant-nnz', 'weak-scaling',
                               'memory-util', 'load-balance', 'large-scale', 'comprehensive'],
                       help='Type of sweep to generate')

    # Common parameters
    parser.add_argument('--rows', type=int, default=10000, help='Matrix rows')
    parser.add_argument('--cols', type=int, default=10000, help='Matrix columns')
    parser.add_argument('--density', type=float, default=0.01, help='Sparsity density')
    parser.add_argument('--num-pes', type=int, default=512, help='Number of PEs')
    parser.add_argument('--output-dir', type=str, default='tests/sweep-configs/cluster',
                       help='Output directory for sweep configs')

    # Sweep-specific parameters
    parser.add_argument('--num-points', type=int, default=50, help='Number of points in sweep')
    parser.add_argument('--density-min', type=float, default=0.001, help='Min density for sweep')
    parser.add_argument('--density-max', type=float, default=0.1, help='Max density for sweep')

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    configs = []

    if args.sweep_type == 'density':
        configs = generate_density_sweep(
            args.rows, args.cols, args.num_pes,
            args.density_min, args.density_max, args.num_points
        )
        output_file = output_dir / f'density-sweep-{args.rows}x{args.cols}-{args.num_pes}pe.csv'

    elif args.sweep_type == 'pe-scaling':
        configs = generate_pe_scaling_sweep(
            args.rows, args.cols, args.density,
            pe_start=16, pe_end=min(16384, args.rows)  # Don't exceed row count
        )
        output_file = output_dir / f'pe-scaling-{args.rows}x{args.cols}-d{args.density}.csv'

    elif args.sweep_type == 'constant-nnz':
        target_nnz = int(args.rows * args.cols * args.density)
        configs = generate_constant_nnz_sweep(target_nnz, args.num_pes, args.num_points)
        output_file = output_dir / f'constant-nnz-{target_nnz}-{args.num_pes}pe.csv'

    elif args.sweep_type == 'weak-scaling':
        configs = generate_weak_scaling_sweep(
            base_nnz_per_pe=2000,  # ~40% memory utilization
            pe_start=16,
            pe_end=16384
        )
        output_file = output_dir / 'weak-scaling-2000nnz-per-pe.csv'

    elif args.sweep_type == 'memory-util':
        configs = generate_memory_utilization_sweep(
            args.rows, args.cols,
            target_utilizations=[0.2, 0.4, 0.6, 0.8, 0.95],
            num_pes=args.num_pes
        )
        output_file = output_dir / f'memory-util-{args.rows}x{args.cols}-{args.num_pes}pe.csv'

    elif args.sweep_type == 'load-balance':
        configs = generate_load_balance_study(
            args.rows,
            num_pes_list=[16, 32, 64, 128, 256, 512, 1024, 2048, 4096],
            densities=[0.001, 0.005, 0.01, 0.05, 0.1]
        )
        output_file = output_dir / f'load-balance-{args.rows}x{args.rows}.csv'

    elif args.sweep_type == 'large-scale':
        configs = generate_large_scale_sweep(
            sizes=[10000, 50000, 100000, 200000, 500000],
            densities=[0.0001, 0.0005, 0.001, 0.005, 0.01],
            pe_strategy='min'
        )
        output_file = output_dir / 'large-scale-auto-pe.csv'

    elif args.sweep_type == 'comprehensive':
        print("Generating comprehensive benchmark suite...")

        # 1. Density sweeps at multiple scales
        for size in [10000, 50000, 100000]:
            for num_pes in [512, 2048, 8192]:
                if num_pes <= size:  # Don't exceed row count
                    configs.extend(generate_density_sweep(size, size, num_pes, num_points=30))
        save_sweep_config(configs, output_dir / 'comprehensive-01-density-sweeps.csv')
        configs = []

        # 2. Strong scaling studies
        for size in [10000, 50000, 100000]:
            for density in [0.001, 0.01, 0.05]:
                configs.extend(generate_pe_scaling_sweep(size, size, density))
        save_sweep_config(configs, output_dir / 'comprehensive-02-strong-scaling.csv')
        configs = []

        # 3. Weak scaling
        configs.extend(generate_weak_scaling_sweep(base_nnz_per_pe=2000))
        save_sweep_config(configs, output_dir / 'comprehensive-03-weak-scaling.csv')
        configs = []

        # 4. Constant NNZ (arithmetic intensity)
        for nnz in [100000, 1000000, 10000000]:
            configs.extend(generate_constant_nnz_sweep(nnz, num_pes=2048, num_points=15))
        save_sweep_config(configs, output_dir / 'comprehensive-04-arithmetic-intensity.csv')
        configs = []

        # 5. Memory utilization study
        configs.extend(generate_memory_utilization_sweep(10000, 10000, num_pes=512))
        configs.extend(generate_memory_utilization_sweep(50000, 50000, num_pes=2048))
        save_sweep_config(configs, output_dir / 'comprehensive-05-memory-utilization.csv')
        configs = []

        # 6. Load balancing analysis
        configs.extend(generate_load_balance_study(10000))
        configs.extend(generate_load_balance_study(50000))
        save_sweep_config(configs, output_dir / 'comprehensive-06-load-balance.csv')

        print("\n✓ Comprehensive suite generated!")
        return

    if configs:
        save_sweep_config(configs, output_file)


if __name__ == '__main__':
    main()
