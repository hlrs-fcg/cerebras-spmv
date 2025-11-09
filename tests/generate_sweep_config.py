#!/usr/bin/env python3
"""
Generate comprehensive sweep configuration CSV for SpMV benchmarks.

Creates test configurations with:
- Same NNZ across different matrix dimensions
- Various matrix sizes and densities
- Different PE configurations (min/max)
"""

import csv
import argparse
from pathlib import Path
from typing import List, Dict
import math


def generate_same_nnz_configs(target_nnz: int, sizes: List[int], arch: str = 'wse3') -> List[Dict[str, str]]:
    """
    Generate configurations with same target NNZ but different matrix dimensions.

    For square matrices, density = target_nnz / (rows * cols)

    Args:
        target_nnz: Target number of non-zero elements
        sizes: List of matrix dimensions (for square matrices)
        arch: Architecture (wse2 or wse3)

    Returns:
        List of configuration dictionaries
    """
    configs = []

    for size in sizes:
        rows = cols = size
        total_elements = rows * cols

        # Calculate required density to get approximately target_nnz
        density = target_nnz / total_elements

        # Skip if density is too high (>1) or too low (<0.0001)
        if density > 1.0:
            continue
        if density < 0.0001:
            continue

        # Round density to reasonable precision
        density = round(density, 6)

        # Generate configs for min and max PE configurations
        for num_pes in ['min', 'max']:
            configs.append({
                'rows': str(rows),
                'cols': str(cols),
                'density': str(density),
                'num_pes': num_pes,
                'seed': '42',
                'arch': arch,
                'comment': f'~{target_nnz} NNZ target, {size}x{size} matrix'
            })

    return configs


def generate_density_sweep_configs(base_size: int, densities: List[float], arch: str = 'wse3') -> List[Dict[str, str]]:
    """
    Generate configurations sweeping density for a fixed matrix size.

    Args:
        base_size: Matrix dimension (square matrix)
        densities: List of density values to test
        arch: Architecture

    Returns:
        List of configuration dictionaries
    """
    configs = []

    for density in densities:
        for num_pes in ['min', 'max']:
            expected_nnz = int(base_size * base_size * density)
            configs.append({
                'rows': str(base_size),
                'cols': str(base_size),
                'density': str(density),
                'num_pes': num_pes,
                'seed': '42',
                'arch': arch,
                'comment': f'{base_size}x{base_size}, ~{expected_nnz} NNZ'
            })

    return configs


def generate_size_sweep_configs(sizes: List[int], density: float, arch: str = 'wse3') -> List[Dict[str, str]]:
    """
    Generate configurations sweeping size for a fixed density.

    Args:
        sizes: List of matrix dimensions
        density: Fixed density value
        arch: Architecture

    Returns:
        List of configuration dictionaries
    """
    configs = []

    for size in sizes:
        for num_pes in ['min', 'max']:
            expected_nnz = int(size * size * density)
            configs.append({
                'rows': str(size),
                'cols': str(size),
                'density': str(density),
                'num_pes': num_pes,
                'seed': '42',
                'arch': arch,
                'comment': f'{size}x{size}, density={density}, ~{expected_nnz} NNZ'
            })

    return configs


def main():
    parser = argparse.ArgumentParser(
        description='Generate comprehensive sweep configuration CSV',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate default configuration
  python tests/generate_sweep_config.py -o tests/sweep_config.csv

  # Generate with custom NNZ target
  python tests/generate_sweep_config.py --target-nnz 50000 -o tests/sweep_large.csv

  # Generate multiple test suites
  python tests/generate_sweep_config.py --mode same-nnz -o tests/same_nnz.csv
  python tests/generate_sweep_config.py --mode density-sweep -o tests/density_sweep.csv
  python tests/generate_sweep_config.py --mode size-sweep -o tests/size_sweep.csv
        """
    )

    parser.add_argument('-o', '--output', type=str, default='tests/sweep_config.csv',
                       help='Output CSV file path')
    parser.add_argument('--mode', type=str, default='comprehensive',
                       choices=['same-nnz', 'density-sweep', 'size-sweep', 'comprehensive'],
                       help='Configuration generation mode')
    parser.add_argument('--target-nnz', type=int, default=10000,
                       help='Target NNZ for same-nnz mode')
    parser.add_argument('--arch', type=str, default='wse3',
                       choices=['wse2', 'wse3'],
                       help='Target architecture')

    args = parser.parse_args()

    all_configs = []

    if args.mode == 'same-nnz':
        # Generate configs with same NNZ but different dimensions
        # For ~10000 NNZ:
        # - 1000x1000 with density 0.01 → 10000 NNZ
        # - 2000x2000 with density 0.0025 → 10000 NNZ
        # - 5000x5000 with density 0.0004 → 10000 NNZ
        # - 10000x10000 with density 0.0001 → 10000 NNZ

        sizes = [1000, 1414, 2000, 2828, 4000, 5657, 8000, 10000, 14142, 20000]
        all_configs.extend(generate_same_nnz_configs(args.target_nnz, sizes, args.arch))

    elif args.mode == 'density-sweep':
        # Sweep density for a fixed size
        base_size = 5000
        densities = [0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1]
        all_configs.extend(generate_density_sweep_configs(base_size, densities, args.arch))

    elif args.mode == 'size-sweep':
        # Sweep size for a fixed density
        sizes = [1000, 2000, 3000, 5000, 7000, 10000, 15000, 20000, 30000, 50000]
        density = 0.001
        all_configs.extend(generate_size_sweep_configs(sizes, density, args.arch))

    elif args.mode == 'comprehensive':
        # Generate all three types

        # 1. Same NNZ, different dimensions (~10000 NNZ)
        sizes_10k = [1000, 1414, 2000, 2828, 4000, 5657, 8000, 10000]
        all_configs.extend(generate_same_nnz_configs(10000, sizes_10k, args.arch))

        # 2. Density sweep (5000x5000 matrix)
        densities = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05]
        all_configs.extend(generate_density_sweep_configs(5000, densities, args.arch))

        # 3. Size sweep (density=0.001)
        sizes = [1000, 2000, 5000, 10000, 20000]
        all_configs.extend(generate_size_sweep_configs(sizes, 0.001, args.arch))

    # Write to CSV
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = ['rows', 'cols', 'density', 'num_pes', 'seed', 'arch', 'comment']

    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_configs)

    print(f"Generated {len(all_configs)} configurations")
    print(f"Saved to: {output_path.resolve()}")
    print()
    print("Configuration breakdown:")

    # Count by mode
    if args.mode == 'comprehensive':
        same_nnz_count = len([c for c in all_configs if 'NNZ target' in c.get('comment', '')])
        density_sweep_count = len([c for c in all_configs if 'density=' in c.get('comment', '') and 'NNZ target' not in c.get('comment', '')])
        size_sweep_count = len(all_configs) - same_nnz_count - density_sweep_count

        print(f"  Same NNZ tests:      {same_nnz_count}")
        print(f"  Density sweep tests: {density_sweep_count}")
        print(f"  Size sweep tests:    {size_sweep_count}")

    print()
    print("Run with:")
    print(f"  python tests/sweep.py {output_path} --continue-on-error")


if __name__ == "__main__":
    main()
