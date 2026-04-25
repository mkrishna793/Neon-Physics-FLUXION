#!/usr/bin/env python3
"""
FLUXION Industry Benchmark CLI

Runs FLUXION on industry-standard benchmarks:
  - ISPD 2005/2006 (GSRC Bookshelf)
  - ICCAD 2014/2015 (LEF/DEF)
  - IWLS 2005+ (BLIF)

Usage:
    python benchmarks/run_industry_benchmarks.py --file path/to/design.aux --steps 10000
    python benchmarks/run_industry_benchmarks.py --dir path/to/benchmarks/ --format bookshelf
    python benchmarks/run_industry_benchmarks.py --generate-sample --steps 2000
"""

import sys
import os
import argparse
import time
import json
import numpy as np
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "python"))

from fluxion.benchmarks.benchmark_runner import BenchmarkRunner, BenchmarkMetrics
from fluxion.benchmarks.bookshelf_parser import BookshelfParser
from fluxion.benchmarks.blif_parser import BLIFParser
from fluxion.particle_system import CircuitParticles, FluxionParticle, FluxionConnection
from fluxion.tpe import ThermodynamicPlacementEngine, PlacementConfig


def generate_sample_bookshelf(output_dir: str, num_cells: int = 200):
    """Generate a sample Bookshelf benchmark for testing."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    np.random.seed(42)
    name = "fluxion_sample"

    # .aux
    with open(out / f"{name}.aux", 'w') as f:
        f.write(f"RowBasedPlacement : {name}.nodes {name}.nets {name}.pl {name}.scl\n")

    # .nodes
    with open(out / f"{name}.nodes", 'w') as f:
        f.write(f"UCLA nodes 1.0\n")
        f.write(f"NumNodes : {num_cells}\n")
        f.write(f"NumTerminals : 0\n")
        for i in range(num_cells):
            w = np.random.choice([2, 4, 6, 12])
            h = 4
            f.write(f"cell_{i}\t{w}\t{h}\n")

    # .nets
    num_nets = int(num_cells * 1.2)
    with open(out / f"{name}.nets", 'w') as f:
        f.write(f"UCLA nets 1.0\n")
        f.write(f"NumNets : {num_nets}\n")
        f.write(f"NumPins : {num_nets * 2}\n")
        for i in range(num_nets):
            src = np.random.randint(0, num_cells)
            dst = np.random.randint(0, num_cells)
            while dst == src:
                dst = np.random.randint(0, num_cells)
            f.write(f"NetDegree : 2 net_{i}\n")
            f.write(f"cell_{src} O\n")
            f.write(f"cell_{dst} I\n")

    # .pl
    die_size = int(np.sqrt(num_cells) * 20)
    with open(out / f"{name}.pl", 'w') as f:
        f.write(f"UCLA pl 1.0\n")
        for i in range(num_cells):
            x = np.random.randint(0, die_size)
            y = np.random.randint(0, die_size)
            f.write(f"cell_{i}\t{x}\t{y}\t: N\n")

    # .scl
    row_height = 4
    num_rows = die_size // row_height
    with open(out / f"{name}.scl", 'w') as f:
        f.write(f"UCLA scl 1.0\n")
        f.write(f"NumRows : {num_rows}\n")
        for r in range(num_rows):
            f.write(f"CoreRow Horizontal\n")
            f.write(f"  Coordinate : {r * row_height}\n")
            f.write(f"  Height : {row_height}\n")
            f.write(f"  Sitewidth : 1\n")
            f.write(f"  NumSites : {die_size}\n")
            f.write(f"  SubrowOrigin : 0\n")
            f.write(f"End\n")

    print(f"Generated sample benchmark in {out}/")
    print(f"  Cells: {num_cells}, Nets: {num_nets}, Die: {die_size}x{die_size}")
    return str(out / f"{name}.aux")


def generate_sample_blif(output_dir: str, num_gates: int = 100):
    """Generate a sample BLIF benchmark for testing."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    np.random.seed(42)
    name = "fluxion_sample"

    with open(out / f"{name}.blif", 'w') as f:
        n_inputs = max(4, num_gates // 10)
        n_outputs = max(2, num_gates // 20)

        f.write(f".model {name}\n")
        f.write(f".inputs {' '.join(f'in_{i}' for i in range(n_inputs))}\n")
        f.write(f".outputs {' '.join(f'out_{i}' for i in range(n_outputs))}\n\n")

        # Combinational gates
        for i in range(num_gates):
            n_in = np.random.choice([1, 2, 2, 2, 3])
            inputs = []
            for _ in range(n_in):
                if i < n_inputs:
                    inputs.append(f"in_{np.random.randint(0, n_inputs)}")
                else:
                    inputs.append(f"w_{np.random.randint(0, i)}")
            f.write(f".names {' '.join(inputs)} w_{i}\n")
            f.write("1" * n_in + " 1\n\n")

        # Connect to outputs
        for i in range(n_outputs):
            src = np.random.randint(num_gates // 2, num_gates)
            f.write(f".names w_{src} out_{i}\n")
            f.write("1 1\n\n")

        f.write(".end\n")

    print(f"Generated sample BLIF in {out}/{name}.blif")
    return str(out / f"{name}.blif")


def main():
    parser = argparse.ArgumentParser(
        description="FLUXION Industry Benchmark Runner"
    )
    parser.add_argument("--file", type=str, help="Path to a single benchmark file")
    parser.add_argument("--dir", type=str, help="Directory of benchmark files")
    parser.add_argument("--format", type=str, choices=["bookshelf", "lefdef", "blif"],
                        help="Force benchmark format")
    parser.add_argument("--steps", type=int, default=5000, help="Annealing steps")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--discovery", action="store_true",
                        help="Use discovery mode (reheating + Lévy flights)")
    parser.add_argument("--adaptive", action="store_true", default=True,
                        help="Use adaptive weight scheduling")
    parser.add_argument("--output", type=str, default=None,
                        help="Save results JSON to this path")
    parser.add_argument("--generate-sample", action="store_true",
                        help="Generate sample benchmarks and run them")
    parser.add_argument("--sample-format", type=str, default="bookshelf",
                        choices=["bookshelf", "blif"],
                        help="Format for generated sample")
    parser.add_argument("--sample-size", type=int, default=200,
                        help="Number of cells/gates in generated sample")

    args = parser.parse_args()

    runner = BenchmarkRunner(
        steps=args.steps,
        verbose=True,
        discovery_mode=args.discovery,
        adaptive_weights=args.adaptive,
        seed=args.seed,
    )

    all_metrics = []

    if args.generate_sample:
        sample_dir = Path(__file__).parent / "samples"
        if args.sample_format == "bookshelf":
            bench_path = generate_sample_bookshelf(str(sample_dir), args.sample_size)
            metrics = runner.run(bench_path, format="bookshelf")
        else:
            bench_path = generate_sample_blif(str(sample_dir), args.sample_size)
            metrics = runner.run(bench_path, format="blif")
        all_metrics.append(metrics)

    elif args.file:
        metrics = runner.run(args.file, format=args.format)
        all_metrics.append(metrics)

    elif args.dir:
        bench_dir = Path(args.dir)
        # Find all benchmark files
        patterns = ["*.aux", "*.def", "*.blif"]
        files = []
        for pat in patterns:
            files.extend(bench_dir.glob(pat))
        files.sort()

        if not files:
            print(f"No benchmark files found in {bench_dir}")
            return 1

        all_metrics = runner.run_suite([str(f) for f in files], format=args.format)

    else:
        parser.print_help()
        return 1

    # Print summary
    if all_metrics:
        BenchmarkRunner.print_results_table(all_metrics)

        if args.output:
            BenchmarkRunner.save_results(all_metrics, args.output)
            print(f"\nResults saved to {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
