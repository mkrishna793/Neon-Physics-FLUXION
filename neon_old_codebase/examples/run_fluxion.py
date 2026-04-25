#!/usr/bin/env python3
"""
FLUXION Runner Script

This script demonstrates the complete FLUXION workflow:
1. Parse Verilog with Verilator (simulated for demo)
2. Export circuit particles
3. Run thermodynamic placement
4. Output optimized placement

For actual Verilator integration, compile the V3FluxionExport pass
and run Verilator with --fluxion-export flag.
"""

import sys
import json
import argparse
from pathlib import Path
import numpy as np

# Add parent directory to path for local development
sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "python"))

from fluxion import (
    ThermodynamicPlacementEngine,
    PlacementConfig,
    CircuitParticles,
    FluxionParticle,
    FluxionConnection,
    CriticalPath,
)


def create_demo_circuit_from_verilog(verilog_file=None, num_gates=100):
    """
    Create a demo circuit representing Verilator output.

    In production, this would be replaced by actual Verilator parsing
    with the V3FluxionExport pass.

    Args:
        verilog_file: Path to Verilog file (for reference)
        num_gates: Number of gates to simulate

    Returns:
        CircuitParticles object
    """
    print(f"Creating demo circuit with {num_gates} gates...")

    circuit = CircuitParticles(
        module_name="demo_circuit",
        die_width=1000.0,
        die_height=1000.0,
    )
    circuit.target_clock_period_ps = 1000.0  # 1ns clock

    # Gate type distribution (typical ASIC)
    gate_types = [
        ("NAND", 30),  # 30% NAND gates
        ("NOR", 20),   # 20% NOR gates
        ("AND", 10),   # 10% AND gates
        ("OR", 10),    # 10% OR gates
        ("XOR", 10),   # 10% XOR gates
        ("DFF", 15),   # 15% D flip-flops
        ("MUX", 5),    # 5% multiplexers
    ]

    # Create gates
    np.random.seed(42)  # Reproducibility

    gate_id = 0
    for gtype, count_ratio in gate_types:
        count = int(num_gates * count_ratio / 100)
        for _ in range(count):
            # Properties based on gate type
            if gtype == "DFF":
                delay = np.random.uniform(40, 80)
                area = np.random.uniform(8, 15)
                power = np.random.uniform(80, 200)
            elif gtype == "MUX":
                delay = np.random.uniform(15, 30)
                area = np.random.uniform(5, 12)
                power = np.random.uniform(15, 40)
            elif gtype == "XOR":
                delay = np.random.uniform(10, 20)
                area = np.random.uniform(3, 6)
                power = np.random.uniform(8, 15)
            else:  # Basic gates
                delay = np.random.uniform(5, 15)
                area = np.random.uniform(1.5, 4)
                power = np.random.uniform(3, 10)

            particle = FluxionParticle(
                id=gate_id,
                name=f"{gtype.lower()}_{gate_id}",
                type=gtype,
                power_pw=power,
                area_um2=area,
                delay_ps=delay,
            )
            circuit.add_particle(particle)
            gate_id += 1

    # Create connections (realistic connectivity)
    # Primary inputs feed first level
    # Consecutive levels connect forward
    # Some feedback paths for sequential logic

    gate_ids = list(circuit.particles.keys())
    np.random.shuffle(gate_ids)

    # Forward connections
    connectivity = 0.08  # Average fanout
    for i, src_id in enumerate(gate_ids):
        # Each gate has ~connectivity * N outputs
        num_outputs = int(np.random.poisson(connectivity * num_gates))
        num_outputs = min(num_outputs, 5)  # Cap at 5 outputs

        for _ in range(num_outputs):
            # Prefer forward connections
            dst_idx = min(i + np.random.randint(1, 10), len(gate_ids) - 1)
            dst_id = gate_ids[dst_idx]

            if src_id != dst_id:
                conn = FluxionConnection(
                    source_id=src_id,
                    dest_id=dst_id,
                    name=f"net_{len(circuit.connections)}",
                    is_critical_path=(np.random.random() < 0.1),  # 10% critical
                )
                circuit.add_connection(conn)

    # Create a few critical paths
    for path_id in range(3):
        path_start = np.random.randint(0, len(gate_ids) // 2)
        path_end = min(path_start + np.random.randint(5, 15), len(gate_ids) - 1)
        path_nodes = gate_ids[path_start:path_end:path_id + 1]

        total_delay = sum(
            circuit.particles[nid].delay_ps for nid in path_nodes
        )

        circuit.add_critical_path(CriticalPath(
            node_ids=path_nodes,
            total_delay_ps=total_delay,
            slack_ps=100.0 - path_id * 20,  # Varying slack
        ))

    # Compute statistics
    circuit.compute_statistics()

    print(f"  Created {len(circuit.particles)} gates")
    print(f"  Created {len(circuit.connections)} connections")
    print(f"  Created {len(circuit.critical_paths)} critical paths")

    return circuit


def run_fluxion(circuit, config):
    """Run FLUXION optimization."""
    print("\nRunning FLUXION Thermodynamic Placement Engine...")
    print("-" * 60)

    engine = ThermodynamicPlacementEngine(config)
    engine.set_circuit(circuit)

    result = engine.optimize()

    return result, engine


def main():
    parser = argparse.ArgumentParser(
        description="Run FLUXION placement optimization"
    )
    parser.add_argument(
        "-i", "--input",
        type=str,
        help="Input Verilog file (for reference)",
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default="fluxion_output",
        help="Output directory",
    )
    parser.add_argument(
        "-n", "--num-gates",
        type=int,
        default=100,
        help="Number of gates for demo circuit",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=5000,
        help="Annealing steps",
    )
    parser.add_argument(
        "--die-width",
        type=float,
        default=1000.0,
        help="Die width in micrometers",
    )
    parser.add_argument(
        "--die-height",
        type=float,
        default=1000.0,
        help="Die height in micrometers",
    )
    parser.add_argument(
        "--clock",
        type=float,
        default=1000.0,
        help="Target clock period in picoseconds",
    )
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="Use GPU acceleration",
    )
    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Suppress output",
    )
    parser.add_argument(
        "--legalize",
        action="store_true",
        help="Run hybrid legalizer after placement",
    )
    parser.add_argument(
        "--def-output",
        action="store_true",
        help="Export placement to DEF format",
    )
    parser.add_argument(
        "--tech-node",
        type=str,
        default="7nm",
        help="Tech node for grid",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create circuit
    circuit = create_demo_circuit_from_verilog(
        verilog_file=args.input,
        num_gates=args.num_gates,
    )

    # Configure placement
    config = PlacementConfig(
        die_width=args.die_width,
        die_height=args.die_height,
        target_clock_period_ps=args.clock,
        annealing_steps=args.steps,
        initial_temperature=100.0,
        final_temperature=0.01,
        cooling_rate=0.95,
        random_seed=args.seed,
        use_gpu=args.gpu,
        verbose=not args.quiet,
        legalize=args.legalize,
        output_def=args.def_output,
        tech_node=args.tech_node,
    )

    # Run optimization
    result, engine = run_fluxion(circuit, config)

    # Save results
    print("\nSaving results...")

    # Save circuit particles
    circuit_file = output_dir / "circuit_particles.json"
    circuit.save(str(circuit_file))
    print(f"  Circuit: {circuit_file}")

    # Save placement result
    result_file = output_dir / "placement_result.json"
    engine.save_result(str(result_file))
    print(f"  Result: {result_file}")

    # Save optimized positions as simple list
    positions_file = output_dir / "positions.csv"
    with open(positions_file, "w") as f:
        f.write("id,name,type,x,y\n")
        for i, particle in enumerate(circuit.particles.values()):
            f.write(f"{particle.id},{particle.name},{particle.type},"
                   f"{particle.x:.4f},{particle.y:.4f}\n")
    print(f"  Positions: {positions_file}")

    if result.def_file_path:
        import os
        import shutil
        dest = output_dir / "placement.def"
        if os.path.exists(result.def_file_path):
            shutil.move(result.def_file_path, str(dest))
            print(f"  DEF Export: {dest}")

    # Summary
    print("\n" + "=" * 60)
    print("FLUXION Placement Summary")
    print("=" * 60)
    print(f"Gates: {len(circuit.particles)}")
    print(f"Connections: {len(circuit.connections)}")
    print(f"Total Wirelength: {result.total_wirelength:.2f} um")
    print(f"Max Temperature: {result.max_temperature:.2f} K")
    print(f"Critical Path Delay: {result.critical_path_delay:.2f} ps")
    print(f"Optimization Time: {result.annealing_time:.2f} s")
    
    if result.legalizer_stats:
        print("-" * 60)
        s = result.legalizer_stats
        print("Legalization Results:")
        print(f"  Tetris placed: {s['tetris_success']}")
        print(f"  Z3 resolved:   {s['z3_resolved']}")
        print(f"  Failed:        {s['failed_illegal']}")
        print(f"  Legalize Time: {s['time_s']:.2f} s")
        
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())