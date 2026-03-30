#!/usr/bin/env python3
"""
FLUXION Command Line Interface

Main entry point for running FLUXION from the command line.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

from .tpe import ThermodynamicPlacementEngine, PlacementConfig, PlacementResult
from .particle_system import CircuitParticles, load_circuit_particles


def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser."""
    parser = argparse.ArgumentParser(
        prog="fluxion",
        description="FLUXION - Physics-Native Chip Placement Engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run placement on a circuit
  fluxion optimize circuit_particles.json -o output.json

  # Quick placement with fewer steps
  fluxion optimize circuit.json --fast -o output.json

  # Analyze a circuit
  fluxion analyze circuit_particles.json

  # Validate placement result
  fluxion validate result.json

For more information, visit: https://github.com/fluxion-project/fluxion
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Optimize command
    opt_parser = subparsers.add_parser(
        "optimize",
        help="Run placement optimization",
        description="Run the thermodynamic placement engine on a circuit",
    )
    opt_parser.add_argument("input", type=str, help="Input circuit file (JSON)")
    opt_parser.add_argument(
        "-o", "--output",
        type=str,
        default="placement_result.json",
        help="Output file for placement result (default: placement_result.json)",
    )
    opt_parser.add_argument(
        "--fast",
        action="store_true",
        help="Run fast optimization with fewer steps",
    )
    opt_parser.add_argument(
        "--steps",
        type=int,
        default=10000,
        help="Number of annealing steps (default: 10000)",
    )
    opt_parser.add_argument(
        "--initial-temp",
        type=float,
        default=1000.0,
        help="Initial temperature (default: 1000.0)",
    )
    opt_parser.add_argument(
        "--final-temp",
        type=float,
        default=0.01,
        help="Final temperature (default: 0.01)",
    )
    opt_parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    opt_parser.add_argument(
        "--no-gpu",
        action="store_true",
        help="Disable GPU acceleration",
    )
    opt_parser.add_argument(
        "--wire-weight",
        type=float,
        default=1.0,
        help="Wire tension force weight (default: 1.0)",
    )
    opt_parser.add_argument(
        "--thermal-weight",
        type=float,
        default=0.5,
        help="Thermal repulsion force weight (default: 0.5)",
    )
    opt_parser.add_argument(
        "--timing-weight",
        type=float,
        default=0.8,
        help="Timing gravity force weight (default: 0.8)",
    )
    opt_parser.add_argument(
        "--topoloss-weight",
        type=float,
        default=0.3,
        help="TopoLoss force weight (default: 0.3)",
    )
    opt_parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Suppress output",
    )

    # Analyze command
    analyze_parser = subparsers.add_parser(
        "analyze",
        help="Analyze a circuit",
        description="Analyze circuit statistics without running optimization",
    )
    analyze_parser.add_argument("input", type=str, help="Input circuit file (JSON)")
    analyze_parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Show detailed statistics",
    )

    # Validate command
    validate_parser = subparsers.add_parser(
        "validate",
        help="Validate a placement result",
        description="Validate that a placement result meets constraints",
    )
    validate_parser.add_argument("input", type=str, help="Placement result file (JSON)")
    validate_parser.add_argument(
        "--circuit",
        type=str,
        help="Original circuit file for comparison",
    )

    # Verify command (Phase 8 logic structural integrity)
    verify_parser = subparsers.add_parser(
        "verify",
        help="Run Phase 8 Verilator validation",
        description="Verify layout logic remains unbroken by checking output against original Verilog",
    )
    verify_parser.add_argument("optimized_json", type=str, help="Optimized placement result (JSON)")
    verify_parser.add_argument(
        "--original",
        type=str,
        required=True,
        help="Original Verilog (.v) file",
    )
    verify_parser.add_argument(
        "--verilator-path",
        type=str,
        default="verilator",
        help="Path to verilator executable",
    )

    # Generate command
    generate_parser = subparsers.add_parser(
        "generate",
        help="Generate a test circuit",
        description="Generate a random test circuit for testing",
    )
    generate_parser.add_argument(
        "-n", "--num-gates",
        type=int,
        default=100,
        help="Number of gates (default: 100)",
    )
    generate_parser.add_argument(
        "-c", "--connectivity",
        type=float,
        default=0.1,
        help="Connectivity factor (default: 0.1)",
    )
    generate_parser.add_argument(
        "-o", "--output",
        type=str,
        default="test_circuit.json",
        help="Output file (default: test_circuit.json)",
    )
    generate_parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )

    # Info command
    info_parser = subparsers.add_parser(
        "info",
        help="Show FLUXION information",
        description="Display information about FLUXION and system",
    )
    info_parser.add_argument(
        "--check-gpu",
        action="store_true",
        help="Check GPU availability",
    )

    return parser


def cmd_optimize(args) -> int:
    """Run the optimize command."""
    # Load circuit
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file not found: {args.input}", file=sys.stderr)
        return 1

    print(f"Loading circuit from {args.input}")
    circuit = load_circuit_particles(str(input_path))

    # Configure engine
    config = PlacementConfig(
        die_width=circuit.die_width,
        die_height=circuit.die_height,
        target_clock_period_ps=circuit.target_clock_period_ps,
        annealing_steps=args.steps,
        initial_temperature=args.initial_temp,
        final_temperature=args.final_temp,
        random_seed=args.seed,
        use_gpu=not args.no_gpu,
        verbose=not args.quiet,
        wire_tension_weight=args.wire_weight,
        thermal_repulsion_weight=args.thermal_weight,
        timing_gravity_weight=args.timing_weight,
        topoloss_weight=args.topoloss_weight,
    )

    # Create and run engine
    engine = ThermodynamicPlacementEngine(config)
    engine.set_circuit(circuit)

    if args.fast:
        result = engine.fast_optimize(steps=min(args.steps // 2, 5000))
    else:
        result = engine.optimize()

    # Save result
    engine.save_result(args.output)
    engine.save_circuit(args.output.replace(".json", "_circuit.json"))

    print(f"\nResult saved to {args.output}")

    return 0


def cmd_analyze(args) -> int:
    """Run the analyze command."""
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file not found: {args.input}", file=sys.stderr)
        return 1

    print(f"Analyzing {args.input}")
    circuit = load_circuit_particles(str(input_path))

    print("\nCircuit Statistics")
    print("=" * 50)
    print(f"Module: {circuit.module_name}")
    print(f"Gates: {len(circuit.particles)}")
    print(f"Connections: {len(circuit.connections)}")
    print(f"Critical paths: {len(circuit.critical_paths)}")
    print(f"Die size: {circuit.die_width} x {circuit.die_height} um")
    print(f"Target clock: {circuit.target_clock_period_ps} ps")

    if args.verbose:
        print("\nDetailed Statistics")
        print("-" * 50)
        print(f"Total gates: {circuit.total_gates}")
        print(f"Total nets: {circuit.total_nets}")
        print(f"Total power: {circuit.total_power_pw} pW")
        print(f"Total area: {circuit.total_area_um2} um²")
        print(f"Max logic level: {circuit.max_logic_level}")

        # Gate type distribution
        type_counts = {}
        for p in circuit.particles.values():
            type_counts[p.type] = type_counts.get(p.type, 0) + 1

        print("\nGate type distribution:")
        for gtype, count in sorted(type_counts.items()):
            print(f"  {gtype}: {count}")

    return 0


def cmd_validate(args) -> int:
    """Run the validate command."""
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file not found: {args.input}", file=sys.stderr)
        return 1

    print(f"Validating {args.input}")
    result = PlacementResult.load(str(input_path))

    print("\nPlacement Result")
    print("=" * 50)
    print(f"Total energy: {result.total_energy:.2f}")
    print(f"Wire energy: {result.wire_energy:.2f}")
    print(f"Thermal energy: {result.thermal_energy:.2f}")
    print(f"Timing energy: {result.timing_energy:.2f}")
    print(f"TopoLoss energy: {result.topoloss_energy:.2f}")
    print(f"Total wirelength: {result.total_wirelength:.2f} um")
    print(f"Max temperature: {result.max_temperature:.2f} K")
    print(f"Critical path delay: {result.critical_path_delay:.2f} ps")
    print(f"Annealing time: {result.annealing_time:.2f} s")
    print(f"Acceptance rate: {result.acceptance_rate:.2%}")

    # Validation checks
    warnings = []

    if result.max_temperature > 400:  # 127°C
        warnings.append(f"High temperature: {result.max_temperature:.1f} K (max recommended: 400 K)")

    if result.acceptance_rate < 0.1:
        warnings.append(f"Low acceptance rate: {result.acceptance_rate:.1%}")

    if warnings:
        print("\nWarnings:")
        for warning in warnings:
            print(f"  ⚠ {warning}")
    else:
        print("\n✓ All checks passed")

    return 0


def cmd_verify(args) -> int:
    """Run the verify command (Phase 8)."""
    from .verify import VerilatorVerifyLoop
    print(f"Verifying {args.optimized_json} against {args.original}")
    verifier = VerilatorVerifyLoop(verilator_path=args.verilator_path)
    success = verifier.verify_topology(args.original, args.optimized_json)
    if success:
        return 0
    else:
        print("Verification failed!", file=sys.stderr)
        return 1




def cmd_generate(args) -> int:
    """Run the generate command."""
    import numpy as np

    print(f"Generating test circuit with {args.num_gates} gates")
    np.random.seed(args.seed)

    circuit = CircuitParticles(
        module_name="test_circuit",
        die_width=1000.0,
        die_height=1000.0,
    )

    # Generate gates
    gate_types = ["NAND", "NOR", "AND", "OR", "XOR", "DFF", "MUX"]
    for i in range(args.num_gates):
        gate_type = np.random.choice(gate_types)

        # Random properties based on gate type
        if gate_type in ["DFF"]:
            delay = np.random.uniform(40, 80)
            area = np.random.uniform(8, 15)
            power = np.random.uniform(50, 200)
        elif gate_type in ["MUX"]:
            delay = np.random.uniform(15, 30)
            area = np.random.uniform(4, 10)
            power = np.random.uniform(10, 30)
        else:
            delay = np.random.uniform(5, 20)
            area = np.random.uniform(1, 5)
            power = np.random.uniform(2, 15)

        particle = FluxionParticle(
            id=i,
            name=f"gate_{i}",
            type=gate_type,
            power_pw=power,
            area_um2=area,
            delay_ps=delay,
        )
        circuit.add_particle(particle)

    # Generate connections
    conn_count = 0
    for i in range(args.num_gates):
        # Each gate connects to ~connectivity * num_gates others
        num_outputs = int(args.connectivity * args.num_gates * np.random.uniform(0.5, 1.5))
        num_outputs = min(num_outputs, args.num_gates - i - 1)  # Can only connect forward
        num_outputs = max(num_outputs, 0)

        if num_outputs > 0:
            dests = np.random.choice(
                range(i + 1, args.num_gates),
                size=min(num_outputs, args.num_gates - i - 1),
                replace=False
            )

            for dest in dests:
                conn = FluxionConnection(
                    source_id=i,
                    dest_id=int(dest),
                    name=f"net_{conn_count}",
                    is_critical_path=(np.random.random() < 0.1),  # 10% are critical
                )
                circuit.add_connection(conn)
                conn_count += 1

    # Generate a critical path
    if args.num_gates > 5:
        path_length = min(10, args.num_gates // 5)
        critical_nodes = list(range(0, args.num_gates, args.num_gates // path_length))[:path_length]
        circuit.add_critical_path(CriticalPath(
            node_ids=critical_nodes,
            total_delay_ps=sum(
                circuit.particles[i].delay_ps for i in critical_nodes
            ),
        ))

    # Save
    circuit.save(args.output)
    print(f"Circuit saved to {args.output}")
    print(f"  Gates: {len(circuit.particles)}")
    print(f"  Connections: {len(circuit.connections)}")

    return 0


def cmd_info(args) -> int:
    """Run the info command."""
    print("=" * 60)
    print("FLUXION - Physics-Native Silicon Intelligence")
    print("Thermodynamic Placement Engine v1.0.0")
    print("=" * 60)
    print()
    print("FLUXION is an open-source, physics-native chip placement engine.")
    print("It treats every logic gate as a physical particle and uses")
    print("thermodynamic forces to arrange them into optimal layouts.")
    print()
    print("Features:")
    print("  • Wire Tension - Minimizes wire length")
    print("  • Thermal Repulsion - Spreads heat across die")
    print("  • Timing Gravity - Optimizes critical paths")
    print("  • TopoLoss - Preserves circuit topology")
    print()
    print("Built on Verilator for trusted, industry-standard processing.")
    print()

    if args.check_gpu:
        print("Checking GPU availability...")
        try:
            from .gpu_accelerator import create_accelerator
            accelerator = create_accelerator()
            if accelerator.is_available():
                print("✓ GPU acceleration available")
                devices = accelerator.get_devices()
                for device in devices:
                    print(f"  Device: {device.name}")
                    print(f"  Vendor: {device.vendor}")
                    print(f"  Compute units: {device.compute_units}")
            else:
                print("✗ GPU not available, using CPU")
        except Exception as e:
            print(f"✗ GPU check failed: {e}")

    return 0


def main() -> int:
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return 1

    commands = {
        "optimize": cmd_optimize,
        "analyze": cmd_analyze,
        "validate": cmd_validate,
        "verify": cmd_verify,
        "generate": cmd_generate,
        "info": cmd_info,
    }

    return commands[args.command](args)


if __name__ == "__main__":
    sys.exit(main())