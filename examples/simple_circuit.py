#!/usr/bin/env python3
"""
FLUXION Simple Example

This example demonstrates basic usage of FLUXION for circuit placement.
"""

import sys
from pathlib import Path

# Add parent directory to path for local development
sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "python"))

import numpy as np
from fluxion import (
    ThermodynamicPlacementEngine,
    PlacementConfig,
    CircuitParticles,
    FluxionParticle,
    FluxionConnection,
)


def create_simple_circuit():
    """
    Create a simple test circuit with NAND gates and D flip-flops.

    This represents a small digital circuit with:
    - 20 gates (mix of NAND and DFF)
    - Connections between consecutive gates
    - A few critical timing paths
    """
    print("Creating simple test circuit...")

    circuit = CircuitParticles(
        module_name="simple_test_circuit",
        die_width=500.0,   # 500 micrometers
        die_height=500.0,  # 500 micrometers
    )
    circuit.target_clock_period_ps = 500.0  # 500 ps = 2 GHz

    # Create 20 gates
    for i in range(20):
        # Alternate between combinational and sequential
        if i % 5 == 0:
            # D flip-flop (sequential)
            gate_type = "DFF"
            delay = 30.0 + np.random.uniform(-5, 5)  # ps
            area = 10.0 + np.random.uniform(-2, 2)   # um²
            power = 100.0 + np.random.uniform(-20, 20)  # pW
        else:
            # NAND gate (combinational)
            gate_type = "NAND"
            delay = 8.0 + np.random.uniform(-2, 2)    # ps
            area = 2.0 + np.random.uniform(-0.5, 0.5)  # um²
            power = 5.0 + np.random.uniform(-1, 1)     # pW

        particle = FluxionParticle(
            id=i,
            name=f"gate_{i}",
            type=gate_type,
            power_pw=power,
            area_um2=area,
            delay_ps=delay,
        )
        circuit.add_particle(particle)

    # Create connections (feed-forward design)
    for i in range(19):
        # Each gate connects to the next
        is_critical = (i < 5)  # First 5 connections are critical timing paths
        conn = FluxionConnection(
            source_id=i,
            dest_id=i + 1,
            name=f"net_{i}",
            is_critical_path=is_critical,
        )
        circuit.add_connection(conn)

    # Add some branching connections
    for i in range(0, 15, 3):
        conn = FluxionConnection(
            source_id=i,
            dest_id=min(i + 5, 19),
            name=f"branch_net_{i}",
            is_critical_path=False,
        )
        circuit.add_connection(conn)

    print(f"  Created {len(circuit.particles)} gates")
    print(f"  Created {len(circuit.connections)} connections")

    return circuit


def main():
    """Run the simple example."""
    print("=" * 60)
    print("FLUXION Simple Example")
    print("=" * 60)
    print()

    # Step 1: Create circuit
    circuit = create_simple_circuit()
    print()

    # Step 2: Configure placement engine
    print("Configuring placement engine...")
    config = PlacementConfig(
        die_width=500.0,
        die_height=500.0,
        target_clock_period_ps=500.0,
        annealing_steps=2000,  # Quick run for example
        initial_temperature=100.0,
        final_temperature=0.1,
        cooling_rate=0.95,
        random_seed=42,  # For reproducibility
        use_gpu=False,   # Use CPU for compatibility
        verbose=True,
    )
    print()

    # Step 3: Create and run engine
    print("Running thermodynamic placement optimization...")
    print("-" * 60)

    engine = ThermodynamicPlacementEngine(config)
    engine.set_circuit(circuit)

    # Run optimization
    result = engine.optimize()

    print()
    print("=" * 60)
    print("Optimization Results")
    print("=" * 60)

    # Step 4: Display results
    print(f"\nEnergy Components:")
    print(f"  Wire Tension Energy:    {result.wire_energy:,.2f}")
    print(f"  Thermal Energy:         {result.thermal_energy:,.2f}")
    print(f"  Timing Energy:          {result.timing_energy:,.2f}")
    print(f"  TopoLoss Energy:        {result.topoloss_energy:,.2f}")
    print(f"  ────────────────────────────────────")
    print(f"  Total Energy:           {result.total_energy:,.2f}")

    print(f"\nPhysical Metrics:")
    print(f"  Total Wirelength:       {result.total_wirelength:,.2f} um")
    print(f"  Max Temperature:        {result.max_temperature:,.2f} K")
    print(f"  Critical Path Delay:     {result.critical_path_delay:,.2f} ps")

    print(f"\nOptimization Statistics:")
    print(f"  Annealing Time:         {result.annealing_time:,.2f} seconds")
    print(f"  Total Steps:            {result.total_steps:,}")
    print(f"  Final Temperature:      {result.final_temperature:.4f}")
    print(f"  Acceptance Rate:        {result.acceptance_rate:.2%}")

    # Step 5: Show final positions
    print(f"\nFinal Gate Positions (first 10):")
    print("  Gate          X         Y       Type")
    print("  ──────────────────────────────────────")
    for i, (pid, particle) in enumerate(circuit.particles.items()):
        if i >= 10:
            break
        print(f"  {particle.name:12} {particle.x:8.2f}  {particle.y:8.2f}  {particle.type}")

    # Step 6: Save results
    print("\nSaving results...")
    engine.save_result("simple_example_result.json")
    engine.save_circuit("simple_example_circuit.json")
    print("  Results saved to simple_example_result.json")
    print("  Circuit saved to simple_example_circuit.json")

    print()
    print("=" * 60)
    print("Example complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()