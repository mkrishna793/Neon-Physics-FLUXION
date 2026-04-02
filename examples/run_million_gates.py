#!/usr/bin/env python3
"""
FLUXION Million-Gate Benchmark

Demonstrates the extreme scalability of the V2 engine running Barnes-Hut
and FFT-based electrostatic forces. Generates a large synthetic dataset and
benchmarks the physics pipeline.
"""
import sys
import time
import argparse
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "python"))

from fluxion import (
    ThermodynamicPlacementEngine,
    PlacementConfig,
    CircuitParticles,
    FluxionParticle,
    FluxionConnection
)

def create_massive_synthetic_circuit(num_gates: int):
    print(f"Generating massive synthetic circuit with {num_gates:,} gates...")
    start_time = time.time()
    
    # Very rough area scaling for visual demonstration
    die_size = float(np.sqrt(num_gates) * 10)
    circuit = CircuitParticles(
        module_name="massive_bench",
        die_width=die_size,
        die_height=die_size,
    )
    
    # 1. Generate Gates
    gate_types = ["NAND", "NOR", "DFF"]
    gate_areas = [2.0, 2.0, 8.0]
    
    # Vectorized generation for speed
    types_idx = np.random.choice([0, 1, 2], size=num_gates, p=[0.4, 0.4, 0.2])
    
    for i in range(num_gates):
        gtype = gate_types[types_idx[i]]
        area = gate_areas[types_idx[i]]
        
        # Initial random placement
        px = np.random.uniform(0, die_size)
        py = np.random.uniform(0, die_size)
        
        p = FluxionParticle(
            id=i, name=f"g_{i}", type=gtype, 
            x=px, y=py, power_pw=5.0, area_um2=area
        )
        circuit.add_particle(p)

    # 2. Generate Connections (roughly 1.5x gates to represent nets)
    print("Generating connections...")
    num_nets = int(num_gates * 1.5)
    
    # Fast vectorized connectivity generation (very synthetic)
    sources = np.random.randint(0, num_gates, size=num_nets)
    # Prefer local connections using a clipped normal distribution
    dests_offset = np.random.normal(0, np.sqrt(num_gates)/2, size=num_nets).astype(int)
    dests = np.clip(sources + dests_offset, 0, num_gates - 1)
    
    connections = set()
    for i in range(num_nets):
        if sources[i] != dests[i]:
            # Avoid self-loops
            connections.add((sources[i], dests[i]))
            
    for i, (src, dst) in enumerate(connections):
        conn = FluxionConnection(source_id=src, dest_id=dst, name=f"n_{i}")
        circuit.add_connection(conn)
        
    print(f"Generation took {time.time() - start_time:.2f}s")
    return circuit

def main():
    parser = argparse.ArgumentParser(description="FLUXION Scale Benchmark")
    parser.add_argument("--gates", type=int, default=100_000, help="Number of gates")
    parser.add_argument("--steps", type=int, default=200, help="Annealing steps")
    parser.add_argument("--def-output", action="store_true", help="Export DEF")
    parser.add_argument("--legalize", action="store_true", help="Run Legalizer")
    args = parser.parse_args()
    
    circuit = create_massive_synthetic_circuit(args.gates)
    
    # Force heavy electrostatic smoothing and barns-hut testing
    config = PlacementConfig(
        die_width=circuit.die_width,
        die_height=circuit.die_height,
        annealing_steps=args.steps,
        verbose=True,
        wire_tension_weight=1.0,
        thermal_repulsion_weight=1.0,  # This will auto-switch to Barnes-Hut > 5k
        density_equalization_weight=0.0,
        electrostatic_smoothing_weight=2.0, # Push hard with FFT
        legalize=args.legalize,
        output_def=args.def_output
    )
    
    print("\nStarting pipeline...")
    engine = ThermodynamicPlacementEngine(config)
    engine.set_circuit(circuit)
    
    start_time = time.time()
    result = engine.optimize()
    optimization_time = time.time() - start_time
    
    print("\n" + "=" * 50)
    print("BENCHMARK RESULTS")
    print("=" * 50)
    print(f"Gates:         {args.gates:,}")
    print(f"Algorithm:     Barnes-Hut (O(N log N)) + FFT Poisson (O(N log N))")
    print(f"Steps:         {args.steps}")
    print(f"Total Time:    {optimization_time:.2f} s")
    print(f"Step speed:    {optimization_time / args.steps * 1000:.2f} ms/step")
    print("=" * 50)

if __name__ == "__main__":
    main()
