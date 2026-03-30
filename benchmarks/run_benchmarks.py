#!/usr/bin/env python3
import time
import sys
import os
import argparse
from pathlib import Path
import numpy as np

# Add the project src dir to pythonpath to import fluxion natively
sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "python"))

from fluxion.particle_system import CircuitParticles, FluxionParticle, FluxionConnection
from fluxion.tpe import ThermodynamicPlacementEngine, PlacementConfig

def generate_benchmark_circuit(num_gates: int) -> CircuitParticles:
    """Generates a synthetic circuit of exactly num_gates size."""
    np.random.seed(42)
    circuit = CircuitParticles(
        module_name=f"bench_{num_gates}",
        die_width=100.0 * np.sqrt(num_gates/10),
        die_height=100.0 * np.sqrt(num_gates/10),
    )

    for i in range(num_gates):
        circuit.add_particle(FluxionParticle(
            id=i,
            name=f"gate_{i}",
            type="NAND" if np.random.random() > 0.5 else "NOR",
            power_pw=np.random.uniform(5, 50),
            area_um2=np.random.uniform(2, 6),
            delay_ps=np.random.uniform(5, 15),
        ))

    # Basic connectivity, ~2 inputs per gate
    for i in range(num_gates):
        if i > 0:
            circuit.add_connection(FluxionConnection(
                source_id=max(0, i - int(np.random.uniform(1, min(i+1, 5)))),
                dest_id=i,
                name=f"net_{i}",
                is_critical_path=(np.random.random() < 0.1)
            ))
            
    return circuit

def run_benchmark(sizes=[10, 50, 100, 500]):
    print("=" * 60)
    print("FLUXION Benchmark Suite (CPU vs GPU)")
    print("=" * 60)
    print(f"{'Size (Gates)':<15} | {'CPU Runtime (s)':<18} | {'GPU Runtime (s)':<18}")
    print("-" * 60)
    
    for size in sizes:
        circuit = generate_benchmark_circuit(size)
        
        # Test CPU
        config_cpu = PlacementConfig(
            die_width=circuit.die_width,
            die_height=circuit.die_height,
            annealing_steps=5000,
            use_gpu=False,
            verbose=False
        )
        engine_cpu = ThermodynamicPlacementEngine(config_cpu)
        engine_cpu.set_circuit(circuit)
        
        t0 = time.time()
        engine_cpu.optimize()
        cpu_time = time.time() - t0
        
        # Test GPU
        config_gpu = PlacementConfig(
            die_width=circuit.die_width,
            die_height=circuit.die_height,
            annealing_steps=5000,
            use_gpu=True,
            verbose=False
        )
        engine_gpu = ThermodynamicPlacementEngine(config_gpu)
        engine_gpu.set_circuit(circuit)
        
        t0 = time.time()
        engine_gpu.optimize()
        gpu_time = time.time() - t0
        
        print(f"{size:<15} | {cpu_time:<18.2f} | {gpu_time:<18.2f}")
        
    print("=" * 60)

if __name__ == "__main__":
    os.makedirs(Path(__file__).parent, exist_ok=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", type=int, nargs="+", default=[10, 50, 100, 250, 500])
    args = parser.parse_args()
    run_benchmark(args.sizes)
