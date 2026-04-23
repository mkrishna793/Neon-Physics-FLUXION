"""
FLUXION Thermodynamic Placement Engine (TPE)

The main engine that orchestrates all components:
1. Loads circuit from Verilator export
2. Creates particle system
3. Applies four force fields
4. Runs thermodynamic annealing
5. Verifies and outputs optimized placement

This is the entry point for FLUXION optimization.
"""

import numpy as np
import json
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Callable
from pathlib import Path

from .particle_system import (
    FluxionParticleSystem,
    CircuitParticles,
    FluxionParticle,
    FluxionConnection,
    load_circuit_particles,
)
from .force_fields import (
    ForceField,
    CompositeForceField,
    WireTensionForce,
    ThermalRepulsionForce,
    TimingGravityForce,
    TopoLossForce,
    ForceResult,
)
from .legalizer import HybridLegalizer
from .def_exporter import export_def
from .annealing import (
    ThermodynamicAnnealing,
    TemperatureSchedule,
    AnnealingState,
    DiscoveryBasin,
    ScheduleType,
)
from .gpu_accelerator import GPUAccelerator, OpenCLAccelerator, create_accelerator


@dataclass
class PlacementConfig:
    """
    Configuration for the placement engine.
    """
    # Die dimensions (micrometers)
    die_width: float = 1000.0
    die_height: float = 1000.0

    # Target timing (picoseconds)
    target_clock_period_ps: float = 1000.0

    # Pipeline stages
    legalize: bool = False
    output_def: bool = False
    tech_node: str = "7nm"  # 2nm, 3nm, 7nm, 14nm, 28nm
    z3_timeout_s: int = 60

    # Force field weights
    wire_tension_weight: float = 1.0
    thermal_repulsion_weight: float = 0.5
    timing_gravity_weight: float = 0.8
    topoloss_weight: float = 0.3
    density_equalization_weight: float = 0.0
    electrostatic_smoothing_weight: float = 0.0
    congestion_aware_weight: float = 0.0

    # Annealing parameters
    initial_temperature: float = 1000.0
    final_temperature: float = 0.01
    annealing_steps: int = 10000
    cooling_rate: float = 0.95
    steps_per_temp: int = 100

    # GPU acceleration
    use_gpu: bool = True
    gpu_device_index: int = 0

    # Optimization
    random_seed: Optional[int] = 42
    verbose: bool = True

    # Output
    output_dir: str = "./output"
    save_history: bool = True
    save_intermediate: bool = False

    # Discovery mode (stochastic exploration)
    discovery_mode: bool = False
    discovery_cycles: int = 3
    discovery_basins: int = 5
    levy_probability: float = 0.15
    reheat_fraction: float = 0.6

    # Adaptive weight scheduling
    adaptive_weights: bool = False


@dataclass
class PlacementResult:
    """
    Result of placement optimization.
    """
    # Optimized positions
    positions: np.ndarray  # Nx2 array

    # Final energy
    total_energy: float
    wire_energy: float
    thermal_energy: float
    timing_energy: float
    topoloss_energy: float

    # Statistics
    total_wirelength: float
    max_temperature: float
    critical_path_delay: float
    max_logic_level: int

    # Annealing statistics
    annealing_time: float
    total_steps: int
    final_temperature: float
    acceptance_rate: float

    # Energy history
    energy_history: List[float] = field(default_factory=list)
    temperature_history: List[float] = field(default_factory=list)

    # Optional pipeline results
    legalizer_stats: dict = None
    def_file_path: str = None

    # Particle system
    circuit: Optional[CircuitParticles] = None

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            'total_energy': self.total_energy,
            'wire_energy': self.wire_energy,
            'thermal_energy': self.thermal_energy,
            'timing_energy': self.timing_energy,
            'topoloss_energy': self.topoloss_energy,
            'total_wirelength': self.total_wirelength,
            'max_temperature': self.max_temperature,
            'critical_path_delay': self.critical_path_delay,
            'max_logic_level': self.max_logic_level,
            'annealing_time': self.annealing_time,
            'total_steps': self.total_steps,
            'final_temperature': self.final_temperature,
            'acceptance_rate': self.acceptance_rate,
            'legalizer_stats': self.legalizer_stats,
            'def_file_path': self.def_file_path,
        }

    def save(self, filepath: str) -> None:
        """Save result to JSON file."""
        result_dict = self.to_dict()

        # Add positions
        result_dict['positions'] = self.positions.tolist()

        # Add energy history
        result_dict['energy_history'] = self.energy_history
        result_dict['temperature_history'] = self.temperature_history

        with open(filepath, 'w') as f:
            json.dump(result_dict, f, indent=2)

    @classmethod
    def load(cls, filepath: str) -> 'PlacementResult':
        """Load result from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)

        return cls(
            positions=np.array(data['positions']),
            total_energy=data['total_energy'],
            wire_energy=data.get('wire_energy', 0),
            thermal_energy=data.get('thermal_energy', 0),
            timing_energy=data.get('timing_energy', 0),
            topoloss_energy=data.get('topoloss_energy', 0),
            total_wirelength=data['total_wirelength'],
            max_temperature=data['max_temperature'],
            critical_path_delay=data['critical_path_delay'],
            max_logic_level=data['max_logic_level'],
            annealing_time=data['annealing_time'],
            total_steps=data['total_steps'],
            final_temperature=data['final_temperature'],
            acceptance_rate=data['acceptance_rate'],
            energy_history=data.get('energy_history', []),
            temperature_history=data.get('temperature_history', []),
            legalizer_stats=data.get('legalizer_stats'),
            def_file_path=data.get('def_file_path'),
        )


class ThermodynamicPlacementEngine:
    """
    FLUXION Thermodynamic Placement Engine.

    Main engine for physics-based chip placement optimization.
    """

    def __init__(self, config: PlacementConfig = None):
        """
        Initialize the placement engine.

        Args:
            config: Placement configuration
        """
        self.config = config or PlacementConfig()

        # Initialize components
        self.circuit: Optional[CircuitParticles] = None
        self.force_field = CompositeForceField()
        self.annealing = ThermodynamicAnnealing(
            schedule=TemperatureSchedule(
                initial_temp=self.config.initial_temperature,
                final_temp=self.config.final_temperature,
                cooling_rate=self.config.cooling_rate,
                steps_per_temp=self.config.steps_per_temp,
            ),
            seed=self.config.random_seed,
        )
        self.gpu_accelerator: Optional[GPUAccelerator] = None

        # Set force field weights
        self.force_field.set_weights(
            wire_tension=self.config.wire_tension_weight,
            thermal_repulsion=self.config.thermal_repulsion_weight,
            timing_gravity=self.config.timing_gravity_weight,
            topoloss=self.config.topoloss_weight,
            density=self.config.density_equalization_weight,
            electrostatic=self.config.electrostatic_smoothing_weight,
            congestion=self.config.congestion_aware_weight,
        )

        # Initialize GPU if available
        if self.config.use_gpu:
            try:
                self.gpu_accelerator = create_accelerator(
                    device_index=self.config.gpu_device_index
                )
                if self.config.verbose:
                    if self.gpu_accelerator.is_available():
                        print("GPU acceleration enabled")
                    else:
                        print("GPU not available, using CPU")
            except Exception as e:
                if self.config.verbose:
                    print(f"GPU initialization failed: {e}, using CPU")

        # Results
        self.result: Optional[PlacementResult] = None

    def load_circuit(self, filepath: str) -> None:
        """
        Load circuit from FLUXION JSON file.

        Args:
            filepath: Path to circuit_particles.json
        """
        if self.config.verbose:
            print(f"Loading circuit from {filepath}")

        self.circuit = load_circuit_particles(filepath)

        # Update configuration from circuit
        self.config.die_width = self.circuit.die_width
        self.config.die_height = self.circuit.die_height
        self.config.target_clock_period_ps = self.circuit.target_clock_period_ps

        if self.config.verbose:
            print(f"Loaded {len(self.circuit.particles)} particles")
            print(f"  Connections: {len(self.circuit.connections)}")
            print(f"  Critical paths: {len(self.circuit.critical_paths)}")
            print(f"  Die size: {self.config.die_width} x {self.config.die_height} um")

    def set_circuit(self, circuit: CircuitParticles) -> None:
        """
        Set circuit directly.

        Args:
            circuit: Circuit particles object
        """
        self.circuit = circuit
        self.config.die_width = circuit.die_width
        self.config.die_height = circuit.die_height
        self.config.target_clock_period_ps = circuit.target_clock_period_ps

    def initialize_positions(self, seed: int = None) -> np.ndarray:
        """
        Initialize random positions for all particles.

        Args:
            seed: Random seed for reproducibility

        Returns:
            Nx2 array of positions
        """
        if seed is not None:
            np.random.seed(seed)

        n = len(self.circuit.particles)
        positions = np.zeros((n, 2))

        for i, particle in enumerate(self.circuit.particles.values()):
            # Random position within die
            positions[i, 0] = np.random.uniform(0, self.config.die_width)
            positions[i, 1] = np.random.uniform(0, self.config.die_height)

        return positions

    def compute_energy(self, positions: np.ndarray) -> float:
        """
        Compute total energy for given positions.

        Args:
            positions: Nx2 array of particle positions

        Returns:
            Total energy
        """
        # Update positions in circuit
        for i, particle in enumerate(self.circuit.particles.values()):
            particle.x = positions[i, 0]
            particle.y = positions[i, 1]

        # Calculate energy from all force fields
        result = self.force_field.calculate(self.circuit)
        return result.energy

    def compute_forces(self, positions: np.ndarray) -> np.ndarray:
        """
        Compute forces on all particles.

        Args:
            positions: Nx2 array of particle positions

        Returns:
            Nx2 array of forces
        """
        # Update positions
        for i, particle in enumerate(self.circuit.particles.values()):
            particle.x = positions[i, 0]
            particle.y = positions[i, 1]

        # Calculate forces
        result = self.force_field.calculate(self.circuit)
        return result.forces

    def optimize(self,
                 initial_positions: np.ndarray = None,
                 callback: Callable[[int, float, float, np.ndarray], None] = None) -> PlacementResult:
        """
        Run placement optimization.

        Args:
            initial_positions: Starting positions (default: random)
            callback: Optional callback(step, temp, energy, positions)

        Returns:
            PlacementResult with optimized positions
        """
        if self.circuit is None:
            raise ValueError("No circuit loaded. Call load_circuit() first.")

        if self.config.verbose:
            print("=" * 60)
            print("FLUXION Thermodynamic Placement Engine")
            print("=" * 60)
            print(f"Particles: {len(self.circuit.particles)}")
            print(f"Connections: {len(self.circuit.connections)}")
            print(f"Die size: {self.config.die_width} x {self.config.die_height} um")
            print(f"Target clock: {self.config.target_clock_period_ps} ps")
            print("=" * 60)

        # Initialize positions
        if initial_positions is None:
            initial_positions = self.initialize_positions(self.config.random_seed)

        # Create energy function for annealing
        def energy_function(pos: np.ndarray) -> float:
            return self.compute_energy(pos)

        # Create annealing callback with optional adaptive weights
        annealing_callback = callback
        if self.config.adaptive_weights:
            initial_weights = self.force_field.get_weights()
            total_steps = self.config.annealing_steps
            def adaptive_callback(step, temp, energy, positions):
                self.force_field.auto_adjust_weights(step, total_steps)
                if callback:
                    callback(step, temp, energy, positions)
            annealing_callback = adaptive_callback

        # Run annealing (standard or discovery)
        start_time = time.time()

        bounds = (0, 0, self.config.die_width, self.config.die_height)

        self._discovery_basins = None

        if self.config.discovery_mode:
            state, basins = self.annealing.discovery_anneal(
                initial_positions=initial_positions,
                energy_function=energy_function,
                total_steps=self.config.annealing_steps,
                num_cycles=self.config.discovery_cycles,
                reheat_fraction=self.config.reheat_fraction,
                levy_probability=self.config.levy_probability,
                num_basins=self.config.discovery_basins,
                bounds=bounds,
                callback=annealing_callback,
                verbose=self.config.verbose,
            )
            self._discovery_basins = basins
        else:
            state = self.annealing.anneal(
                initial_positions=initial_positions,
                energy_function=energy_function,
                total_steps=self.config.annealing_steps,
                bounds=bounds,
                callback=annealing_callback,
                verbose=self.config.verbose,
            )

        end_time = time.time()

        # Extract best positions
        best_positions = state.best_positions.copy()

        # Phase 2: Post-processing pipeline (Legalization & DEF Export)
        legalizer_stats = None
        def_file_path = None
        
        if self.config.legalize:
            if self.config.verbose:
                print("\n[Phase 2] Running Hybrid Legalizer (Tetris + Z3)...")
            
            # Update circuit with best positions so far
            self.circuit.set_positions(best_positions)
            
            legalizer = HybridLegalizer(node=self.config.tech_node, timeout_s=self.config.z3_timeout_s)
            legalizer_stats = legalizer.run(self.circuit)
            
            # The circuit positions have been updated in place
            # Update the result positions to the legal ones
            best_positions = self.circuit.get_positions()
            
            if self.config.verbose:
                print(f"  ✓ Legalization finished in {legalizer_stats['time_s']:.2f}s")
                print(f"  ✓ Tetris success: {legalizer_stats['tetris_success']} cells")
                print(f"  ✓ Z3 resolved:    {legalizer_stats['z3_resolved']} cells")
                print(f"  ⚠ Failed/Illegal: {legalizer_stats['failed_illegal']} cells")
            
        if self.config.output_def:
            def_path = "output.def"
            if self.config.verbose:
                print(f"\n[Phase 3] Exporting to DEF ({def_path})...")
            
            # Make sure circuit has the final positions
            self.circuit.set_positions(best_positions)
            export_def(self.circuit, def_path)
            def_file_path = def_path
            
            if self.config.verbose:
                print(f"  ✓ Export complete.")

        # Update circuit with best positions
        for i, particle in enumerate(self.circuit.particles.values()):
            particle.x = best_positions[i, 0]
            particle.y = best_positions[i, 1]

        # Compute final energies
        self.circuit.set_positions(best_positions)

        # Update physical timing based on final placement
        if hasattr(self.circuit, 'update_timing'):
            self.circuit.update_timing()

        wire_result = self.force_field.wire_tension.calculate(self.circuit)
        thermal_result = self.force_field.thermal_repulsion.calculate(self.circuit)
        timing_result = self.force_field.timing_gravity.calculate(self.circuit)
        topoloss_result = self.force_field.topoloss.calculate(self.circuit)

        # Create result
        self.result = PlacementResult(
            positions=best_positions,
            total_energy=state.best_energy,
            wire_energy=wire_result.energy,
            thermal_energy=thermal_result.energy,
            timing_energy=timing_result.energy,
            topoloss_energy=topoloss_result.energy,
            total_wirelength=self.circuit.total_wirelength(),
            max_temperature=self.circuit.max_temperature(),
            critical_path_delay=self.circuit.critical_path_delay(),
            max_logic_level=max(p.level for p in self.circuit.particles.values()) if self.circuit.particles else 0,
            annealing_time=end_time - start_time,
            total_steps=self.config.annealing_steps,
            final_temperature=state.temperature_history[-1] if state.temperature_history else 0,
            acceptance_rate=state.acceptance_rate,
            energy_history=state.energy_history,
            temperature_history=state.temperature_history,
            circuit=self.circuit,
        )

        if self.config.verbose:
            print("\n" + "=" * 60)
            print("Optimization Complete")
            print("=" * 60)
            print(f"Final energy: {self.result.total_energy:.2f}")
            print(f"  Wire tension: {self.result.wire_energy:.2f}")
            print(f"  Thermal repulsion: {self.result.thermal_energy:.2f}")
            print(f"  Timing gravity: {self.result.timing_energy:.2f}")
            print(f"  TopoLoss: {self.result.topoloss_energy:.2f}")
            print(f"Total wirelength: {self.result.total_wirelength:.2f} um")
            print(f"Max temperature: {self.result.max_temperature:.2f} K")
            print(f"Critical path delay: {self.result.critical_path_delay:.2f} ps")
            if self.result.critical_path_delay > self.config.target_clock_period_ps:
                violation = self.result.critical_path_delay - self.config.target_clock_period_ps
                print(f"      ⚠ TIMING VIOLATION: Critical path exceeds target by {violation:.2f} ps")
            print(f"Annealing time: {self.result.annealing_time:.2f} s")
            print(f"Acceptance rate: {self.result.acceptance_rate:.2%}")
            print("=" * 60)

        return self.result

    def fast_optimize(self,
                      initial_positions: np.ndarray = None,
                      steps: int = 5000) -> PlacementResult:
        """
        Fast optimization for quick iterations.

        Uses reduced annealing steps and simplified forces.

        Args:
            initial_positions: Starting positions
            steps: Number of annealing steps

        Returns:
            PlacementResult
        """
        # Temporarily adjust for fast optimization
        original_steps = self.config.annealing_steps
        original_temp = self.config.initial_temperature
        original_verbose = self.config.verbose

        self.config.annealing_steps = steps
        self.config.initial_temperature = 100.0  # Lower temp for fast convergence
        self.config.verbose = False

        # Update annealing schedule
        self.annealing.schedule = TemperatureSchedule(
            initial_temp=self.config.initial_temperature,
            final_temp=self.config.final_temperature,
            cooling_rate=0.90,
        )

        result = self.optimize(initial_positions)

        # Restore original settings
        self.config.annealing_steps = original_steps
        self.config.initial_temperature = original_temp
        self.config.verbose = original_verbose

        return result

    def save_result(self, filepath: str) -> None:
        """
        Save placement result to file.

        Args:
            filepath: Output file path
        """
        if self.result is None:
            raise ValueError("No result to save. Run optimize() first.")

        self.result.save(filepath)

        if self.config.verbose:
            print(f"Result saved to {filepath}")

    def save_circuit(self, filepath: str) -> None:
        """
        Save optimized circuit to file.

        Args:
            filepath: Output file path
        """
        if self.circuit is None:
            raise ValueError("No circuit to save.")

        self.circuit.save(filepath)

        if self.config.verbose:
            print(f"Circuit saved to {filepath}")

    def get_statistics(self) -> Dict:
        """
        Get placement statistics.

        Returns:
            Dictionary of statistics
        """
        if self.circuit is None:
            return {}

        return {
            'num_particles': len(self.circuit.particles),
            'num_connections': len(self.circuit.connections),
            'num_critical_paths': len(self.circuit.critical_paths),
            'die_size': (self.circuit.die_width, self.circuit.die_height),
            'total_wirelength': self.circuit.total_wirelength(),
            'max_temperature': self.circuit.max_temperature(),
            'critical_path_delay': self.circuit.critical_path_delay(),
        }


def run_tpe(circuit_file: str,
            output_file: str = None,
            config: PlacementConfig = None) -> PlacementResult:
    """
    Convenience function to run TPE on a circuit file.

    Args:
        circuit_file: Path to circuit_particles.json
        output_file: Optional output file for result
        config: Optional placement configuration

    Returns:
        PlacementResult
    """
    config = config or PlacementConfig()
    engine = ThermodynamicPlacementEngine(config)
    engine.load_circuit(circuit_file)
    result = engine.optimize()

    if output_file:
        engine.save_result(output_file)

    return result


# Example usage and testing
if __name__ == "__main__":
    import sys

    # Create a simple test circuit
    print("FLUXION TPE Test Mode")
    print("=" * 60)

    # Create a small test circuit
    circuit = CircuitParticles(module_name="test_circuit", die_width=100.0, die_height=100.0)

    # Add some particles
    for i in range(10):
        particle = FluxionParticle(
            id=i,
            name=f"gate_{i}",
            type="NAND" if i % 2 == 0 else "DFF",
            power_pw=5.0 + i * 0.5,
            area_um2=2.0 + i * 0.2,
            delay_ps=8.0 + i * 0.3,
        )
        circuit.add_particle(particle)

    # Add some connections
    for i in range(9):
        conn = FluxionConnection(
            source_id=i,
            dest_id=i + 1,
            name=f"net_{i}",
            is_critical_path=(i < 3),  # First 3 connections are critical
        )
        circuit.add_connection(conn)

    # Add a critical path
    circuit.add_critical_path(CriticalPath(
        node_ids=list(range(5)),
        total_delay_ps=50.0,
        slack_ps=10.0,
    ))

    # Create and run engine
    config = PlacementConfig(
        die_width=100.0,
        die_height=100.0,
        annealing_steps=1000,
        verbose=True,
        use_gpu=False,  # Use CPU for test
    )

    engine = ThermodynamicPlacementEngine(config)
    engine.set_circuit(circuit)

    print("\nRunning optimization...")
    result = engine.optimize()

    print("\nFinal positions:")
    for i, particle in enumerate(circuit.particles.values()):
        print(f"  {particle.name}: ({particle.x:.2f}, {particle.y:.2f})")

    print("\nDone!")