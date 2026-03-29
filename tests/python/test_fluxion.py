"""
FLUXION Test Suite

Tests for particle system, force fields, annealing, and TPE.
"""

import pytest
import numpy as np
import tempfile
import json
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "python"))

from fluxion.particle_system import (
    FluxionParticle,
    FluxionConnection,
    CriticalPath,
    FluxionParticleSystem,
    CircuitParticles,
    load_circuit_particles,
)
from fluxion.force_fields import (
    WireTensionForce,
    ThermalRepulsionForce,
    TimingGravityForce,
    TopoLossForce,
    CompositeForceField,
)
from fluxion.annealing import (
    ThermodynamicAnnealing,
    TemperatureSchedule,
    ScheduleType,
)
from fluxion.tpe import (
    ThermodynamicPlacementEngine,
    PlacementConfig,
    PlacementResult,
)


class TestFluxionParticle:
    """Tests for FluxionParticle."""

    def test_particle_creation(self):
        """Test basic particle creation."""
        particle = FluxionParticle(
            id=1,
            name="gate_1",
            type="NAND",
            power_pw=5.0,
            area_um2=2.0,
        )
        assert particle.id == 1
        assert particle.name == "gate_1"
        assert particle.type == "NAND"
        assert particle.power_pw == 5.0
        assert particle.area_um2 == 2.0

    def test_particle_mass(self):
        """Test particle mass calculation."""
        particle = FluxionParticle(id=1, name="g1", type="NAND", gate_count=5)
        assert particle.mass == 5.0

    def test_particle_radius(self):
        """Test particle radius calculation."""
        particle = FluxionParticle(id=1, name="g1", type="NAND", area_um2=12.566)
        # Area = pi * r^2 => r = sqrt(Area/pi)
        assert abs(particle.radius - 2.0) < 0.01

    def test_particle_distance(self):
        """Test distance calculation between particles."""
        p1 = FluxionParticle(id=1, name="g1", type="NAND", x=0, y=0)
        p2 = FluxionParticle(id=2, name="g2", type="NAND", x=3, y=4)
        assert abs(p1.distance_to(p2) - 5.0) < 0.01

    def test_particle_to_dict(self):
        """Test particle serialization."""
        particle = FluxionParticle(
            id=1,
            name="g1",
            type="NAND",
            inputs=[0, 2],
            outputs=[3],
        )
        data = particle.to_dict()
        assert data["id"] == 1
        assert data["name"] == "g1"
        assert data["inputs"] == [0, 2]
        assert data["outputs"] == [3]

    def test_particle_from_dict(self):
        """Test particle deserialization."""
        data = {
            "id": 1,
            "name": "g1",
            "type": "NAND",
            "power_pw": 5.0,
            "area_um2": 2.0,
            "x": 10.0,
            "y": 20.0,
            "inputs": [0, 2],
            "outputs": [3],
        }
        particle = FluxionParticle.from_dict(data)
        assert particle.id == 1
        assert particle.name == "g1"
        assert particle.x == 10.0
        assert particle.y == 20.0


class TestFluxionParticleSystem:
    """Tests for FluxionParticleSystem."""

    def test_system_creation(self):
        """Test particle system creation."""
        system = FluxionParticleSystem(die_width=100, die_height=100)
        assert system.die_width == 100
        assert system.die_height == 100
        assert len(system.particles) == 0

    def test_add_particle(self):
        """Test adding particles."""
        system = FluxionParticleSystem()
        particle = FluxionParticle(id=1, name="g1", type="NAND")
        system.add_particle(particle)
        assert len(system.particles) == 1
        assert 1 in system.particles

    def test_add_connection(self):
        """Test adding connections."""
        system = FluxionParticleSystem()
        system.add_particle(FluxionParticle(id=1, name="g1", type="NAND"))
        system.add_particle(FluxionParticle(id=2, name="g2", type="NAND"))
        conn = FluxionConnection(source_id=1, dest_id=2)
        system.add_connection(conn)
        assert len(system.connections) == 1

    def test_get_positions(self):
        """Test getting positions as array."""
        system = FluxionParticleSystem()
        system.add_particle(FluxionParticle(id=1, name="g1", type="NAND", x=0, y=0))
        system.add_particle(FluxionParticle(id=2, name="g2", type="NAND", x=10, y=20))

        positions = system.get_positions()
        assert positions.shape == (2, 2)
        assert positions[0, 0] == 0
        assert positions[1, 0] == 10

    def test_set_positions(self):
        """Test setting positions from array."""
        system = FluxionParticleSystem()
        system.add_particle(FluxionParticle(id=1, name="g1", type="NAND"))
        system.add_particle(FluxionParticle(id=2, name="g2", type="NAND"))

        positions = np.array([[5, 10], [15, 20]])
        system.set_positions(positions)

        assert system.particles[1].x == 5
        assert system.particles[1].y == 10
        assert system.particles[2].x == 15
        assert system.particles[2].y == 20

    def test_randomize_positions(self):
        """Test randomizing positions."""
        system = FluxionParticleSystem(die_width=100, die_height=100)
        system.add_particle(FluxionParticle(id=1, name="g1", type="NAND"))
        system.add_particle(FluxionParticle(id=2, name="g2", type="NAND"))

        system.randomize_positions(seed=42)

        # Check that positions are within bounds
        for particle in system.particles.values():
            assert 0 <= particle.x <= 100
            assert 0 <= particle.y <= 100

    def test_total_wirelength(self):
        """Test wirelength calculation."""
        system = FluxionParticleSystem()
        system.add_particle(FluxionParticle(id=1, name="g1", type="NAND", x=0, y=0))
        system.add_particle(FluxionParticle(id=2, name="g2", type="NAND", x=10, y=0))
        system.add_connection(FluxionConnection(source_id=1, dest_id=2))

        wl = system.total_wirelength()
        assert wl == 10.0  # Manhattan distance


class TestForceFields:
    """Tests for force fields."""

    def create_test_system(self):
        """Create a simple test system."""
        system = FluxionParticleSystem(die_width=100, die_height=100)
        system.add_particle(FluxionParticle(id=1, name="g1", type="NAND", x=0, y=0, power_pw=5))
        system.add_particle(FluxionParticle(id=2, name="g2", type="NAND", x=10, y=0, power_pw=5))
        system.add_particle(FluxionParticle(id=3, name="g3", type="NAND", x=20, y=0, power_pw=5))
        system.add_connection(FluxionConnection(source_id=1, dest_id=2))
        system.add_connection(FluxionConnection(source_id=2, dest_id=3))
        return system

    def test_wire_tension_force(self):
        """Test wire tension force calculation."""
        system = self.create_test_system()
        force = WireTensionForce(weight=1.0, spring_constant=1.0)

        result = force.calculate(system)
        assert result.forces.shape == (3, 2)
        assert result.energy >= 0
        assert result.max_force >= 0

    def test_thermal_repulsion_force(self):
        """Test thermal repulsion force calculation."""
        system = self.create_test_system()
        force = ThermalRepulsionForce(weight=1.0)

        result = force.calculate(system)
        assert result.forces.shape == (3, 2)
        assert result.energy >= 0

    def test_timing_gravity_force(self):
        """Test timing gravity force calculation."""
        system = self.create_test_system()
        force = TimingGravityForce(weight=1.0)

        result = force.calculate(system)
        assert result.forces.shape == (3, 2)

    def test_topoloss_force(self):
        """Test TopoLoss force calculation."""
        system = self.create_test_system()
        force = TopoLossForce(weight=1.0)

        result = force.calculate(system)
        assert result.forces.shape == (3, 2)

    def test_composite_force_field(self):
        """Test composite force field."""
        system = self.create_test_system()
        force = CompositeForceField()

        result = force.calculate(system)
        assert result.forces.shape == (3, 2)
        assert result.energy >= 0

        # Test weight setting
        force.set_weights(wire_tension=2.0, thermal_repulsion=1.0)
        weights = force.get_weights()
        assert weights["wire_tension"] == 2.0


class TestAnnealing:
    """Tests for thermodynamic annealing."""

    def test_temperature_schedule_linear(self):
        """Test linear temperature schedule."""
        schedule = TemperatureSchedule(
            initial_temp=100.0,
            final_temp=0.0,
            schedule_type=ScheduleType.LINEAR,
        )

        temps = schedule.get_schedule(10)
        assert temps[0] == 100.0
        assert temps[-1] == 0.0
        assert len(temps) == 10

    def test_temperature_schedule_exponential(self):
        """Test exponential temperature schedule."""
        schedule = TemperatureSchedule(
            initial_temp=100.0,
            final_temp=0.1,
            schedule_type=ScheduleType.EXPONENTIAL,
            cooling_rate=0.9,
        )

        temps = schedule.get_schedule(10)
        assert temps[0] > temps[-1]  # Temperature decreases
        assert all(temps[i] >= temps[i+1] for i in range(len(temps)-1))

    def test_metropolis_hastings_accept(self):
        """Test Metropolis-Hastings acceptance."""
        annealing = ThermodynamicAnnealing(seed=42)

        # Always accept improvements
        assert annealing.metropolis_hastings_accept(100.0, 90.0, 1.0)

        # Sometimes accept worse states
        accepted = 0
        for _ in range(100):
            if annealing.metropolis_hastings_accept(100.0, 110.0, 10.0):
                accepted += 1
        # Should accept roughly exp(-10/10) = exp(-1) ≈ 37%
        assert 10 < accepted < 60  # Rough check

    def test_anneal_simple(self):
        """Test annealing on simple problem."""
        # Simple quadratic minimization: minimize sum((x-5)^2)
        def energy(pos):
            return np.sum((pos - 5) ** 2)

        annealing = ThermodynamicAnnealing(
            schedule=TemperatureSchedule(
                initial_temp=100.0,
                final_temp=0.01,
                schedule_type=ScheduleType.EXPONENTIAL,
            ),
            seed=42,
        )

        initial = np.array([[0.0, 0.0], [10.0, 10.0]])
        state = annealing.anneal(
            initial_positions=initial,
            energy_function=energy,
            total_steps=1000,
            verbose=False,
        )

        # Should converge near (5, 5)
        assert state.best_energy < 10.0


class TestTPE:
    """Tests for Thermodynamic Placement Engine."""

    def create_test_circuit(self):
        """Create a test circuit."""
        circuit = CircuitParticles(module_name="test", die_width=100, die_height=100)

        for i in range(10):
            circuit.add_particle(FluxionParticle(
                id=i,
                name=f"gate_{i}",
                type="NAND" if i % 2 == 0 else "DFF",
                power_pw=5.0 + i,
                area_um2=2.0 + i * 0.1,
            ))

        for i in range(9):
            circuit.add_connection(FluxionConnection(
                source_id=i,
                dest_id=i + 1,
                is_critical_path=(i < 3),
            ))

        return circuit

    def test_engine_creation(self):
        """Test TPE creation."""
        config = PlacementConfig(die_width=100, die_height=100, use_gpu=False)
        engine = ThermodynamicPlacementEngine(config)
        assert engine.config.die_width == 100

    def test_engine_optimize(self):
        """Test TPE optimization."""
        circuit = self.create_test_circuit()
        config = PlacementConfig(
            die_width=100,
            die_height=100,
            annealing_steps=100,  # Quick test
            use_gpu=False,
            verbose=False,
        )
        engine = ThermodynamicPlacementEngine(config)
        engine.set_circuit(circuit)

        result = engine.optimize()

        assert result is not None
        assert result.positions.shape == (10, 2)
        assert result.total_energy < float('inf')

    def test_result_save_load(self):
        """Test saving and loading results."""
        circuit = self.create_test_circuit()
        config = PlacementConfig(
            die_width=100,
            die_height=100,
            annealing_steps=50,
            use_gpu=False,
            verbose=False,
        )
        engine = ThermodynamicPlacementEngine(config)
        engine.set_circuit(circuit)
        result = engine.optimize()

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            result.save(f.name)
            loaded = PlacementResult.load(f.name)

            assert loaded.total_energy == result.total_energy
            assert loaded.positions.shape == result.positions.shape


class TestCircuitParticles:
    """Tests for CircuitParticles."""

    def test_circuit_creation(self):
        """Test circuit particle creation."""
        circuit = CircuitParticles(module_name="test", die_width=200, die_height=200)
        assert circuit.module_name == "test"
        assert circuit.die_width == 200

    def test_circuit_save_load(self):
        """Test circuit serialization."""
        circuit = CircuitParticles(module_name="test")
        circuit.add_particle(FluxionParticle(id=1, name="g1", type="NAND"))
        circuit.add_particle(FluxionParticle(id=2, name="g2", type="NAND"))
        circuit.add_connection(FluxionConnection(source_id=1, dest_id=2))

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            circuit.save(f.name)
            loaded = CircuitParticles.load(f.name)

            assert loaded.module_name == "test"
            assert len(loaded.particles) == 2
            assert len(loaded.connections) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])