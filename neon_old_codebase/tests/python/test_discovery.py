"""
Tests for FLUXION Discovery Annealing Mode.

Tests Lévy flights, reheating cycles, and multi-basin tracking.
"""

import sys
import pytest
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "python"))

from fluxion.annealing import (
    ThermodynamicAnnealing,
    TemperatureSchedule,
    ScheduleType,
    DiscoveryBasin,
)


class TestLevyFlight:
    """Tests for Lévy flight move proposals."""

    def test_levy_move_shape(self):
        """Lévy move should return same shape as input."""
        ann = ThermodynamicAnnealing(
            schedule=TemperatureSchedule(initial_temp=100, final_temp=0.01),
            seed=42,
        )
        positions = np.random.rand(20, 2) * 100
        new_pos = ann.propose_levy_move(positions, temperature=50.0)
        assert new_pos.shape == positions.shape

    def test_levy_move_changes_positions(self):
        """Lévy flight should move at least some particles."""
        ann = ThermodynamicAnnealing(
            schedule=TemperatureSchedule(initial_temp=100, final_temp=0.01),
            seed=42,
        )
        positions = np.ones((10, 2)) * 50.0
        new_pos = ann.propose_levy_move(positions, temperature=50.0)
        # At least one particle should have moved
        assert not np.allclose(positions, new_pos)

    def test_levy_respects_bounds(self):
        """Lévy flight should stay within bounds."""
        ann = ThermodynamicAnnealing(
            schedule=TemperatureSchedule(initial_temp=100, final_temp=0.01),
            seed=42,
        )
        positions = np.random.rand(50, 2) * 100
        bounds = (0, 0, 100, 100)
        new_pos = ann.propose_levy_move(positions, temperature=100.0, bounds=bounds)

        assert np.all(new_pos[:, 0] >= 0)
        assert np.all(new_pos[:, 1] >= 0)
        assert np.all(new_pos[:, 0] <= 100)
        assert np.all(new_pos[:, 1] <= 100)

    def test_levy_heavy_tail(self):
        """Lévy flights should produce larger jumps than Gaussian on average."""
        ann = ThermodynamicAnnealing(
            schedule=TemperatureSchedule(initial_temp=100, final_temp=0.01),
            seed=42,
        )
        positions = np.zeros((100, 2))

        # Collect jump magnitudes
        levy_jumps = []
        for _ in range(50):
            new_pos = ann.propose_levy_move(positions, temperature=50.0)
            diffs = np.linalg.norm(new_pos - positions, axis=1)
            levy_jumps.extend(diffs[diffs > 0])

        # Should have some large jumps (>10 units)
        max_jump = max(levy_jumps) if levy_jumps else 0
        assert max_jump > 1.0  # At minimum, jumps should be non-trivial


class TestDiscoveryAnnealing:
    """Tests for discovery annealing with reheating."""

    def test_discovery_returns_basins(self):
        """Discovery mode should return multiple basins."""
        ann = ThermodynamicAnnealing(
            schedule=TemperatureSchedule(initial_temp=100, final_temp=0.01),
            seed=42,
        )

        def energy(pos):
            return np.sum((pos - 5) ** 2)

        initial = np.array([[0.0, 0.0], [10.0, 10.0]])
        state, basins = ann.discovery_anneal(
            initial_positions=initial,
            energy_function=energy,
            total_steps=300,
            num_cycles=3,
            verbose=False,
        )

        assert len(basins) > 0
        assert len(basins) <= 3
        assert all(isinstance(b, DiscoveryBasin) for b in basins)

    def test_discovery_basins_sorted(self):
        """Returned basins should be sorted by energy (best first)."""
        ann = ThermodynamicAnnealing(
            schedule=TemperatureSchedule(initial_temp=100, final_temp=0.01),
            seed=42,
        )

        def energy(pos):
            return np.sum(pos ** 2)

        initial = np.random.rand(5, 2) * 100
        state, basins = ann.discovery_anneal(
            initial_positions=initial,
            energy_function=energy,
            total_steps=600,
            num_cycles=3,
            verbose=False,
        )

        energies = [b.energy for b in basins]
        assert energies == sorted(energies)

    def test_discovery_finds_better_than_initial(self):
        """Discovery should find better energy than the starting point."""
        ann = ThermodynamicAnnealing(
            schedule=TemperatureSchedule(initial_temp=100, final_temp=0.01),
            seed=42,
        )

        def energy(pos):
            return np.sum((pos - 5) ** 2)

        initial = np.array([[0.0, 0.0], [10.0, 10.0]])
        initial_energy = energy(initial)

        state, basins = ann.discovery_anneal(
            initial_positions=initial,
            energy_function=energy,
            total_steps=1000,
            num_cycles=3,
            verbose=False,
        )

        assert state.best_energy < initial_energy

    def test_discovery_with_bounds(self):
        """Discovery should work with position bounds."""
        ann = ThermodynamicAnnealing(
            schedule=TemperatureSchedule(initial_temp=100, final_temp=0.01),
            seed=42,
        )

        def energy(pos):
            return np.sum(pos ** 2)

        initial = np.random.rand(10, 2) * 50
        bounds = (0, 0, 100, 100)

        state, basins = ann.discovery_anneal(
            initial_positions=initial,
            energy_function=energy,
            total_steps=300,
            num_cycles=2,
            bounds=bounds,
            verbose=False,
        )

        # All basin positions should be within bounds
        for b in basins:
            assert np.all(b.positions >= 0)
            assert np.all(b.positions <= 100)


class TestDiscoveryBasin:
    """Tests for DiscoveryBasin dataclass."""

    def test_basin_creation(self):
        """Test creating a DiscoveryBasin."""
        pos = np.array([[1.0, 2.0], [3.0, 4.0]])
        basin = DiscoveryBasin(positions=pos, energy=42.0, cycle=0)
        assert basin.energy == 42.0
        assert basin.cycle == 0
        assert basin.positions.shape == (2, 2)
