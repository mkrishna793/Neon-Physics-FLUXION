"""
FLUXION Density Equalization Force (Force #5)

A grid-based force that pushes particles out of overpopulated regions.
This distributes gates evenly across the die, preventing unroutable
congestion and thermal hotspots.

It works by dividing the die into NxN bins, computing the density of each bin,
and applying a force proportional to the density gradient.
"""

import numpy as np
from typing import Dict, List, Tuple

from .particle_system import FluxionParticleSystem
from .force_fields import ForceField, ForceResult
from .spatial_hash import SpatialHashGrid


class DensityEqualizationForce(ForceField):
    """
    Density Equalization Force - Pushes gates to achieve uniform density.

    This force counters Wire Tension and Timing Gravity which tend to cluster
    gates together. By dividing the die into a grid and computing the continuous
    density gradient, it smoothly pushes gates from high-density bins to
    low-density bins.

    For a million gates, pairwise thermal repulsion is too slow, even with
    Barnes-Hut. Grid-based density equalization provides O(N) global spreading.
    """

    def __init__(self, weight: float = 1.0, grid_size: int = 50,
                 target_density_ratio: float = 0.8):
        """
        Initialize density equalization force.

        Args:
            weight: Overall weight factor
            grid_size: Number of bins per dimension (grid_size x grid_size)
            target_density_ratio: Desired maximum utilization (e.g., 0.8 = 80%)
        """
        super().__init__("DensityEqualization", weight)
        self.grid_size = grid_size
        self.target_density_ratio = target_density_ratio
        self.spatial_hash = None

    def calculate(self, system: FluxionParticleSystem) -> ForceResult:
        """Calculate density equalization forces on all particles."""
        n = len(system.particles)
        forces = np.zeros((n, 2))
        total_energy = 0.0
        max_force = 0.0

        if n == 0:
            return ForceResult(forces=forces, energy=0.0, max_force=0.0)

        particles = list(system.particles.values())
        positions = np.array([[p.x, p.y] for p in particles])

        # 1. Build spatial grid
        bin_width = system.die_width / self.grid_size
        bin_height = system.die_height / self.grid_size

        if self.spatial_hash is None or \
           self.spatial_hash.cols != self.grid_size or \
           self.spatial_hash.rows != self.grid_size:
            self.spatial_hash = SpatialHashGrid(
                system.die_width, system.die_height,
                cell_size=max(bin_width, bin_height),
                auto_cell_size=False
            )

        self.spatial_hash.build(positions)

        # 2. Compute bin densities (area-based, not just count-based)
        # We need the physical area each particle occupies
        bin_areas = np.zeros((self.grid_size, self.grid_size))
        for i, p in enumerate(particles):
            row, col = self.spatial_hash._cell_coords(p.x, p.y)
            # Add particle's physical area to the bin it belongs to
            # (In a more advanced version, overlap physics would distribute area across bin boundaries)
            if 0 <= row < self.grid_size and 0 <= col < self.grid_size:
                bin_areas[row, col] += p.area_um2

        # Physical area of a single bin
        bin_physical_area = bin_width * bin_height
        target_area_per_bin = bin_physical_area * self.target_density_ratio

        # 3. Compute density gradients (differences between adjacent bins)
        # Positive gradient means density increases in that direction
        grad_x = np.zeros_like(bin_areas)
        grad_y = np.zeros_like(bin_areas)

        # Central difference for interior bins
        grad_x[:, 1:-1] = (bin_areas[:, 2:] - bin_areas[:, :-2]) / (2 * bin_width)
        grad_y[1:-1, :] = (bin_areas[2:, :] - bin_areas[:-2, :]) / (2 * bin_height)

        # Forward/backward difference for edges
        grad_x[:, 0] = (bin_areas[:, 1] - bin_areas[:, 0]) / bin_width
        grad_x[:, -1] = (bin_areas[:, -1] - bin_areas[:, -2]) / bin_width
        grad_y[0, :] = (bin_areas[1, :] - bin_areas[0, :]) / bin_height
        grad_y[-1, :] = (bin_areas[-1, :] - bin_areas[-2, :]) / bin_height

        # 4. Apply forces pushing particles down the density gradient
        overflow_factor = 2.0  # Multiplier for bins that exceed target density

        for i, p in enumerate(particles):
            row, col = self.spatial_hash._cell_coords(p.x, p.y)

            if 0 <= row < self.grid_size and 0 <= col < self.grid_size:
                current_area = bin_areas[row, col]

                # Only apply significant force if bin is crowded
                if current_area > target_area_per_bin * 0.5:
                    # Force is proportional to negative gradient (push away from higher density)
                    # and proportional to particle's own area (bigger particles feel more force)
                    penalty = 1.0
                    if current_area > target_area_per_bin:
                        penalty = overflow_factor * (current_area / target_area_per_bin)

                    fx = -grad_x[row, col] * p.area_um2 * penalty
                    fy = -grad_y[row, col] * p.area_um2 * penalty

                    forces[i, 0] = fx
                    forces[i, 1] = fy

                    force_mag = np.sqrt(fx*fx + fy*fy)
                    max_force = max(max_force, force_mag)

                    # Pseudo-energy: penalty for being in an overdense region
                    if current_area > target_area_per_bin:
                        overage = current_area - target_area_per_bin
                        total_energy += 0.5 * p.area_um2 * overage / bin_physical_area

        # Add global centering force to prevent the entire design from drifting off-center
        # when density pushes it outwards
        center_x, center_y = system.die_width / 2, system.die_height / 2
        for i, p in enumerate(particles):
            dx = center_x - p.x
            dy = center_y - p.y
            dist = np.sqrt(dx*dx + dy*dy)
            # Weak force pulling back to center, proportional to distance squared
            if dist > min(system.die_width, system.die_height) * 0.4:
                pull = 0.001 * dist
                forces[i, 0] += pull * dx / dist
                forces[i, 1] += pull * dy / dist

        return ForceResult(
            forces=forces * self.weight,
            energy=total_energy * self.weight,
            max_force=max_force * self.weight,
            force_details={
                'max_bin_utilization': np.max(bin_areas) / bin_physical_area,
                'avg_bin_utilization': np.mean(bin_areas[bin_areas > 0]) / bin_physical_area if np.any(bin_areas > 0) else 0.0
            }
        )

    def calculate_energy(self, system: FluxionParticleSystem) -> float:
        """Calculate total density penalty energy."""
        # This requires a full recalculation of bin areas
        bin_width = system.die_width / self.grid_size
        bin_height = system.die_height / self.grid_size
        bin_physical_area = bin_width * bin_height
        target_area_per_bin = bin_physical_area * self.target_density_ratio

        bin_areas = np.zeros((self.grid_size, self.grid_size))
        for p in system.particles.values():
            col = int(np.clip(p.x / bin_width, 0, self.grid_size - 1))
            row = int(np.clip(p.y / bin_height, 0, self.grid_size - 1))
            bin_areas[row, col] += p.area_um2

        total_energy = 0.0
        for row in range(self.grid_size):
            for col in range(self.grid_size):
                if bin_areas[row, col] > target_area_per_bin:
                    overage = bin_areas[row, col] - target_area_per_bin
                    total_energy += 0.5 * overage * overage / bin_physical_area

        return total_energy * self.weight
