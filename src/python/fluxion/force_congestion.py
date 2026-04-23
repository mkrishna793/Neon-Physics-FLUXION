"""
FLUXION Congestion-Aware Force (Force #7)

A predictive routing congestion force that prevents hotspots.
Unlike simple density equalization, this force considers the "routing demand"
of each net based on its length and pin count.

It identifies regions where the available routing tracks would be exceeded
and applies a repulsive pressure to move cells out of those regions.
"""

import numpy as np
from typing import Dict, List, Tuple

from .particle_system import FluxionParticleSystem
from .force_fields import ForceField, ForceResult

class CongestionAwareForce(ForceField):
    """
    Congestion-Aware Force.

    Models routing supply vs demand.
    Supply: Number of tracks available in a GCell (global routing cell).
    Demand: Estimated wire crossings based on HPWL and RUDY (Rectangular Uniform Density).
    """

    def __init__(self, weight: float = 1.0, grid_size: int = 32,
                 max_tracks_per_bin: float = 100.0):
        super().__init__("CongestionAware", weight)
        self.grid_size = grid_size
        self.max_tracks_per_bin = max_tracks_per_bin

    def calculate(self, system: FluxionParticleSystem) -> ForceResult:
        n = len(system.particles)
        forces = np.zeros((n, 2))

        if n == 0:
            return ForceResult(forces=forces, energy=0.0, max_force=0.0)

        # 1. Initialize RUDY map (demand map)
        demand_map = np.zeros((self.grid_size, self.grid_size))
        bin_w = system.die_width / self.grid_size
        bin_h = system.die_height / self.grid_size

        # 2. Populate RUDY map
        # For each net, distribute its routing demand over its bounding box
        for conn in system.connections:
            if conn.source_id not in system.particles or conn.dest_id not in system.particles:
                continue

            p1 = system.particles[conn.source_id]
            p2 = system.particles[conn.dest_id]

            x1, x2 = min(p1.x, p2.x), max(p1.x, p2.x)
            y1, y2 = min(p1.y, p2.y), max(p1.y, p2.y)

            # HPWL-based demand
            width = max(bin_w, x2 - x1)
            height = max(bin_h, y2 - y1)

            # Demand value (normalized)
            demand = (width + height) / (width * height)

            # Map to grid
            c1, c2 = int(x1 / bin_w), int(x2 / bin_w)
            r1, r2 = int(y1 / bin_h), int(y2 / bin_h)

            c1, c2 = np.clip([c1, c2], 0, self.grid_size - 1)
            r1, r2 = np.clip([r1, r2], 0, self.grid_size - 1)

            demand_map[r1:r2+1, c1:c2+1] += demand

        # 3. Compute Congestion Gradient
        # We only care about areas where demand > supply
        congestion = np.maximum(0, demand_map - self.max_tracks_per_bin)

        grad_y, grad_x = np.gradient(congestion, bin_h, bin_w)

        # 4. Apply forces
        total_energy = 0.0
        max_f = 0.0

        for i, p in enumerate(system.particles.values()):
            col = int(np.clip(p.x / bin_w, 0, self.grid_size - 1))
            row = int(np.clip(p.y / bin_h, 0, self.grid_size - 1))

            if congestion[row, col] > 0:
                # Force is negative gradient (away from congestion)
                fx = -grad_x[row, col] * p.area_um2 * 10.0
                fy = -grad_y[row, col] * p.area_um2 * 10.0

                forces[i, 0] = fx
                forces[i, 1] = fy

                mag = np.sqrt(fx*fx + fy*fy)
                max_f = max(max_f, mag)

                total_energy += 0.5 * congestion[row, col]**2

        return ForceResult(
            forces=forces * self.weight,
            energy=total_energy * self.weight,
            max_force=max_f * self.weight,
            force_details={'max_congestion': np.max(demand_map)}
        )

    def calculate_energy(self, system: FluxionParticleSystem) -> float:
        res = self.calculate(system)
        return res.energy
