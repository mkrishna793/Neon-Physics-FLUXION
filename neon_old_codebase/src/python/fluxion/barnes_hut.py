"""
FLUXION Barnes-Hut Tree

Implements the Barnes-Hut algorithm for O(N log N) force approximation.
Instead of computing N² pairwise interactions, the algorithm groups
distant particles into clusters and computes approximate forces.

This is the single most important optimization for million-gate scalability.
The thermal repulsion force drops from O(N²) → O(N log N).

Algorithm:
    1. Build a quadtree over all particles
    2. For each particle, walk the tree:
       - If a tree node is "far enough" (θ criterion), treat it as a single mass
       - Otherwise, recurse into children
    3. θ = 0.5 gives ~1% error with 100x speedup at N=1M
"""

import numpy as np
from typing import Optional, List, Tuple
from dataclasses import dataclass, field


@dataclass
class BHNode:
    """A node in the Barnes-Hut quadtree."""
    # Bounding box
    x_min: float
    y_min: float
    x_max: float
    y_max: float

    # Center of mass
    cx: float = 0.0
    cy: float = 0.0
    total_charge: float = 0.0
    count: int = 0

    # Children (NW, NE, SW, SE)
    children: Optional[List[Optional['BHNode']]] = None

    # If leaf, particle index (-1 if internal or empty)
    particle_idx: int = -1

    @property
    def is_leaf(self) -> bool:
        return self.children is None

    @property
    def is_empty(self) -> bool:
        return self.count == 0

    @property
    def size(self) -> float:
        return max(self.x_max - self.x_min, self.y_max - self.y_min)


class BarnesHutTree:
    """
    Barnes-Hut quadtree for fast N-body force calculation.

    Reduces pairwise repulsion from O(N²) to O(N log N) by grouping
    distant particles and treating them as point masses.

    Usage:
        tree = BarnesHutTree(theta=0.5)
        tree.build(positions, charges)
        fx, fy = tree.compute_forces(
            positions, charges,
            force_law=lambda q1, q2, dx, dy, r: k * q1 * q2 / (r * r)
        )

    Args:
        theta: Opening angle parameter (0.0 = exact, 1.0 = very approximate)
               Default 0.5 gives ~1% error with massive speedup.
    """

    def __init__(self, theta: float = 0.5):
        self.theta = theta
        self.root: Optional[BHNode] = None
        self._positions: Optional[np.ndarray] = None
        self._charges: Optional[np.ndarray] = None

    def build(self, positions: np.ndarray, charges: np.ndarray) -> None:
        """
        Build the quadtree from particle positions and charges.

        Args:
            positions: Nx2 array of (x, y) positions
            charges: N array of charge values (e.g., sqrt(power + 1))
        """
        self._positions = positions
        self._charges = charges
        n = positions.shape[0]

        if n == 0:
            self.root = None
            return

        # Compute bounding box with small margin
        margin = 1.0
        x_min = positions[:, 0].min() - margin
        x_max = positions[:, 0].max() + margin
        y_min = positions[:, 1].min() - margin
        y_max = positions[:, 1].max() + margin

        # Make it square for uniform subdivision
        size = max(x_max - x_min, y_max - y_min)
        cx = (x_min + x_max) / 2
        cy = (y_min + y_max) / 2
        x_min = cx - size / 2
        x_max = cx + size / 2
        y_min = cy - size / 2
        y_max = cy + size / 2

        self.root = BHNode(x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max)

        # Insert all particles
        for i in range(n):
            self._insert(self.root, i,
                         positions[i, 0], positions[i, 1], charges[i])

    def _insert(self, node: BHNode, idx: int,
                px: float, py: float, charge: float) -> None:
        """Insert a particle into the tree."""
        if node.is_empty:
            # Empty node: just store the particle
            node.particle_idx = idx
            node.cx = px
            node.cy = py
            node.total_charge = charge
            node.count = 1
            return

        if node.is_leaf:
            # Leaf node with existing particle: subdivide
            if node.size < 1e-8:
                # Extremely small node, co-located particles — aggregate
                old_charge = node.total_charge
                node.total_charge = old_charge + charge
                if node.total_charge > 1e-12:
                    node.cx = (node.cx * old_charge + px * charge) / node.total_charge
                    node.cy = (node.cy * old_charge + py * charge) / node.total_charge
                node.count += 1
                return

            # Save old particle
            old_idx = node.particle_idx
            old_px = self._positions[old_idx, 0]
            old_py = self._positions[old_idx, 1]
            old_charge = self._charges[old_idx]

            # Create children
            node.children = [None, None, None, None]
            node.particle_idx = -1

            # Re-insert old particle
            self._insert_into_child(node, old_idx, old_px, old_py, old_charge)

            # Insert new particle
            self._insert_into_child(node, idx, px, py, charge)

            # Update center of mass
            total = old_charge + charge
            if total > 1e-12:
                node.cx = (old_px * old_charge + px * charge) / total
                node.cy = (old_py * old_charge + py * charge) / total
            node.total_charge = total
            node.count = 2
        else:
            # Internal node: insert into appropriate child
            self._insert_into_child(node, idx, px, py, charge)

            # Update center of mass
            old_total = node.total_charge
            node.total_charge = old_total + charge
            if node.total_charge > 1e-12:
                node.cx = (node.cx * old_total + px * charge) / node.total_charge
                node.cy = (node.cy * old_total + py * charge) / node.total_charge
            node.count += 1

    def _insert_into_child(self, node: BHNode, idx: int,
                            px: float, py: float, charge: float) -> None:
        """Insert particle into the appropriate child quadrant."""
        mid_x = (node.x_min + node.x_max) / 2
        mid_y = (node.y_min + node.y_max) / 2

        if px <= mid_x:
            if py <= mid_y:
                quadrant = 2  # SW
                bounds = (node.x_min, node.y_min, mid_x, mid_y)
            else:
                quadrant = 0  # NW
                bounds = (node.x_min, mid_y, mid_x, node.y_max)
        else:
            if py <= mid_y:
                quadrant = 3  # SE
                bounds = (mid_x, node.y_min, node.x_max, mid_y)
            else:
                quadrant = 1  # NE
                bounds = (mid_x, mid_y, node.x_max, node.y_max)

        if node.children[quadrant] is None:
            node.children[quadrant] = BHNode(
                x_min=bounds[0], y_min=bounds[1],
                x_max=bounds[2], y_max=bounds[3]
            )

        self._insert(node.children[quadrant], idx, px, py, charge)

    def compute_repulsion_forces(
        self,
        positions: np.ndarray,
        charges: np.ndarray,
        repulsion_constant: float = 100.0,
        min_distance: float = 5.0,
    ) -> Tuple[np.ndarray, float]:
        """
        Compute repulsive forces on all particles using Barnes-Hut.

        Uses Coulomb-like repulsion: F = k * q1 * q2 / r²

        Args:
            positions: Nx2 array of positions
            charges: N array of charges
            repulsion_constant: Force constant k
            min_distance: Minimum distance to avoid singularity

        Returns:
            (forces, energy): Nx2 force array and total energy
        """
        n = positions.shape[0]
        forces = np.zeros((n, 2), dtype=np.float64)
        total_energy = 0.0

        if self.root is None or n == 0:
            return forces, total_energy

        for i in range(n):
            fx, fy, energy = self._compute_force_on_particle(
                self.root, i,
                positions[i, 0], positions[i, 1],
                charges[i],
                repulsion_constant, min_distance
            )
            forces[i, 0] = fx
            forces[i, 1] = fy
            total_energy += energy

        # Avoid double counting energy (each pair counted twice)
        total_energy *= 0.5

        return forces, total_energy

    def _compute_force_on_particle(
        self,
        node: BHNode,
        particle_idx: int,
        px: float, py: float, charge: float,
        k: float, min_dist: float,
    ) -> Tuple[float, float, float]:
        """Compute force on a single particle from a tree node."""
        if node is None or node.is_empty:
            return 0.0, 0.0, 0.0

        # If this is a leaf containing only this particle, skip
        if node.is_leaf and node.particle_idx == particle_idx:
            return 0.0, 0.0, 0.0

        dx = node.cx - px
        dy = node.cy - py
        dist = np.sqrt(dx * dx + dy * dy)
        dist = max(dist, min_dist)

        # Barnes-Hut criterion: if node is far enough, use approximation
        if node.is_leaf or (node.size / dist < self.theta):
            # Treat entire node as single point mass
            force_mag = k * charge * node.total_charge / (dist * dist)

            # Repulsion: force away from cluster center
            fx = -force_mag * dx / dist
            fy = -force_mag * dy / dist

            energy = k * charge * node.total_charge / dist

            return fx, fy, energy
        else:
            # Node is too close: recurse into children
            total_fx, total_fy, total_energy = 0.0, 0.0, 0.0

            if node.children is not None:
                for child in node.children:
                    if child is not None and not child.is_empty:
                        fx, fy, energy = self._compute_force_on_particle(
                            child, particle_idx, px, py, charge,
                            k, min_dist
                        )
                        total_fx += fx
                        total_fy += fy
                        total_energy += energy

            return total_fx, total_fy, total_energy

    def get_stats(self) -> dict:
        """Get tree statistics for debugging."""
        if self.root is None:
            return {'nodes': 0, 'depth': 0, 'particles': 0}

        nodes = 0
        max_depth = 0

        def count(node, depth):
            nonlocal nodes, max_depth
            if node is None:
                return
            nodes += 1
            max_depth = max(max_depth, depth)
            if node.children is not None:
                for child in node.children:
                    count(child, depth + 1)

        count(self.root, 0)
        return {
            'nodes': nodes,
            'depth': max_depth,
            'particles': self.root.count if self.root else 0,
            'theta': self.theta,
        }
