"""
FLUXION Spatial Hash Grid

Provides O(1) average-case neighbor lookups for local force calculations.
Instead of checking every pair of gates (O(N²)), gates are bucketed into
grid cells and only nearby cells are queried.

This is the backbone of million-gate scalability.
"""

import numpy as np
from typing import List, Tuple, Set, Optional, Dict
from dataclasses import dataclass


@dataclass
class SpatialHashEntry:
    """An entry in the spatial hash grid."""
    index: int      # Particle index in the positions array
    x: float
    y: float


class SpatialHashGrid:
    """
    2D spatial hash grid for fast neighbor queries.

    Divides the placement area into uniform cells. Each cell stores
    indices of particles within it. Neighbor queries only check
    adjacent cells, giving O(1) average lookup time.

    Usage:
        grid = SpatialHashGrid(die_width=1000, die_height=1000, cell_size=50)
        grid.build(positions)  # positions is Nx2 array
        neighbors = grid.query_radius(x, y, radius=100)
    """

    def __init__(self, die_width: float, die_height: float,
                 cell_size: Optional[float] = None,
                 auto_cell_size: bool = True):
        """
        Initialize spatial hash grid.

        Args:
            die_width: Width of the placement area
            die_height: Height of the placement area
            cell_size: Size of each grid cell (None for auto)
            auto_cell_size: Auto-compute cell size from particle density
        """
        self.die_width = die_width
        self.die_height = die_height
        self.auto_cell_size = auto_cell_size

        if cell_size is not None:
            self.cell_size = cell_size
        else:
            # Default: 1/20th of die dimension
            self.cell_size = max(die_width, die_height) / 20.0

        self.cols = max(1, int(np.ceil(die_width / self.cell_size)))
        self.rows = max(1, int(np.ceil(die_height / self.cell_size)))

        # Grid storage: dict mapping (row, col) -> list of particle indices
        self.cells: Dict[Tuple[int, int], List[int]] = {}
        self.positions: Optional[np.ndarray] = None
        self.n_particles = 0

    def _cell_coords(self, x: float, y: float) -> Tuple[int, int]:
        """Convert world coordinates to grid cell coordinates."""
        col = int(np.clip(x / self.cell_size, 0, self.cols - 1))
        row = int(np.clip(y / self.cell_size, 0, self.rows - 1))
        return (row, col)

    def build(self, positions: np.ndarray) -> None:
        """
        Build the spatial hash from particle positions.

        Args:
            positions: Nx2 array of (x, y) positions
        """
        self.positions = positions
        self.n_particles = positions.shape[0]
        self.cells.clear()

        # Auto-adjust cell size based on particle density
        if self.auto_cell_size and self.n_particles > 0:
            area = self.die_width * self.die_height
            avg_spacing = np.sqrt(area / max(self.n_particles, 1))
            # Cell size = 2x average spacing for good bucket distribution
            self.cell_size = max(avg_spacing * 2.0, 1.0)
            self.cols = max(1, int(np.ceil(self.die_width / self.cell_size)))
            self.rows = max(1, int(np.ceil(self.die_height / self.cell_size)))

        # Insert all particles
        for i in range(self.n_particles):
            cell = self._cell_coords(positions[i, 0], positions[i, 1])
            if cell not in self.cells:
                self.cells[cell] = []
            self.cells[cell].append(i)

    def query_radius(self, x: float, y: float, radius: float) -> List[int]:
        """
        Find all particle indices within radius of (x, y).

        Args:
            x: Query x coordinate
            y: Query y coordinate
            radius: Search radius

        Returns:
            List of particle indices within radius
        """
        if self.positions is None:
            return []

        result = []
        radius_sq = radius * radius

        # Determine which cells to check
        min_col = max(0, int((x - radius) / self.cell_size))
        max_col = min(self.cols - 1, int((x + radius) / self.cell_size))
        min_row = max(0, int((y - radius) / self.cell_size))
        max_row = min(self.rows - 1, int((y + radius) / self.cell_size))

        for row in range(min_row, max_row + 1):
            for col in range(min_col, max_col + 1):
                cell = (row, col)
                if cell in self.cells:
                    for idx in self.cells[cell]:
                        dx = self.positions[idx, 0] - x
                        dy = self.positions[idx, 1] - y
                        if dx * dx + dy * dy <= radius_sq:
                            result.append(idx)

        return result

    def query_cell(self, x: float, y: float) -> List[int]:
        """
        Get all particle indices in the same cell as (x, y).

        Args:
            x: Query x coordinate
            y: Query y coordinate

        Returns:
            List of particle indices in the cell
        """
        cell = self._cell_coords(x, y)
        return self.cells.get(cell, [])

    def query_neighbors(self, index: int, radius: float) -> List[int]:
        """
        Find all neighbors of particle at given index within radius.

        Args:
            index: Particle index
            radius: Search radius

        Returns:
            List of neighbor indices (excluding self)
        """
        if self.positions is None or index >= self.n_particles:
            return []

        x, y = self.positions[index]
        neighbors = self.query_radius(x, y, radius)
        return [n for n in neighbors if n != index]

    def get_cell_density(self) -> np.ndarray:
        """
        Get density map as 2D array (particles per cell).

        Returns:
            rows x cols array of particle counts
        """
        density = np.zeros((self.rows, self.cols), dtype=np.int32)
        for (row, col), indices in self.cells.items():
            if 0 <= row < self.rows and 0 <= col < self.cols:
                density[row, col] = len(indices)
        return density

    def get_density_at(self, x: float, y: float) -> int:
        """Get number of particles in the cell containing (x, y)."""
        cell = self._cell_coords(x, y)
        return len(self.cells.get(cell, []))

    def get_overcrowded_cells(self, threshold: float = 2.0) -> List[Tuple[int, int, List[int]]]:
        """
        Find cells with more particles than threshold * average.

        Args:
            threshold: Multiplier over average density

        Returns:
            List of (row, col, particle_indices) for overcrowded cells
        """
        if not self.cells:
            return []

        total = sum(len(v) for v in self.cells.values())
        avg = total / max(len(self.cells), 1)
        cutoff = avg * threshold

        result = []
        for (row, col), indices in self.cells.items():
            if len(indices) > cutoff:
                result.append((row, col, indices))

        return result

    def get_cell_bounds(self, row: int, col: int) -> Tuple[float, float, float, float]:
        """
        Get world-space bounds of a grid cell.

        Returns:
            (x_min, y_min, x_max, y_max)
        """
        x_min = col * self.cell_size
        y_min = row * self.cell_size
        x_max = min(x_min + self.cell_size, self.die_width)
        y_max = min(y_min + self.cell_size, self.die_height)
        return (x_min, y_min, x_max, y_max)
