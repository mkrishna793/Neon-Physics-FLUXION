"""
FLUXION Placement Grid

Physical manufacturing grid for standard cells.
Handles row alignment, site alignment, and tracks placement legality.
"""

import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass

from .lef_library import LEFLibrary
from .particle_system import CircuitParticles, FluxionParticle


@dataclass
class PlacementBoundingBox:
    x_min: float
    y_min: float
    x_max: float
    y_max: float
    particles: List[int] = None


class PlacementGrid:
    """
    Manufacturing placement grid. Tracking free/occupied sites.
    """

    def __init__(self, die_width: float, die_height: float, lef_lib: LEFLibrary):
        self.width = die_width
        self.height = die_height
        self.row_height = lef_lib.row_height
        self.site_width = lef_lib.site_width
        self.lef_lib = lef_lib

        self.num_rows = int(np.floor(self.height / self.row_height))
        self.num_cols = int(np.floor(self.width / self.site_width))

        # Core logic: track which sites are occupied
        # For simplicity in Python prototyping, we use a 2D boolean array.
        # In a C++ prod engine, this is typically interval trees per row.
        self.sites = np.zeros((self.num_rows, self.num_cols), dtype=bool)

        # Reverse lookup: row/col to particle ID
        self.occupants = np.full((self.num_rows, self.num_cols), -1, dtype=np.int32)
        
        # Track inserted particles for quick lookup
        self.placed_particles = {}

    def is_legal(self, x: float, y: float, w: float, h: float) -> bool:
        """Check if placing a box here is legal (aligned + unoccupied)."""
        # Alignment check
        if abs(x % self.site_width) > 1e-6 or abs(y % self.row_height) > 1e-6:
            return False

        # Bounds check
        if x < 0 or y < 0 or x + w > self.width or y + h > self.height:
            return False

        row_start = int(round(y / self.row_height))
        col_start = int(round(x / self.site_width))
        row_span = max(1, int(round(h / self.row_height)))
        col_span = max(1, int(round(w / self.site_width)))

        # Check occupancy
        if np.any(self.sites[row_start:row_start+row_span, col_start:col_start+col_span]):
            return False

        return True

    def place_cell(self, particle_id: int, x: float, y: float, w: float, h: float) -> bool:
        """Attempt to place a cell, mark sites as occupied."""
        row_start = int(round(y / self.row_height))
        col_start = int(round(x / self.site_width))
        row_span = max(1, int(round(h / self.row_height)))
        col_span = max(1, int(round(w / self.site_width)))

        if row_start < 0 or col_start < 0 or \
           row_start+row_span > self.num_rows or col_start+col_span > self.num_cols:
            return False

        if np.any(self.sites[row_start:row_start+row_span, col_start:col_start+col_span]):
            return False

        self.sites[row_start:row_start+row_span, col_start:col_start+col_span] = True
        self.occupants[row_start:row_start+row_span, col_start:col_start+col_span] = particle_id
        
        self.placed_particles[particle_id] = (x, y, w, h)
        return True

    def remove_cell(self, particle_id: int) -> None:
        """Free up sites used by a cell."""
        if particle_id in self.placed_particles:
            x, y, w, h = self.placed_particles.pop(particle_id)
            row_start = int(round(y / self.row_height))
            col_start = int(round(x / self.site_width))
            row_span = max(1, int(round(h / self.row_height)))
            col_span = max(1, int(round(w / self.site_width)))
            
            self.sites[row_start:row_start+row_span, col_start:col_start+col_span] = False
            self.occupants[row_start:row_start+row_span, col_start:col_start+col_span] = -1

    def find_nearest_free(self, x: float, y: float, w: float, h: float, search_radius: int = 50) -> Tuple[float, float]:
        """
        Diamond pattern search for nearest legal placement site.
        Very basic approach; a real tool uses dynamic programming or min-cost flow.
        """
        row_start = int(round(y / self.row_height))
        col_start = int(round(x / self.site_width))
        
        row_span = max(1, int(round(h / self.row_height)))
        col_span = max(1, int(round(w / self.site_width)))

        best_cost = float('inf')
        best_r, best_c = -1, -1

        for r in range(max(0, row_start - search_radius), min(self.num_rows - row_span + 1, row_start + search_radius)):
            for c in range(max(0, col_start - search_radius), min(self.num_cols - col_span + 1, col_start + search_radius)):
                if not np.any(self.sites[r:r+row_span, c:c+col_span]):
                    # Manhattan distance cost
                    cost = abs(r - row_start) * self.row_height + abs(c - col_start) * self.site_width
                    if cost < best_cost:
                        best_cost = cost
                        best_r, best_c = r, c

        if best_r != -1:
            return float(best_c * self.site_width), float(best_r * self.row_height)
        
        return -1.0, -1.0
