"""
FLUXION Hybrid Legalizer

Legalization transforms continuous float coordinates into legal, discrete layout coordinates
that do not overlap and adhere to manufacturing rows and sites.

This uses a hybrid approach:
1. Greedy Tetris/Abacus approach for 95% of cells (extremely fast)
2. Z3 SAT formulation for the final 5% (congested hotspots, guaranteed optimal but slow)
"""

import numpy as np
import time
from typing import List, Tuple, Dict
from dataclasses import dataclass

try:
    import z3
    Z3_AVAILABLE = True
except ImportError:
    Z3_AVAILABLE = False

from .particle_system import CircuitParticles
from .lef_library import LEFLibrary
from .grid import PlacementGrid, PlacementBoundingBox


class TetrisLegalizer:
    """Fast greedy legalizer mapping particles to optimal grid sites."""
    
    def __init__(self, grid: PlacementGrid):
        self.grid = grid
        
    def legalize(self, circuit: CircuitParticles) -> Tuple[int, List[int]]:
        """
        Legalize all particles. Returns (success_count, failed_particle_ids).
        """
        # Sort particles roughly by X coordinate to sweep left-to-right
        particles = list(circuit.particles.values())
        particles.sort(key=lambda p: p.x)
        
        failed = []
        success = 0
        
        for p in particles:
            w, h = self.grid.lef_lib.get_macro_dimensions(p.type)
            
            # Snap to nearest intended target
            target_x = round(p.x / self.grid.site_width) * self.grid.site_width
            target_y = round(p.y / self.grid.row_height) * self.grid.row_height
            
            target_x = max(0, min(target_x, self.grid.width - w))
            target_y = max(0, min(target_y, self.grid.height - h))
            
            if self.grid.is_legal(target_x, target_y, w, h):
                self.grid.place_cell(p.id, target_x, target_y, w, h)
                p.x, p.y = target_x, target_y
                success += 1
            else:
                # Need to find nearest free
                # Search incrementally outwards
                found_x, found_y = self.grid.find_nearest_free(target_x, target_y, w, h)
                if found_x >= 0:
                    self.grid.place_cell(p.id, found_x, found_y, w, h)
                    p.x, p.y = found_x, found_y
                    success += 1
                else:
                    failed.append(p.id)
                    
        return success, failed


class Z3HotspotSolver:
    """Solves exact placement for heavily congested regions using SMT."""
    
    def __init__(self, grid: PlacementGrid, timeout_s: int = 10):
        self.grid = grid
        self.timeout_s = timeout_s
        self.z3_ctx = z3.Context()

    def solve_region(self, circuit: CircuitParticles, particle_ids: List[int], 
                     box_x: float, box_y: float, box_w: float, box_h: float) -> bool:
        """
        Given a list of particle IDs that failed greedy placement, and a bounding 
        box they should roughly fit in, generate a precise SAT formula to place them.
        """
        if not Z3_AVAILABLE:
            print("Z3 solver not available. Skipping SAT legalization step.")
            return False
            
        opt = z3.Optimize(ctx=self.z3_ctx)
        opt.set("timeout", self.timeout_s * 1000) 
        
        # Grid dimensions within box (integer grid coordinates)
        col_start = int(round(box_x / self.grid.site_width))
        row_start = int(round(box_y / self.grid.row_height))
        max_cols = int(round(box_w / self.grid.site_width))
        max_rows = int(round(box_h / self.grid.row_height))
        
        # Variables: x, y positions in site units
        positions_x = {}
        positions_y = {}
        widths = {}
        heights = {}
        target_x = {}
        target_y = {}
        
        for pid in particle_ids:
            p = circuit.particles[pid]
            w, h = self.grid.lef_lib.get_macro_dimensions(p.type)
            site_w = max(1, int(round(w / self.grid.site_width)))
            site_h = max(1, int(round(h / self.grid.row_height)))
            
            var_x = z3.Int(f'x_{pid}', ctx=self.z3_ctx)
            var_y = z3.Int(f'y_{pid}', ctx=self.z3_ctx)
            
            positions_x[pid] = var_x
            positions_y[pid] = var_y
            widths[pid] = site_w
            heights[pid] = site_h
            
            # Snap intended target to local grid
            t_col = int(round(p.x / self.grid.site_width)) - col_start
            t_row = int(round(p.y / self.grid.row_height)) - row_start
            target_x[pid] = t_col
            target_y[pid] = t_row
            
            # Constraint 1: Bounding box limits
            opt.add(var_x >= 0, var_y >= 0)
            opt.add(var_x + site_w <= max_cols)
            opt.add(var_y + site_h <= max_rows)
            
            # Objective: minimize displacement from target (Manhattan)
            dx = z3.Int(f'dx_{pid}', ctx=self.z3_ctx)
            dy = z3.Int(f'dy_{pid}', ctx=self.z3_ctx)
            
            # Z3 doesn't have an absolute value function out of the box in Ints, so we use max
            opt.add(dx >= var_x - t_col, dx >= t_col - var_x)
            opt.add(dy >= var_y - t_row, dy >= t_row - var_y)
            
            # Small weight on Y to prefer horizontal sliding along rows
            opt.minimize(dx + 2 * dy)
            
        # Constraint 2: Non-overlapping + account for already occupied sites
        for i, pid1 in enumerate(particle_ids):
            x1, y1 = positions_x[pid1], positions_y[pid1]
            w1, h1 = widths[pid1], heights[pid1]
            
            for j, pid2 in enumerate(particle_ids):
                if i >= j: continue
                
                x2, y2 = positions_x[pid2], positions_y[pid2]
                w2, h2 = widths[pid2], heights[pid2]
                
                # Rectangles 1 and 2 do not overlap if:
                # 1 is left of 2, 1 is right of 2, 1 is below 2, OR 1 is above 2
                opt.add(z3.Or(
                    x1 + w1 <= x2,
                    x2 + w2 <= x1,
                    y1 + h1 <= y2,
                    y2 + h2 <= y1
                ))
                
            # Block out sites already occupied by Tetris
            for r in range(max_rows):
                world_row = row_start + r
                if world_row >= self.grid.num_rows: continue
                
                for c in range(max_cols):
                    world_col = col_start + c
                    if world_col >= self.grid.num_cols: continue
                    
                    if self.grid.sites[world_row, world_col]:
                        # If site is occupied, this cell cannot overlap it
                        opt.add(z3.Or(
                            x1 + w1 <= c,
                            c + 1 <= x1,
                            y1 + h1 <= r,
                            r + 1 <= y1
                        ))
                        
        result = opt.check()
        
        if result == z3.sat:
            m = opt.model()
            # Apply assignments
            for pid in particle_ids:
                p = circuit.particles[pid]
                c_x = m.eval(positions_x[pid]).as_long()
                c_y = m.eval(positions_y[pid]).as_long()
                
                world_x = (c_x + col_start) * self.grid.site_width
                world_y = (c_y + row_start) * self.grid.row_height
                
                p.x, p.y = world_x, world_y
                self.grid.place_cell(pid, world_x, world_y, widths[pid] * self.grid.site_width, heights[pid] * self.grid.row_height)
            return True
            
        print("Z3 SAT solver failed to find legal placement within timeout.")
        return False
        
        
class HybridLegalizer:
    """Orchestrates greedy + SAT legalization."""
    
    def __init__(self, node: str = "7nm", timeout_s: int = 10):
        self.node = node
        self.timeout_s = timeout_s
        
    def run(self, circuit: CircuitParticles) -> dict:
        """Execute full legalization flow."""
        start_time = time.time()
        
        lef_lib = LEFLibrary(node=self.node)
        grid = PlacementGrid(circuit.die_width, circuit.die_height, lef_lib)
        
        # 1. Greedy approach
        tetris = TetrisLegalizer(grid)
        success_count, failed_ids = tetris.legalize(circuit)
        
        z3_resolved = 0
        z3_failed = 0
        
        # 2. SAT approach for failures
        if failed_ids and Z3_AVAILABLE:
            z3_solver = Z3HotspotSolver(grid, timeout_s=self.timeout_s)
            
            # Simple clustering for hotspots
            # In a real tool, we use DBSCAN or similar to find local dense clusters.
            # Here we chunk them up into groups of 10.
            chunk_size = 10
            for i in range(0, len(failed_ids), chunk_size):
                chunk = failed_ids[i:i+chunk_size]
                
                # Compute bounding box
                min_x = min(circuit.particles[pid].x for pid in chunk)
                max_x = max(circuit.particles[pid].x for pid in chunk)
                min_y = min(circuit.particles[pid].y for pid in chunk)
                max_y = max(circuit.particles[pid].y for pid in chunk)
                
                # Expand box considerably to give SAT solver legal space to place them
                pad = 10 * grid.site_width
                box_x = max(0, min_x - pad)
                box_y = max(0, min_y - pad)
                box_w = max_x - min_x + 2*pad
                box_h = max_y - min_y + 2*pad
                
                if z3_solver.solve_region(circuit, chunk, box_x, box_y, box_w, box_h):
                    z3_resolved += len(chunk)
                else:
                    # Fallback: force them into whatever the nearest slot is, even if far away
                    # This violates optimal wirelength but preserves legality
                    for pid in chunk:
                        p = circuit.particles[pid]
                        w, h = lef_lib.get_macro_dimensions(p.type)
                        fx, fy = grid.find_nearest_free(p.x, p.y, w, h, search_radius=500)
                        if fx >= 0:
                            grid.place_cell(p.id, fx, fy, w, h)
                            p.x, p.y = fx, fy
                            z3_resolved += 1
                        else:
                            z3_failed += 1
        elif failed_ids:
            # Fallback when Z3 not available
            z3_failed = len(failed_ids)
            for pid in failed_ids:
                p = circuit.particles[pid]
                w, h = lef_lib.get_macro_dimensions(p.type)
                fx, fy = grid.find_nearest_free(p.x, p.y, w, h, search_radius=500)
                if fx >= 0:
                    grid.place_cell(p.id, fx, fy, w, h)
                    p.x, p.y = fx, fy
            
        elapsed = time.time() - start_time
        
        return {
            "total_particles": len(circuit.particles),
            "tetris_success": success_count,
            "z3_resolved": z3_resolved,
            "failed_illegal": z3_failed,
            "time_s": elapsed
        }
