//! Routing grid for congestion prediction.
//!
//! Overlays the chip area with a grid. Each cell tracks how many
//! horizontal and vertical wire segments cross through it.

use serde::{Deserialize, Serialize};

/// Routing congestion grid.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoutingGrid {
    pub cols: usize,
    pub rows: usize,
    pub cell_width: f32,
    pub cell_height: f32,
    /// Horizontal wire crossing count per cell (row-major: rows * cols).
    pub h_congestion: Vec<f32>,
    /// Vertical wire crossing count per cell.
    pub v_congestion: Vec<f32>,
    /// Routing capacity per cell (how many wires CAN cross).
    pub capacity: Vec<f32>,
}

impl RoutingGrid {
    /// Create a new grid overlaying the chip area.
    pub fn new(chip_width: f32, chip_height: f32, cols: usize, rows: usize) -> Self {
        let num_cells = rows * cols;
        let cell_width = chip_width / cols as f32;
        let cell_height = chip_height / rows as f32;

        // Default capacity: proportional to cell perimeter (a rough heuristic)
        let default_capacity = (cell_width + cell_height) * 0.5;

        Self {
            cols,
            rows,
            cell_width,
            cell_height,
            h_congestion: vec![0.0; num_cells],
            v_congestion: vec![0.0; num_cells],
            capacity: vec![default_capacity; num_cells],
        }
    }

    /// Reset all congestion counts to zero.
    pub fn clear(&mut self) {
        self.h_congestion.fill(0.0);
        self.v_congestion.fill(0.0);
    }

    /// Cell index from (col, row).
    #[inline]
    pub fn cell_index(&self, col: usize, row: usize) -> usize {
        row * self.cols + col
    }

    /// Which grid cell contains point (x, y)?
    #[inline]
    pub fn cell_at(&self, x: f32, y: f32) -> (usize, usize) {
        let col = ((x / self.cell_width) as usize).min(self.cols - 1);
        let row = ((y / self.cell_height) as usize).min(self.rows - 1);
        (col, row)
    }

    /// Total congestion (h + v) at a cell.
    pub fn total_congestion_at(&self, col: usize, row: usize) -> f32 {
        let idx = self.cell_index(col, row);
        self.h_congestion[idx] + self.v_congestion[idx]
    }

    /// Overflow at a cell = max(0, congestion - capacity).
    pub fn overflow_at(&self, col: usize, row: usize) -> f32 {
        let idx = self.cell_index(col, row);
        let total = self.h_congestion[idx] + self.v_congestion[idx];
        (total - self.capacity[idx]).max(0.0)
    }

    /// Maximum overflow across all cells.
    pub fn max_overflow(&self) -> f32 {
        (0..self.rows)
            .flat_map(|r| (0..self.cols).map(move |c| (c, r)))
            .map(|(c, r)| self.overflow_at(c, r))
            .fold(0.0f32, f32::max)
    }

    /// Total overflow across all cells.
    pub fn total_overflow(&self) -> f32 {
        (0..self.rows)
            .flat_map(|r| (0..self.cols).map(move |c| (c, r)))
            .map(|(c, r)| self.overflow_at(c, r))
            .sum()
    }

    /// Find all cells with overflow above a threshold.
    pub fn hot_cells(&self, threshold: f32) -> Vec<(usize, usize)> {
        let mut result = Vec::new();
        for row in 0..self.rows {
            for col in 0..self.cols {
                if self.overflow_at(col, row) > threshold {
                    result.push((col, row));
                }
            }
        }
        result
    }

    /// Get neighbor cells within a radius (Manhattan distance in cells).
    pub fn neighbors(&self, col: usize, row: usize, radius: usize) -> Vec<(usize, usize)> {
        let mut result = Vec::new();
        let r = radius as i32;
        for dr in -r..=r {
            for dc in -r..=r {
                if dr == 0 && dc == 0 {
                    continue;
                }
                let nc = col as i32 + dc;
                let nr = row as i32 + dr;
                if nc >= 0 && nc < self.cols as i32 && nr >= 0 && nr < self.rows as i32 {
                    result.push((nc as usize, nr as usize));
                }
            }
        }
        result
    }

    /// Center point of a grid cell.
    pub fn cell_center(&self, col: usize, row: usize) -> (f32, f32) {
        let x = (col as f32 + 0.5) * self.cell_width;
        let y = (row as f32 + 0.5) * self.cell_height;
        (x, y)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_grid_basic() {
        let grid = RoutingGrid::new(100.0, 100.0, 10, 10);
        assert_eq!(grid.cell_at(5.0, 5.0), (0, 0));
        assert_eq!(grid.cell_at(95.0, 95.0), (9, 9));
        assert_eq!(grid.max_overflow(), 0.0);
    }

    #[test]
    fn test_cell_center() {
        let grid = RoutingGrid::new(100.0, 100.0, 10, 10);
        let (cx, cy) = grid.cell_center(0, 0);
        assert!((cx - 5.0).abs() < 0.01);
        assert!((cy - 5.0).abs() < 0.01);
    }
}
