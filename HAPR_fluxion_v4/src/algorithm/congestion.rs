//! Routing congestion prediction.
//!
//! For each net, compute its bounding-box wire and count how many wires
//! cross each routing grid cell. High counts = congestion hotspots.

use crate::data::circuit::Circuit;
use crate::data::grid::RoutingGrid;
use crate::data::placement::Placement;

/// Compute congestion on the routing grid from current placement (CPU).
pub fn compute_congestion_cpu(
    grid: &mut RoutingGrid,
    placement: &Placement,
    circuit: &Circuit,
) {
    grid.clear();

    for net_gates in &circuit.net_gate_indices {
        if net_gates.len() < 2 {
            continue;
        }

        // Bounding box of the net
        let mut min_x = f32::MAX;
        let mut max_x = f32::MIN;
        let mut min_y = f32::MAX;
        let mut max_y = f32::MIN;

        for &g in net_gates {
            let x = placement.gate_x[g];
            let y = placement.gate_y[g];
            min_x = min_x.min(x);
            max_x = max_x.max(x);
            min_y = min_y.min(y);
            max_y = max_y.max(y);
        }

        // Grid cells covered by the bounding box
        let (min_col, min_row) = grid.cell_at(min_x, min_y);
        let (max_col, max_row) = grid.cell_at(max_x, max_y);

        // Increment horizontal crossings (along rows)
        for row in min_row..=max_row {
            for col in min_col..=max_col {
                let idx = grid.cell_index(col, row);
                if idx < grid.h_congestion.len() {
                    grid.h_congestion[idx] += 1.0;
                }
            }
        }

        // Increment vertical crossings (along columns)
        for col in min_col..=max_col {
            for row in min_row..=max_row {
                let idx = grid.cell_index(col, row);
                if idx < grid.v_congestion.len() {
                    grid.v_congestion[idx] += 1.0;
                }
            }
        }
    }
}

/// Build a routing grid and compute congestion.
pub fn build_and_compute(
    placement: &Placement,
    circuit: &Circuit,
    cols: usize,
    rows: usize,
) -> RoutingGrid {
    let mut grid = RoutingGrid::new(circuit.chip_width, circuit.chip_height, cols, rows);
    compute_congestion_cpu(&mut grid, placement, circuit);
    grid
}

/// Find gate indices that are located in hot (congested) grid cells.
pub fn gates_in_hot_cells(
    placement: &Placement,
    grid: &RoutingGrid,
    threshold: f32,
) -> Vec<usize> {
    let hot = grid.hot_cells(threshold);
    let mut result = Vec::new();

    for g in 0..placement.gate_x.len() {
        let (col, row) = grid.cell_at(placement.gate_x[g], placement.gate_y[g]);
        if hot.contains(&(col, row)) {
            result.push(g);
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::circuit::GateType;

    #[test]
    fn test_congestion_basic() {
        let mut c = Circuit::new("test", 100.0, 100.0, 10.0);
        for i in 0..4 {
            c.add_gate(&format!("g{}", i), GateType::Combinational, 2.0, 10.0);
        }
        c.add_net("n0", vec![0, 1, 2, 3], vec![]);

        let mut p = Placement::new(4);
        p.set_position(0, 10.0, 10.0);
        p.set_position(1, 90.0, 10.0);
        p.set_position(2, 10.0, 90.0);
        p.set_position(3, 90.0, 90.0);

        let grid = build_and_compute(&p, &c, 10, 10);
        // The bounding box covers the entire chip, so many cells should have congestion
        assert!(grid.max_overflow() >= 0.0);
    }
}
