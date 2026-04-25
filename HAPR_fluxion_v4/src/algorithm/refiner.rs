//! Congestion-aware refinement.
//!
//! Iteratively moves gates away from congested routing grid cells
//! toward less congested neighbors. Also performs greedy swaps to
//! reduce wirelength.

use crate::data::circuit::Circuit;
use crate::data::grid::RoutingGrid;
use crate::data::placement::Placement;
use crate::algorithm::congestion;
use log::info;

pub struct RefineConfig {
    pub max_iterations: usize,
    pub move_distance: f32,
    pub overflow_threshold: f32,
    pub grid_cols: usize,
    pub grid_rows: usize,
}

impl Default for RefineConfig {
    fn default() -> Self {
        Self {
            max_iterations: 100,
            move_distance: 5.0,
            overflow_threshold: 0.0,
            grid_cols: 256,
            grid_rows: 256,
        }
    }
}

/// Run congestion-aware refinement on the placement.
pub fn refine_placement(
    placement: &mut Placement,
    circuit: &Circuit,
    config: &RefineConfig,
) -> RoutingGrid {
    let mut grid = RoutingGrid::new(
        circuit.chip_width,
        circuit.chip_height,
        config.grid_cols,
        config.grid_rows,
    );

    let mut prev_overflow = f32::MAX;
    let mut stall_count = 0;

    for iter in 0..config.max_iterations {
        congestion::compute_congestion_cpu(&mut grid, placement, circuit);
        let overflow = grid.total_overflow();

        if overflow <= config.overflow_threshold {
            info!("Refinement converged at iter {}: overflow={:.1}", iter, overflow);
            break;
        }

        // Stall detection
        if (prev_overflow - overflow).abs() < 0.01 * prev_overflow.max(1.0) {
            stall_count += 1;
            if stall_count >= 10 {
                info!("Refinement stalled at iter {}: overflow={:.1}", iter, overflow);
                break;
            }
        } else {
            stall_count = 0;
        }
        prev_overflow = overflow;

        // Move gates from hot cells toward cooler neighbors
        let hot_gates = congestion::gates_in_hot_cells(placement, &grid, config.overflow_threshold);

        for &gate in &hot_gates {
            move_gate_from_hot_cell(gate, placement, &grid, circuit, config.move_distance);
        }
    }

    // Final congestion computation
    congestion::compute_congestion_cpu(&mut grid, placement, circuit);
    placement.max_congestion = grid.max_overflow();
    placement.update_wirelength(&circuit.net_gate_indices);

    info!(
        "Refinement done. HPWL={:.1}, max_overflow={:.1}",
        placement.total_wirelength, placement.max_congestion
    );

    grid
}

/// Move a gate toward the least-congested nearby grid cell.
fn move_gate_from_hot_cell(
    gate: usize,
    placement: &mut Placement,
    grid: &RoutingGrid,
    circuit: &Circuit,
    max_move: f32,
) {
    let gx = placement.gate_x[gate];
    let gy = placement.gate_y[gate];
    let (col, row) = grid.cell_at(gx, gy);

    let neighbors = grid.neighbors(col, row, 3);
    if neighbors.is_empty() {
        return;
    }

    // Find neighbor with lowest total congestion
    let best = neighbors
        .iter()
        .min_by(|a, b| {
            let ca = grid.total_congestion_at(a.0, a.1);
            let cb = grid.total_congestion_at(b.0, b.1);
            ca.partial_cmp(&cb).unwrap_or(std::cmp::Ordering::Equal)
        });

    if let Some(&(bc, br)) = best {
        let (tx, ty) = grid.cell_center(bc, br);
        let dx = (tx - gx).clamp(-max_move, max_move);
        let dy = (ty - gy).clamp(-max_move, max_move);

        let new_x = (gx + dx).clamp(0.0, circuit.chip_width);
        let new_y = (gy + dy).clamp(0.0, circuit.chip_height);
        placement.set_position(gate, new_x, new_y);
    }
}

/// Try swapping two gates; keep the swap if it reduces wirelength.
pub fn swap_if_better(
    gate_a: usize,
    gate_b: usize,
    placement: &mut Placement,
    circuit: &Circuit,
) -> bool {
    let before = placement.compute_hpwl(&circuit.net_gate_indices);

    // Swap
    let ax = placement.gate_x[gate_a];
    let ay = placement.gate_y[gate_a];
    placement.set_position(gate_a, placement.gate_x[gate_b], placement.gate_y[gate_b]);
    placement.set_position(gate_b, ax, ay);

    let after = placement.compute_hpwl(&circuit.net_gate_indices);

    if after < before {
        true // keep swap
    } else {
        // Undo
        let bx = placement.gate_x[gate_a];
        let by = placement.gate_y[gate_a];
        placement.set_position(gate_a, placement.gate_x[gate_b], placement.gate_y[gate_b]);
        placement.set_position(gate_b, bx, by);
        false
    }
}
