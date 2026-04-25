//! Placement result data structure.
//!
//! Stores per-gate (x, y) positions and quality metrics.

use serde::{Deserialize, Serialize};

/// Final placement output.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Placement {
    /// X position per gate (indexed by gate index).
    pub gate_x: Vec<f32>,
    /// Y position per gate (indexed by gate index).
    pub gate_y: Vec<f32>,

    // ── Quality metrics ──
    /// Half-Perimeter Wire Length (HPWL).
    pub total_wirelength: f32,
    /// Worst-case routing grid cell congestion.
    pub max_congestion: f32,
    /// Worst Negative Slack (WNS) — timing.
    pub wns: f32,
    /// Total placement runtime in milliseconds.
    pub runtime_ms: f64,
}

impl Placement {
    /// Create a new placement with all gates at (0, 0).
    pub fn new(num_gates: usize) -> Self {
        Self {
            gate_x: vec![0.0; num_gates],
            gate_y: vec![0.0; num_gates],
            total_wirelength: 0.0,
            max_congestion: 0.0,
            wns: 0.0,
            runtime_ms: 0.0,
        }
    }

    /// Set position for a single gate.
    pub fn set_position(&mut self, gate_index: usize, x: f32, y: f32) {
        self.gate_x[gate_index] = x;
        self.gate_y[gate_index] = y;
    }

    /// Compute HPWL (Half-Perimeter Wire Length) from current gate positions.
    ///
    /// For each net, HPWL = (max_x - min_x) + (max_y - min_y) across all pins.
    pub fn compute_hpwl(&self, net_gate_indices: &[Vec<usize>]) -> f32 {
        let mut total = 0.0f32;

        for net_gates in net_gate_indices {
            if net_gates.len() < 2 {
                continue;
            }

            let mut min_x = f32::MAX;
            let mut max_x = f32::MIN;
            let mut min_y = f32::MAX;
            let mut max_y = f32::MIN;

            for &g in net_gates {
                let x = self.gate_x[g];
                let y = self.gate_y[g];
                min_x = min_x.min(x);
                max_x = max_x.max(x);
                min_y = min_y.min(y);
                max_y = max_y.max(y);
            }

            total += (max_x - min_x) + (max_y - min_y);
        }

        total
    }

    /// Update the total_wirelength metric from current positions.
    pub fn update_wirelength(&mut self, net_gate_indices: &[Vec<usize>]) {
        self.total_wirelength = self.compute_hpwl(net_gate_indices);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hpwl() {
        let mut p = Placement::new(4);
        p.set_position(0, 0.0, 0.0);
        p.set_position(1, 10.0, 0.0);
        p.set_position(2, 10.0, 10.0);
        p.set_position(3, 0.0, 10.0);

        let nets = vec![vec![0, 1, 2, 3]];
        let hpwl = p.compute_hpwl(&nets);
        assert!((hpwl - 20.0).abs() < 0.01); // (10-0) + (10-0) = 20
    }
}
