//! Core circuit data structure for FLUXION v4.
//!
//! Flat arrays, not nested objects. Every gate is an index into parallel arrays.
//! This design enables SIMD, GPU upload, and cache-friendly traversal.

use serde::{Deserialize, Serialize};

/// Type of logic gate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum GateType {
    /// Combinational logic: AND, OR, XOR, MUX, BUF, INV, etc.
    Combinational,
    /// Sequential logic: Flip-flop, latch.
    Sequential,
    /// Primary input/output pad.
    Io,
    /// Hard macro: memory block, hard IP.
    Macro,
}

/// Pin direction.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PinDirection {
    Input,
    Output,
}

/// A single pin on a gate.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Pin {
    pub gate_index: usize,
    pub offset_x: f32,
    pub offset_y: f32,
    pub direction: PinDirection,
    pub name: String,
}

/// The core circuit representation.
///
/// All parallel arrays are indexed by gate index `0..num_gates`.
/// Nets store which gate indices they touch.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Circuit {
    pub name: String,

    // ── Gates (parallel arrays) ──
    pub num_gates: usize,
    pub gate_names: Vec<String>,
    pub gate_types: Vec<GateType>,
    pub gate_widths: Vec<f32>,
    pub gate_heights: Vec<f32>,

    // ── Pins ──
    pub pins: Vec<Pin>,
    /// For each gate, the range of pin indices: pins[pin_ranges[g].0 .. pin_ranges[g].1]
    pub pin_ranges: Vec<(usize, usize)>,

    // ── Nets ──
    pub num_nets: usize,
    pub net_names: Vec<String>,
    /// For each net, the list of gate indices it connects.
    pub net_gate_indices: Vec<Vec<usize>>,
    /// For each net, the list of pin indices it connects.
    pub net_pin_indices: Vec<Vec<usize>>,

    // ── Adjacency (computed once from nets) ──
    /// For each gate, the set of connected gate indices.
    pub adjacency: Vec<Vec<usize>>,

    // ── Timing (optional) ──
    /// Timing slack per gate (positive = met, negative = violation).
    pub gate_slacks: Vec<f32>,

    // ── Chip bounding box ──
    pub chip_width: f32,
    pub chip_height: f32,
    pub row_height: f32,
}

impl Circuit {
    /// Create an empty circuit with the given name and die dimensions.
    pub fn new(name: &str, chip_width: f32, chip_height: f32, row_height: f32) -> Self {
        Self {
            name: name.to_string(),
            num_gates: 0,
            gate_names: Vec::new(),
            gate_types: Vec::new(),
            gate_widths: Vec::new(),
            gate_heights: Vec::new(),
            pins: Vec::new(),
            pin_ranges: Vec::new(),
            num_nets: 0,
            net_names: Vec::new(),
            net_gate_indices: Vec::new(),
            net_pin_indices: Vec::new(),
            adjacency: Vec::new(),
            gate_slacks: Vec::new(),
            chip_width,
            chip_height,
            row_height,
        }
    }

    /// Add a gate to the circuit. Returns the gate index.
    pub fn add_gate(
        &mut self,
        name: &str,
        gate_type: GateType,
        width: f32,
        height: f32,
    ) -> usize {
        let idx = self.num_gates;
        self.gate_names.push(name.to_string());
        self.gate_types.push(gate_type);
        self.gate_widths.push(width);
        self.gate_heights.push(height);
        self.pin_ranges.push((self.pins.len(), self.pins.len()));
        self.gate_slacks.push(0.0);
        self.num_gates += 1;
        idx
    }

    /// Add a pin to a gate. Call after add_gate.
    pub fn add_pin(
        &mut self,
        gate_index: usize,
        name: &str,
        offset_x: f32,
        offset_y: f32,
        direction: PinDirection,
    ) -> usize {
        let pin_idx = self.pins.len();
        self.pins.push(Pin {
            gate_index,
            offset_x,
            offset_y,
            direction,
            name: name.to_string(),
        });
        // Extend the gate's pin range
        self.pin_ranges[gate_index].1 = pin_idx + 1;
        pin_idx
    }

    /// Add a net connecting the given gate indices.
    pub fn add_net(&mut self, name: &str, gate_indices: Vec<usize>, pin_indices: Vec<usize>) {
        self.net_names.push(name.to_string());
        self.net_gate_indices.push(gate_indices);
        self.net_pin_indices.push(pin_indices);
        self.num_nets += 1;
    }

    /// Build adjacency lists from net connectivity. Call once after all nets are added.
    pub fn build_adjacency(&mut self) {
        self.adjacency = vec![Vec::new(); self.num_gates];

        for net_gates in &self.net_gate_indices {
            for i in 0..net_gates.len() {
                for j in (i + 1)..net_gates.len() {
                    let a = net_gates[i];
                    let b = net_gates[j];
                    if !self.adjacency[a].contains(&b) {
                        self.adjacency[a].push(b);
                    }
                    if !self.adjacency[b].contains(&a) {
                        self.adjacency[b].push(a);
                    }
                }
            }
        }
    }

    /// Count the number of nets that touch at least one gate in `gates_a`
    /// AND at least one gate in `gates_b`.
    pub fn count_shared_nets(&self, gates_a: &[usize], gates_b: &[usize]) -> usize {
        let mut count = 0;
        for net_gates in &self.net_gate_indices {
            let touches_a = net_gates.iter().any(|g| gates_a.contains(g));
            let touches_b = net_gates.iter().any(|g| gates_b.contains(g));
            if touches_a && touches_b {
                count += 1;
            }
        }
        count
    }

    /// Count total nets touching any gate in the given set.
    pub fn count_nets_touching(&self, gates: &[usize]) -> usize {
        self.net_gate_indices
            .iter()
            .filter(|net_gates| net_gates.iter().any(|g| gates.contains(g)))
            .count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_circuit_basic() {
        let mut c = Circuit::new("test", 1000.0, 1000.0, 10.0);
        let g0 = c.add_gate("AND1", GateType::Combinational, 2.0, 10.0);
        let g1 = c.add_gate("OR1", GateType::Combinational, 2.0, 10.0);
        let g2 = c.add_gate("FF1", GateType::Sequential, 4.0, 10.0);

        c.add_net("n0", vec![g0, g1], vec![]);
        c.add_net("n1", vec![g1, g2], vec![]);
        c.build_adjacency();

        assert_eq!(c.num_gates, 3);
        assert_eq!(c.adjacency[0], vec![1]);
        assert_eq!(c.adjacency[1], vec![0, 2]);
        assert_eq!(c.count_shared_nets(&[0], &[1]), 1);
        assert_eq!(c.count_nets_touching(&[1]), 2);
    }
}
