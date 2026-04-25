//! Hierarchical circuit partitioner using recursive spectral bisection.
//!
//! Splits the circuit netlist into a binary hierarchy tree. At each level,
//! the graph Laplacian's Fiedler vector (2nd smallest eigenvector) is used
//! to bipartition the gates into two roughly equal halves that minimize
//! the number of cut nets.

use crate::data::circuit::Circuit;
use crate::data::hierarchy::{ClusterNode, HierarchyTree};
use log::info;

/// Configuration for the partitioner.
pub struct PartitionConfig {
    /// Maximum gates per leaf cluster.
    pub max_leaf_size: usize,
    /// Minimum cluster size (don't split below this).
    pub min_cluster_size: usize,
    /// Balance factor: each child gets at least this fraction of the parent.
    pub balance_factor: f32,
    /// Number of power-iteration steps for Fiedler vector.
    pub power_iterations: usize,
}

impl Default for PartitionConfig {
    fn default() -> Self {
        Self {
            max_leaf_size: 64,
            min_cluster_size: 8,
            balance_factor: 0.4,
            power_iterations: 30,
        }
    }
}

/// Build a complete hierarchy tree from the circuit.
pub fn build_hierarchy(circuit: &Circuit, config: &PartitionConfig) -> HierarchyTree {
    let all_gates: Vec<usize> = (0..circuit.num_gates).collect();
    let mut next_id = 0;

    let root = recursive_bisect(circuit, &all_gates, 0, config, &mut next_id);

    info!(
        "Hierarchy built: {} nodes, max_depth={}",
        next_id,
        compute_depth(&root)
    );

    HierarchyTree::new(root, circuit.num_gates)
}

fn compute_depth(node: &ClusterNode) -> usize {
    if node.is_leaf() {
        node.depth
    } else {
        node.children
            .iter()
            .map(|c| compute_depth(c))
            .max()
            .unwrap_or(node.depth)
    }
}

/// Recursively bisect a set of gates using the Fiedler vector.
fn recursive_bisect(
    circuit: &Circuit,
    gate_indices: &[usize],
    depth: usize,
    config: &PartitionConfig,
    next_id: &mut usize,
) -> ClusterNode {
    let id = *next_id;
    *next_id += 1;

    // Base case: small enough to be a leaf
    if gate_indices.len() <= config.max_leaf_size || gate_indices.len() <= config.min_cluster_size {
        return ClusterNode::new_leaf(id, depth, gate_indices.to_vec());
    }

    // Compute Fiedler vector for this subgraph
    let fiedler = compute_fiedler_vector(circuit, gate_indices, config.power_iterations);

    // Split based on Fiedler vector sign
    let (left_gates, right_gates) = split_by_fiedler(&fiedler, gate_indices, config.balance_factor);

    // If split failed (too unbalanced), make a leaf
    if left_gates.is_empty() || right_gates.is_empty() {
        return ClusterNode::new_leaf(id, depth, gate_indices.to_vec());
    }

    // Recurse on each half
    let left_child = recursive_bisect(circuit, &left_gates, depth + 1, config, next_id);
    let right_child = recursive_bisect(circuit, &right_gates, depth + 1, config, next_id);

    ClusterNode::new_internal(
        id,
        depth,
        gate_indices.to_vec(),
        vec![left_child, right_child],
    )
}

/// Compute the Fiedler vector of the subgraph induced by `gate_indices`.
///
/// Uses power iteration on the graph Laplacian to find the eigenvector
/// corresponding to the second-smallest eigenvalue (algebraic connectivity).
fn compute_fiedler_vector(
    circuit: &Circuit,
    gate_indices: &[usize],
    num_iterations: usize,
) -> Vec<f32> {
    let n = gate_indices.len();
    if n <= 1 {
        return vec![0.0; n];
    }

    // Build local index mapping: global gate index → local 0..n
    let mut global_to_local = std::collections::HashMap::new();
    for (local, &global) in gate_indices.iter().enumerate() {
        global_to_local.insert(global, local);
    }

    // Build sparse Laplacian in CSR-like form
    // L = D - A, where D is degree matrix and A is adjacency
    let mut degree = vec![0.0f32; n];
    let mut adj_pairs: Vec<(usize, usize)> = Vec::new();

    for &g in gate_indices {
        if g < circuit.adjacency.len() {
            for &neighbor in &circuit.adjacency[g] {
                if let Some(&local_n) = global_to_local.get(&neighbor) {
                    let local_g = global_to_local[&g];
                    adj_pairs.push((local_g, local_n));
                    degree[local_g] += 1.0;
                }
            }
        }
    }

    // Power iteration to find Fiedler vector
    // We want the 2nd smallest eigenvector of L.
    // Strategy: use inverse power iteration with deflation of the trivial eigenvector (all 1s).

    let mut v = vec![0.0f32; n];
    // Initialize with random-ish values
    for i in 0..n {
        v[i] = (i as f32 * 2.71828).sin();
    }

    // Normalize
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 1e-10 {
        for x in &mut v {
            *x /= norm;
        }
    }

    for _iter in 0..num_iterations {
        // Multiply by Laplacian: L * v = D*v - A*v
        let mut lv = vec![0.0f32; n];
        for i in 0..n {
            lv[i] = degree[i] * v[i];
        }
        for &(i, j) in &adj_pairs {
            lv[i] -= v[j];
        }

        // Deflate: remove component along the trivial eigenvector (1/sqrt(n), ..., 1/sqrt(n))
        let mean: f32 = lv.iter().sum::<f32>() / n as f32;
        for x in &mut lv {
            *x -= mean;
        }

        // Normalize
        let norm: f32 = lv.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 1e-10 {
            for x in &mut lv {
                *x /= norm;
            }
        }

        v = lv;
    }

    v
}

/// Split gate indices based on the sign of the Fiedler vector.
/// Ensures balance: each side gets at least `balance_factor` of the total.
fn split_by_fiedler(
    fiedler: &[f32],
    gate_indices: &[usize],
    balance_factor: f32,
) -> (Vec<usize>, Vec<usize>) {
    let n = gate_indices.len();

    // Sort by Fiedler value
    let mut indexed: Vec<(usize, f32)> = gate_indices
        .iter()
        .enumerate()
        .map(|(i, &g)| (g, fiedler[i]))
        .collect();
    indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

    // Find split point respecting balance
    let min_size = (n as f32 * balance_factor).ceil() as usize;
    let split = n / 2;
    let split = split.max(min_size).min(n - min_size);

    let left: Vec<usize> = indexed[..split].iter().map(|(g, _)| *g).collect();
    let right: Vec<usize> = indexed[split..].iter().map(|(g, _)| *g).collect();

    (left, right)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::circuit::GateType;

    fn make_chain_circuit(n: usize) -> Circuit {
        let mut c = Circuit::new("chain", 1000.0, 1000.0, 10.0);
        for i in 0..n {
            c.add_gate(&format!("g{}", i), GateType::Combinational, 2.0, 10.0);
        }
        for i in 0..(n - 1) {
            c.add_net(&format!("n{}", i), vec![i, i + 1], vec![]);
        }
        c.build_adjacency();
        c
    }

    #[test]
    fn test_partition_small_chain() {
        let c = make_chain_circuit(20);
        let config = PartitionConfig {
            max_leaf_size: 8,
            ..Default::default()
        };
        let tree = build_hierarchy(&c, &config);
        assert!(tree.max_depth >= 1);
        assert!(tree.leaves().len() >= 2);
        // All gates should be covered
        let mut all_leaf_gates: Vec<usize> = tree
            .leaves()
            .iter()
            .flat_map(|l| l.gate_indices.iter().copied())
            .collect();
        all_leaf_gates.sort();
        all_leaf_gates.dedup();
        assert_eq!(all_leaf_gates.len(), 20);
    }
}
