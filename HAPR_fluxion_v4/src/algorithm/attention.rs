//! Attention score computation for HAPR.
//!
//! For each pair of clusters at the same hierarchy level, compute:
//!   attention(A→B) = shared_nets(A, B) / total_nets(A)

use crate::data::circuit::Circuit;
use crate::data::hierarchy::HierarchyTree;
use log::debug;
use std::collections::HashSet;

#[derive(Debug, Clone)]
pub struct AttentionLevel {
    pub level_index: usize,
    pub scores: Vec<Vec<f32>>,
    pub top_k: Vec<Vec<usize>>,
}

#[derive(Debug, Clone)]
pub struct AttentionMap {
    pub levels: Vec<AttentionLevel>,
}

pub struct AttentionConfig {
    pub top_k: usize,
}

impl Default for AttentionConfig {
    fn default() -> Self {
        Self { top_k: 10 }
    }
}

pub fn compute_all_attention(
    tree: &HierarchyTree,
    circuit: &Circuit,
    config: &AttentionConfig,
) -> AttentionMap {
    let mut levels = Vec::new();
    for depth in 0..=tree.max_depth {
        let clusters = tree.clusters_at_depth(depth);
        if clusters.len() < 2 {
            continue;
        }
        let level = compute_level_attention(&clusters, circuit, depth, config);
        levels.push(level);
    }
    AttentionMap { levels }
}

fn compute_level_attention(
    clusters: &[&crate::data::hierarchy::ClusterNode],
    circuit: &Circuit,
    level_index: usize,
    config: &AttentionConfig,
) -> AttentionLevel {
    let n = clusters.len();
    let mut scores = vec![vec![0.0f32; n]; n];

    let cluster_nets: Vec<HashSet<usize>> = clusters
        .iter()
        .map(|cluster| nets_touching_gates(circuit, &cluster.gate_indices))
        .collect();

    for i in 0..n {
        let total_i = cluster_nets[i].len();
        if total_i == 0 { continue; }
        for j in 0..n {
            if i == j { continue; }
            let shared = cluster_nets[i].intersection(&cluster_nets[j]).count();
            scores[i][j] = shared as f32 / total_i as f32;
        }
    }

    let top_k: Vec<Vec<usize>> = (0..n)
        .map(|i| {
            let mut indexed: Vec<(usize, f32)> = scores[i]
                .iter().enumerate()
                .filter(|&(j, _)| j != i)
                .map(|(j, &s)| (j, s))
                .collect();
            indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            indexed.into_iter().take(config.top_k).filter(|(_, s)| *s > 0.0).map(|(j, _)| j).collect()
        })
        .collect();

    debug!("Level {} attention: {} clusters", level_index, n);
    AttentionLevel { level_index, scores, top_k }
}

fn nets_touching_gates(circuit: &Circuit, gates: &[usize]) -> HashSet<usize> {
    let gate_set: HashSet<usize> = gates.iter().copied().collect();
    let mut result = HashSet::new();
    for (net_idx, net_gates) in circuit.net_gate_indices.iter().enumerate() {
        if net_gates.iter().any(|g| gate_set.contains(g)) {
            result.insert(net_idx);
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::circuit::GateType;
    use crate::data::hierarchy::{ClusterNode, HierarchyTree};

    #[test]
    fn test_attention_basic() {
        let mut c = Circuit::new("test", 100.0, 100.0, 10.0);
        for i in 0..6 {
            c.add_gate(&format!("g{}", i), GateType::Combinational, 2.0, 10.0);
        }
        c.add_net("n0", vec![0, 1], vec![]);
        c.add_net("n1", vec![0, 3], vec![]);
        c.add_net("n2", vec![3, 4], vec![]);
        c.build_adjacency();

        let left = ClusterNode::new_leaf(1, 1, vec![0, 1, 2]);
        let right = ClusterNode::new_leaf(2, 1, vec![3, 4, 5]);
        let root = ClusterNode::new_internal(0, 0, vec![0, 1, 2, 3, 4, 5], vec![left, right]);
        let tree = HierarchyTree::new(root, 6);

        let config = AttentionConfig { top_k: 5 };
        let attn = compute_all_attention(&tree, &c, &config);
        assert!(!attn.levels.is_empty());
    }
}
