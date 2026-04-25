//! Hierarchy tree for HAPR multi-level placement.
//!
//! The hierarchy is a tree of clusters. The root contains all gates.
//! Each internal node's children partition its gates.
//! Leaves contain ≤ max_leaf_size gates for force-directed placement.

use serde::{Deserialize, Serialize};

/// A single node in the hierarchy tree.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClusterNode {
    pub id: usize,
    pub depth: usize,
    /// Gate indices belonging to this cluster (union of children's gates for internal nodes).
    pub gate_indices: Vec<usize>,
    /// Child clusters (empty for leaves).
    pub children: Vec<ClusterNode>,

    // ── Placement region (filled during placement) ──
    pub center_x: f32,
    pub center_y: f32,
    pub width: f32,
    pub height: f32,
}

impl ClusterNode {
    pub fn new_leaf(id: usize, depth: usize, gate_indices: Vec<usize>) -> Self {
        Self {
            id,
            depth,
            gate_indices,
            children: Vec::new(),
            center_x: 0.0,
            center_y: 0.0,
            width: 0.0,
            height: 0.0,
        }
    }

    pub fn new_internal(
        id: usize,
        depth: usize,
        gate_indices: Vec<usize>,
        children: Vec<ClusterNode>,
    ) -> Self {
        Self {
            id,
            depth,
            gate_indices,
            children,
            center_x: 0.0,
            center_y: 0.0,
            width: 0.0,
            height: 0.0,
        }
    }

    /// Is this node a leaf (no children)?
    pub fn is_leaf(&self) -> bool {
        self.children.is_empty()
    }

    /// Number of gates in this cluster.
    pub fn num_gates(&self) -> usize {
        self.gate_indices.len()
    }

    /// Set the bounding region for this cluster.
    pub fn set_region(&mut self, cx: f32, cy: f32, w: f32, h: f32) {
        self.center_x = cx;
        self.center_y = cy;
        self.width = w;
        self.height = h;
    }
}

/// The complete hierarchy tree.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HierarchyTree {
    pub root: ClusterNode,
    pub max_depth: usize,
    /// Map from gate_index → leaf cluster id.
    pub leaf_map: Vec<usize>,
    /// Total number of cluster nodes (for id generation).
    pub num_nodes: usize,
}

impl HierarchyTree {
    pub fn new(root: ClusterNode, num_gates: usize) -> Self {
        let max_depth = Self::compute_max_depth(&root);
        let mut leaf_map = vec![0; num_gates];
        Self::build_leaf_map(&root, &mut leaf_map);
        let num_nodes = Self::count_nodes(&root);

        Self {
            root,
            max_depth,
            leaf_map,
            num_nodes,
        }
    }

    fn compute_max_depth(node: &ClusterNode) -> usize {
        if node.is_leaf() {
            node.depth
        } else {
            node.children
                .iter()
                .map(|c| Self::compute_max_depth(c))
                .max()
                .unwrap_or(node.depth)
        }
    }

    fn build_leaf_map(node: &ClusterNode, leaf_map: &mut Vec<usize>) {
        if node.is_leaf() {
            for &g in &node.gate_indices {
                if g < leaf_map.len() {
                    leaf_map[g] = node.id;
                }
            }
        } else {
            for child in &node.children {
                Self::build_leaf_map(child, leaf_map);
            }
        }
    }

    fn count_nodes(node: &ClusterNode) -> usize {
        1 + node.children.iter().map(|c| Self::count_nodes(c)).sum::<usize>()
    }

    /// Collect all clusters at a given depth level.
    pub fn clusters_at_depth(&self, target_depth: usize) -> Vec<&ClusterNode> {
        let mut result = Vec::new();
        Self::collect_at_depth(&self.root, target_depth, &mut result);
        result
    }

    /// Collect mutable references to all clusters at a given depth.
    pub fn clusters_at_depth_mut(&mut self, target_depth: usize) -> Vec<&mut ClusterNode> {
        let mut result = Vec::new();
        Self::collect_at_depth_mut(&mut self.root, target_depth, &mut result);
        result
    }

    fn collect_at_depth<'a>(
        node: &'a ClusterNode,
        target: usize,
        out: &mut Vec<&'a ClusterNode>,
    ) {
        if node.depth == target {
            out.push(node);
        } else if node.depth < target {
            for child in &node.children {
                Self::collect_at_depth(child, target, out);
            }
        }
    }

    fn collect_at_depth_mut<'a>(
        node: &'a mut ClusterNode,
        target: usize,
        out: &mut Vec<&'a mut ClusterNode>,
    ) {
        if node.depth == target {
            out.push(node);
        } else if node.depth < target {
            for child in &mut node.children {
                Self::collect_at_depth_mut(child, target, out);
            }
        }
    }

    /// Collect all leaf nodes.
    pub fn leaves(&self) -> Vec<&ClusterNode> {
        let mut result = Vec::new();
        Self::collect_leaves(&self.root, &mut result);
        result
    }

    fn collect_leaves<'a>(node: &'a ClusterNode, out: &mut Vec<&'a ClusterNode>) {
        if node.is_leaf() {
            out.push(node);
        } else {
            for child in &node.children {
                Self::collect_leaves(child, out);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hierarchy_basic() {
        let left = ClusterNode::new_leaf(1, 1, vec![0, 1, 2]);
        let right = ClusterNode::new_leaf(2, 1, vec![3, 4, 5]);
        let root = ClusterNode::new_internal(0, 0, vec![0, 1, 2, 3, 4, 5], vec![left, right]);
        let tree = HierarchyTree::new(root, 6);

        assert_eq!(tree.max_depth, 1);
        assert_eq!(tree.num_nodes, 3);
        assert_eq!(tree.leaves().len(), 2);
        assert_eq!(tree.clusters_at_depth(0).len(), 1);
        assert_eq!(tree.clusters_at_depth(1).len(), 2);
    }
}
