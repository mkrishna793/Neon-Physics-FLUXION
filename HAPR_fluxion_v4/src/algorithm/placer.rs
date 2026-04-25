//! Multi-level placer: the heart of HAPR.
//!
//! 1. Place top-level clusters via spectral embedding of attention graph
//! 2. Recursively place children within parent regions
//! 3. Place individual gates within leaf clusters using force-directed placement

use crate::data::circuit::Circuit;
use crate::data::hierarchy::{ClusterNode, HierarchyTree};
use crate::data::placement::Placement;
use crate::algorithm::attention::AttentionMap;
use log::info;

pub struct PlacerConfig {
    pub force_iterations: usize,
    pub attract_weight: f32,
    pub repel_weight: f32,
    pub damping: f32,
    pub dt: f32,
}

impl Default for PlacerConfig {
    fn default() -> Self {
        Self {
            force_iterations: 50,
            attract_weight: 1.0,
            repel_weight: 0.5,
            damping: 0.9,
            dt: 0.1,
        }
    }
}

/// Run the full HAPR placement pipeline.
pub fn hapr_placement(
    circuit: &Circuit,
    tree: &mut HierarchyTree,
    attention: &AttentionMap,
    config: &PlacerConfig,
) -> Placement {
    let mut placement = Placement::new(circuit.num_gates);

    // Set root region to full chip
    tree.root.set_region(
        circuit.chip_width / 2.0,
        circuit.chip_height / 2.0,
        circuit.chip_width,
        circuit.chip_height,
    );

    // Place each level top-down
    for depth in 0..=tree.max_depth {
        let attn_level = attention.levels.iter().find(|l| l.level_index == depth);

        if depth == 0 && tree.root.children.len() > 1 {
            // Place top-level clusters using spectral embedding
            if let Some(attn) = attn_level {
                place_children_spectral(&mut tree.root, attn);
            }
        } else if depth > 0 {
            // Place children within parent regions
            place_level_recursive(&mut tree.root, depth, attn_level);
        }
    }

    // Gate-level placement within leaf clusters
    let leaves = tree.leaves();
    for leaf in &leaves {
        place_gates_in_leaf(leaf, circuit, &mut placement, config);
    }

    placement.update_wirelength(&circuit.net_gate_indices);
    info!("HAPR placement complete. HPWL = {:.1}", placement.total_wirelength);
    placement
}

/// Place children of a node using spectral embedding of the attention matrix.
fn place_children_spectral(
    parent: &mut ClusterNode,
    attn: &crate::algorithm::attention::AttentionLevel,
) {
    let n = parent.children.len();
    if n == 0 { return; }

    // Spectral embedding: use attention matrix as weighted adjacency
    let positions = spectral_embed_2d(&attn.scores, n);

    // Scale positions to parent region
    let half_w = parent.width / 2.0;
    let half_h = parent.height / 2.0;
    let origin_x = parent.center_x - half_w;
    let origin_y = parent.center_y - half_h;

    for (i, child) in parent.children.iter_mut().enumerate() {
        if i < positions.len() {
            let (nx, ny) = positions[i];
            // Map [-1,1] → parent region
            let cx = origin_x + (nx + 1.0) * 0.5 * parent.width;
            let cy = origin_y + (ny + 1.0) * 0.5 * parent.height;
            let cw = parent.width / (n as f32).sqrt();
            let ch = parent.height / (n as f32).sqrt();
            child.set_region(
                cx.clamp(origin_x + cw / 2.0, origin_x + parent.width - cw / 2.0),
                cy.clamp(origin_y + ch / 2.0, origin_y + parent.height - ch / 2.0),
                cw, ch,
            );
        }
    }
}

/// Recursively place children at a specific depth level.
fn place_level_recursive(
    node: &mut ClusterNode,
    target_depth: usize,
    attn: Option<&crate::algorithm::attention::AttentionLevel>,
) {
    if node.depth + 1 == target_depth && !node.children.is_empty() {
        let n = node.children.len();
        let half_w = node.width / 2.0;
        let half_h = node.height / 2.0;
        let origin_x = node.center_x - half_w;
        let origin_y = node.center_y - half_h;

        // Simple grid-based placement within parent region
        let cols = (n as f32).sqrt().ceil() as usize;
        let rows = (n + cols - 1) / cols;

        for (i, child) in node.children.iter_mut().enumerate() {
            let col = i % cols;
            let row = i / cols;
            let cw = node.width / cols as f32;
            let ch = node.height / rows as f32;
            let cx = origin_x + (col as f32 + 0.5) * cw;
            let cy = origin_y + (row as f32 + 0.5) * ch;
            child.set_region(cx, cy, cw, ch);
        }
    } else {
        for child in &mut node.children {
            place_level_recursive(child, target_depth, attn);
        }
    }
}

/// Simple 2D spectral embedding using the Laplacian of a similarity matrix.
fn spectral_embed_2d(similarity: &[Vec<f32>], n: usize) -> Vec<(f32, f32)> {
    if n <= 1 {
        return vec![(0.0, 0.0); n];
    }

    // Build Laplacian: L = D - W
    let mut degree = vec![0.0f32; n];
    for i in 0..n {
        for j in 0..n {
            if i < similarity.len() && j < similarity[i].len() {
                degree[i] += similarity[i][j];
            }
        }
    }

    // Find 2nd and 3rd eigenvectors via power iteration with deflation
    let eigvec1 = power_iteration_laplacian(similarity, &degree, n, 30, None);
    let eigvec2 = power_iteration_laplacian(similarity, &degree, n, 30, Some(&eigvec1));

    eigvec1.iter().zip(eigvec2.iter())
        .map(|(&x, &y)| (x, y))
        .collect()
}

/// Power iteration on L to find an eigenvector, with optional deflation.
fn power_iteration_laplacian(
    similarity: &[Vec<f32>],
    degree: &[f32],
    n: usize,
    iters: usize,
    deflate: Option<&[f32]>,
) -> Vec<f32> {
    let mut v: Vec<f32> = (0..n).map(|i| (i as f32 * 1.618).sin()).collect();

    for _ in 0..iters {
        // L*v = D*v - W*v
        let mut lv = vec![0.0f32; n];
        for i in 0..n {
            lv[i] = degree[i] * v[i];
            for j in 0..n {
                if i < similarity.len() && j < similarity[i].len() {
                    lv[i] -= similarity[i][j] * v[j];
                }
            }
        }

        // Deflate trivial eigenvector
        let mean: f32 = lv.iter().sum::<f32>() / n as f32;
        for x in &mut lv { *x -= mean; }

        // Deflate previous eigenvector if provided
        if let Some(prev) = deflate {
            let dot: f32 = lv.iter().zip(prev).map(|(a, b)| a * b).sum();
            for (x, p) in lv.iter_mut().zip(prev) { *x -= dot * p; }
        }

        let norm: f32 = lv.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 1e-10 {
            for x in &mut lv { *x /= norm; }
        }
        v = lv;
    }
    v
}

/// Place individual gates within a leaf cluster using force-directed placement.
fn place_gates_in_leaf(
    leaf: &ClusterNode,
    circuit: &Circuit,
    placement: &mut Placement,
    config: &PlacerConfig,
) {
    let gates = &leaf.gate_indices;
    let n = gates.len();
    if n == 0 { return; }

    let half_w = leaf.width / 2.0;
    let half_h = leaf.height / 2.0;
    let min_x = leaf.center_x - half_w;
    let min_y = leaf.center_y - half_h;

    // Initialize positions within the leaf region
    let mut pos_x: Vec<f32> = (0..n).map(|i| min_x + (i as f32 / n as f32) * leaf.width).collect();
    let mut pos_y: Vec<f32> = (0..n).map(|i| min_y + ((i * 7 % n) as f32 / n as f32) * leaf.height).collect();
    let mut vel_x = vec![0.0f32; n];
    let mut vel_y = vec![0.0f32; n];

    // Local index mapping
    let mut global_to_local = std::collections::HashMap::new();
    for (local, &global) in gates.iter().enumerate() {
        global_to_local.insert(global, local);
    }

    // Force-directed iterations (CPU fallback)
    for _ in 0..config.force_iterations {
        let mut fx = vec![0.0f32; n];
        let mut fy = vec![0.0f32; n];

        // Attractive forces: connected gates pull together
        for &g in gates {
            let li = global_to_local[&g];
            if g < circuit.adjacency.len() {
                for &neighbor in &circuit.adjacency[g] {
                    if let Some(&lj) = global_to_local.get(&neighbor) {
                        let dx = pos_x[lj] - pos_x[li];
                        let dy = pos_y[lj] - pos_y[li];
                        fx[li] += config.attract_weight * dx;
                        fy[li] += config.attract_weight * dy;
                    }
                }
            }
        }

        // Repulsive forces: all gates push apart
        for i in 0..n {
            for j in (i + 1)..n {
                let dx = pos_x[i] - pos_x[j];
                let dy = pos_y[i] - pos_y[j];
                let dist_sq = dx * dx + dy * dy;
                let dist = dist_sq.sqrt().max(0.1);
                let force = config.repel_weight / (dist * dist);
                let fx_ij = force * dx / dist;
                let fy_ij = force * dy / dist;
                fx[i] += fx_ij;
                fy[i] += fy_ij;
                fx[j] -= fx_ij;
                fy[j] -= fy_ij;
            }
        }

        // Update velocities and positions
        for i in 0..n {
            vel_x[i] = (vel_x[i] + fx[i] * config.dt) * config.damping;
            vel_y[i] = (vel_y[i] + fy[i] * config.dt) * config.damping;
            pos_x[i] = (pos_x[i] + vel_x[i] * config.dt).clamp(min_x, min_x + leaf.width);
            pos_y[i] = (pos_y[i] + vel_y[i] * config.dt).clamp(min_y, min_y + leaf.height);
        }
    }

    // Write final positions to placement
    for (local, &global) in gates.iter().enumerate() {
        placement.set_position(global, pos_x[local], pos_y[local]);
    }
}
