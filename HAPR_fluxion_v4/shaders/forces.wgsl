// FLUXION v4 — Force computation shader (WebGPU/WGSL)
// Runs on ANY GPU: NVIDIA, AMD, Intel, Apple via Vulkan/Metal/DX12

struct Gate {
    x: f32,
    y: f32,
    width: f32,
    height: f32,
    force_x: f32,
    force_y: f32,
    _pad0: f32,
    _pad1: f32,
}

struct Params {
    num_gates: u32,
    attract_weight: f32,
    repel_weight: f32,
    dt: f32,
    iteration: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<storage, read_write> gates: array<Gate>;
@group(0) @binding(1) var<uniform> params: Params;

@compute @workgroup_size(256)
fn compute_forces(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if (i >= params.num_gates) { return; }

    var fx: f32 = 0.0;
    var fy: f32 = 0.0;

    // REPULSIVE FORCE: push away from nearby gates
    for (var j: u32 = 0u; j < params.num_gates; j++) {
        if (j == i) { continue; }

        let dx = gates[i].x - gates[j].x;
        let dy = gates[i].y - gates[j].y;
        let dist_sq = dx * dx + dy * dy;
        let dist = max(sqrt(dist_sq), 0.1);

        // Repulsion inversely proportional to distance squared
        let force = params.repel_weight / (dist * dist);
        fx += force * dx / dist;
        fy += force * dy / dist;
    }

    // ATTRACTIVE FORCE: pull toward centroid of all gates (simplified)
    // In full version, this uses net connectivity data
    var cx: f32 = 0.0;
    var cy: f32 = 0.0;
    for (var j: u32 = 0u; j < params.num_gates; j++) {
        cx += gates[j].x;
        cy += gates[j].y;
    }
    cx /= f32(params.num_gates);
    cy /= f32(params.num_gates);

    fx += params.attract_weight * (cx - gates[i].x) * 0.01;
    fy += params.attract_weight * (cy - gates[i].y) * 0.01;

    gates[i].force_x = fx;
    gates[i].force_y = fy;
}

@compute @workgroup_size(256)
fn apply_forces(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if (i >= params.num_gates) { return; }

    gates[i].x += gates[i].force_x * params.dt;
    gates[i].y += gates[i].force_y * params.dt;

    // Reset forces for next iteration
    gates[i].force_x = 0.0;
    gates[i].force_y = 0.0;
}
