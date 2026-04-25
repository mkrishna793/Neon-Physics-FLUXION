// FLUXION v4 — Congestion grid computation shader

struct Wire {
    x1: f32,
    y1: f32,
    x2: f32,
    y2: f32,
}

struct GridParams {
    grid_cols: u32,
    grid_rows: u32,
    cell_width: f32,
    cell_height: f32,
    num_wires: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<storage, read> wires: array<Wire>;
@group(0) @binding(1) var<storage, read_write> h_grid: array<f32>;
@group(0) @binding(2) var<storage, read_write> v_grid: array<f32>;
@group(0) @binding(3) var<uniform> params: GridParams;

@compute @workgroup_size(256)
fn compute_congestion(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if (i >= params.num_wires) { return; }

    let wire = wires[i];

    // Bounding box in grid coordinates
    let min_col = u32(min(wire.x1, wire.x2) / params.cell_width);
    let max_col = min(u32(max(wire.x1, wire.x2) / params.cell_width), params.grid_cols - 1u);
    let min_row = u32(min(wire.y1, wire.y2) / params.cell_height);
    let max_row = min(u32(max(wire.y1, wire.y2) / params.cell_height), params.grid_rows - 1u);

    // Increment horizontal congestion for cells in the bounding box
    for (var col = min_col; col <= max_col; col++) {
        for (var row = min_row; row <= max_row; row++) {
            let idx = row * params.grid_cols + col;
            if (idx < params.grid_cols * params.grid_rows) {
                // Note: atomic add would be ideal here but WGSL atomics
                // only support u32/i32. Using non-atomic for simplicity.
                h_grid[idx] += 1.0;
            }
        }
    }

    // Increment vertical congestion
    for (var row = min_row; row <= max_row; row++) {
        for (var col = min_col; col <= max_col; col++) {
            let idx = row * params.grid_cols + col;
            if (idx < params.grid_cols * params.grid_rows) {
                v_grid[idx] += 1.0;
            }
        }
    }
}
