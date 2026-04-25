// FLUXION v4 — Parallel bitonic sort shader
// Used for ranking attention scores on GPU

@group(0) @binding(0) var<storage, read_write> data: array<f32>;
@group(0) @binding(1) var<uniform> sort_params: SortParams;

struct SortParams {
    n: u32,
    stage: u32,
    step_val: u32,
    _pad: u32,
}

@compute @workgroup_size(256)
fn bitonic_step(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if (i >= sort_params.n) { return; }

    let block_size = 1u << (sort_params.stage + 1u);
    let half_block = 1u << sort_params.stage;

    let partner = i ^ half_block;

    if (partner > i && partner < sort_params.n) {
        let ascending = (i & block_size) == 0u;

        let val_i = data[i];
        let val_p = data[partner];

        if (ascending && val_i > val_p) {
            data[i] = val_p;
            data[partner] = val_i;
        }
        if (!ascending && val_i < val_p) {
            data[i] = val_p;
            data[partner] = val_i;
        }
    }
}
