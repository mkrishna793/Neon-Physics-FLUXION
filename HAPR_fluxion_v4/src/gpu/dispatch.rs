//! GPU compute shader dispatch helpers.
//!
//! Orchestrates running WGSL shaders for force computation and congestion.

use super::buffers::*;
use super::context::GpuContext;
use crate::data::circuit::Circuit;
use crate::data::grid::RoutingGrid;
use crate::data::placement::Placement;

const WORKGROUP_SIZE: u32 = 256;

/// Shader source constants (embedded at compile time).
const FORCES_WGSL: &str = include_str!("../../shaders/forces.wgsl");
const CONGESTION_WGSL: &str = include_str!("../../shaders/congestion.wgsl");

/// Run GPU-accelerated force-directed placement on a set of gates.
pub fn dispatch_forces(
    ctx: &GpuContext,
    gate_indices: &[usize],
    circuit: &Circuit,
    placement: &mut Placement,
    attract_weight: f32,
    repel_weight: f32,
    dt: f32,
    iterations: usize,
) {
    let n = gate_indices.len();
    if n == 0 { return; }

    // Prepare GPU gate data
    let mut gpu_gates: Vec<GpuGate> = gate_indices
        .iter()
        .map(|&g| GpuGate {
            x: placement.gate_x[g],
            y: placement.gate_y[g],
            width: circuit.gate_widths[g],
            height: circuit.gate_heights[g],
            force_x: 0.0,
            force_y: 0.0,
            _pad: [0.0; 2],
        })
        .collect();

    let gate_buf = create_storage_buffer(ctx, "gates", &gpu_gates, true);

    let params = ForceParams {
        num_gates: n as u32,
        attract_weight,
        repel_weight,
        dt,
        iteration: 0,
        _pad: [0; 3],
    };
    let params_buf = create_uniform_buffer(ctx, "force_params", &params);

    let pipeline = ctx.create_pipeline(FORCES_WGSL, "compute_forces");
    let apply_pipeline = ctx.create_pipeline(FORCES_WGSL, "apply_forces");

    let bind_group_layout = pipeline.get_bind_group_layout(0);
    let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("force_bind_group"),
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: gate_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: params_buf.as_entire_binding() },
        ],
    });

    let workgroups = (n as u32 + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE;

    for _ in 0..iterations {
        let mut encoder = ctx.device.create_command_encoder(&Default::default());

        // Compute forces
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(workgroups, 1, 1);
        }
        // Apply forces
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&apply_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(workgroups, 1, 1);
        }

        ctx.queue.submit(Some(encoder.finish()));
    }

    // Read back
    let staging = create_staging_buffer(
        ctx,
        "gate_staging",
        (n * std::mem::size_of::<GpuGate>()) as u64,
    );
    let results: Vec<GpuGate> = read_buffer(ctx, &gate_buf, &staging, n);

    for (i, &g) in gate_indices.iter().enumerate() {
        if i < results.len() {
            placement.set_position(g, results[i].x, results[i].y);
        }
    }
}

/// Run GPU-accelerated congestion computation.
pub fn dispatch_congestion(
    ctx: &GpuContext,
    grid: &mut RoutingGrid,
    placement: &Placement,
    circuit: &Circuit,
) {
    // Build wire list from nets
    let mut wires: Vec<GpuWire> = Vec::new();
    for net_gates in &circuit.net_gate_indices {
        if net_gates.len() < 2 { continue; }
        let mut min_x = f32::MAX;
        let mut max_x = f32::MIN;
        let mut min_y = f32::MAX;
        let mut max_y = f32::MIN;
        for &g in net_gates {
            min_x = min_x.min(placement.gate_x[g]);
            max_x = max_x.max(placement.gate_x[g]);
            min_y = min_y.min(placement.gate_y[g]);
            max_y = max_y.max(placement.gate_y[g]);
        }
        wires.push(GpuWire { x1: min_x, y1: min_y, x2: max_x, y2: max_y });
    }

    if wires.is_empty() { return; }

    let num_cells = grid.cols * grid.rows;
    grid.clear();

    let wire_buf = create_storage_buffer(ctx, "wires", &wires, false);
    let h_grid_buf = create_storage_buffer(ctx, "h_grid", &grid.h_congestion, true);
    let v_grid_buf = create_storage_buffer(ctx, "v_grid", &grid.v_congestion, true);

    let params = GridParams {
        grid_cols: grid.cols as u32,
        grid_rows: grid.rows as u32,
        cell_width: grid.cell_width,
        cell_height: grid.cell_height,
        num_wires: wires.len() as u32,
        _pad: [0; 3],
    };
    let params_buf = create_uniform_buffer(ctx, "grid_params", &params);

    let pipeline = ctx.create_pipeline(CONGESTION_WGSL, "compute_congestion");
    let bgl = pipeline.get_bind_group_layout(0);
    let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("congestion_bg"),
        layout: &bgl,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: wire_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: h_grid_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: v_grid_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: params_buf.as_entire_binding() },
        ],
    });

    let workgroups = (wires.len() as u32 + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE;
    let mut encoder = ctx.device.create_command_encoder(&Default::default());
    {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(workgroups, 1, 1);
    }
    ctx.queue.submit(Some(encoder.finish()));

    // Read back congestion grids
    let h_staging = create_staging_buffer(ctx, "h_staging", (num_cells * 4) as u64);
    let v_staging = create_staging_buffer(ctx, "v_staging", (num_cells * 4) as u64);

    grid.h_congestion = read_buffer(ctx, &h_grid_buf, &h_staging, num_cells);
    grid.v_congestion = read_buffer(ctx, &v_grid_buf, &v_staging, num_cells);
}
