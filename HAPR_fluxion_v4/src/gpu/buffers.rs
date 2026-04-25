//! GPU buffer management utilities.
//!
//! Helpers to upload data to GPU, create storage/uniform buffers,
//! and read results back to CPU.

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

use super::context::GpuContext;

/// Gate data packed for GPU upload.
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct GpuGate {
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
    pub force_x: f32,
    pub force_y: f32,
    pub _pad: [f32; 2], // align to 32 bytes
}

/// Force computation parameters.
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct ForceParams {
    pub num_gates: u32,
    pub attract_weight: f32,
    pub repel_weight: f32,
    pub dt: f32,
    pub iteration: u32,
    pub _pad: [u32; 3],
}

/// Congestion grid parameters.
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct GridParams {
    pub grid_cols: u32,
    pub grid_rows: u32,
    pub cell_width: f32,
    pub cell_height: f32,
    pub num_wires: u32,
    pub _pad: [u32; 3],
}

/// Wire data for congestion computation.
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct GpuWire {
    pub x1: f32,
    pub y1: f32,
    pub x2: f32,
    pub y2: f32,
}

/// Create a storage buffer from a slice of Pod data.
pub fn create_storage_buffer<T: Pod>(
    ctx: &GpuContext,
    label: &str,
    data: &[T],
    read_write: bool,
) -> wgpu::Buffer {
    let usage = if read_write {
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST
    } else {
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST
    };

    ctx.device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(label),
            contents: bytemuck::cast_slice(data),
            usage,
        })
}

/// Create a uniform buffer from a Pod struct.
pub fn create_uniform_buffer<T: Pod>(ctx: &GpuContext, label: &str, data: &T) -> wgpu::Buffer {
    ctx.device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(label),
            contents: bytemuck::bytes_of(data),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        })
}

/// Create a staging buffer for reading GPU results back to CPU.
pub fn create_staging_buffer(ctx: &GpuContext, label: &str, size: u64) -> wgpu::Buffer {
    ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

/// Read back a buffer's contents to CPU.
pub fn read_buffer<T: Pod + Clone>(
    ctx: &GpuContext,
    buffer: &wgpu::Buffer,
    staging: &wgpu::Buffer,
    count: usize,
) -> Vec<T> {
    let size = (count * std::mem::size_of::<T>()) as u64;

    let mut encoder = ctx.device.create_command_encoder(&Default::default());
    encoder.copy_buffer_to_buffer(buffer, 0, staging, 0, size);
    ctx.queue.submit(Some(encoder.finish()));

    let slice = staging.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        tx.send(result).unwrap();
    });
    ctx.device.poll(wgpu::Maintain::Wait);
    rx.recv().unwrap().unwrap();

    let data = slice.get_mapped_range();
    let result: Vec<T> = bytemuck::cast_slice(&data).to_vec();
    drop(data);
    staging.unmap();

    result
}
