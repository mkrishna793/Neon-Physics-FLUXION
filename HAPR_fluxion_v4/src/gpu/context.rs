//! WebGPU context initialization.
//!
//! Creates a wgpu device and queue that works on ANY GPU:
//! NVIDIA (Vulkan), AMD (Vulkan), Intel (Vulkan/DX12), Apple (Metal).

use log::{info, warn};

/// GPU compute context wrapping wgpu device and queue.
pub struct GpuContext {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub adapter_name: String,
}

impl GpuContext {
    /// Try to create a GPU context. Returns None if no GPU available.
    pub fn new() -> Option<Self> {
        pollster::block_on(Self::new_async())
    }

    async fn new_async() -> Option<Self> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await;

        let adapter = match adapter {
            Some(a) => a,
            None => {
                warn!("No GPU adapter found. Using CPU fallback.");
                return None;
            }
        };

        let adapter_name = adapter.get_info().name.clone();
        info!("GPU adapter: {}", adapter_name);

        let (device, queue) = match adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("FLUXION GPU"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::default(),
                    memory_hints: wgpu::MemoryHints::Performance,
                },
                None,
            )
            .await
        {
            Ok(dq) => dq,
            Err(e) => {
                warn!("GPU device request failed: {}. Using CPU fallback.", e);
                return None;
            }
        };

        Some(Self {
            device,
            queue,
            adapter_name,
        })
    }

    /// Create a compute pipeline from WGSL shader source.
    pub fn create_pipeline(&self, shader_src: &str, entry_point: &str) -> wgpu::ComputePipeline {
        let shader_module = self.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(entry_point),
            source: wgpu::ShaderSource::Wgsl(shader_src.into()),
        });

        self.device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry_point),
                layout: None,
                module: &shader_module,
                entry_point: Some(entry_point),
                compilation_options: Default::default(),
                cache: None,
            })
    }
}
