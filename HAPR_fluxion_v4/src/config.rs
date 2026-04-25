//! Configuration loading and validation.

use serde::{Deserialize, Serialize};
use std::fs::File;
use std::path::Path;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    pub input: String,
    pub output: String,
    pub format: String,
    pub partitioning: PartitioningConfig,
    pub attention: AttentionConfig,
    pub placement: PlacementConfig,
    pub congestion: CongestionConfig,
    pub legalization: LegalizationConfig,
    pub gpu: GpuConfig,
    pub logging: LoggingConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PartitioningConfig {
    pub max_leaf_size: usize,
    pub min_cluster_size: usize,
    pub balance_factor: f32,
    pub method: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttentionConfig {
    pub top_k: usize,
    pub decay: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlacementConfig {
    pub force_iterations: usize,
    pub attract_weight: f32,
    pub repel_weight: f32,
    pub damping: f32,
    pub dt: f32,
    pub embed_method: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CongestionConfig {
    pub grid_cols: usize,
    pub grid_rows: usize,
    pub overflow_threshold: f32,
    pub refinement_iterations: usize,
    pub move_distance: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LegalizationConfig {
    pub method: String,
    pub timeout_seconds: usize,
    pub region_size: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuConfig {
    pub enabled: bool,
    pub workgroup_size: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoggingConfig {
    pub level: String,
    pub progress: bool,
    pub metrics_per_level: bool,
}

impl Config {
    pub fn load(path: &Path) -> Result<Self, String> {
        let file = File::open(path).map_err(|e| format!("Failed to open config: {}", e))?;
        let config: Config = serde_yaml::from_reader(file)
            .map_err(|e| format!("Failed to parse YAML: {}", e))?;
        Ok(config)
    }
}
