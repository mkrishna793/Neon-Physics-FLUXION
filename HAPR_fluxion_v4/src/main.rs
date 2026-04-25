//! FLUXION v4 CLI Entry Point.

use clap::Parser;
use log::{error, info};
use std::path::PathBuf;

use fluxion::config::Config;
use fluxion::parser::{blif, def};
use fluxion::algorithm::{partitioner, attention, placer, refiner};

#[derive(Parser, Debug)]
#[command(author, version, about = "FLUXION v4 — HAPR Placement Engine", long_about = None)]
struct Args {
    /// Path to config YAML
    #[arg(short, long, default_value = "config/default.yaml")]
    config: PathBuf,

    /// Override input circuit file
    #[arg(short, long)]
    input: Option<PathBuf>,

    /// Override output DEF file
    #[arg(short, long)]
    output: Option<PathBuf>,
}

fn main() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
    let args = Args::parse();

    info!("Starting FLUXION v4");

    let mut config = match Config::load(&args.config) {
        Ok(c) => c,
        Err(e) => {
            error!("Config error: {}", e);
            std::process::exit(1);
        }
    };

    if let Some(in_path) = args.input {
        config.input = in_path.to_string_lossy().to_string();
    }
    if let Some(out_path) = args.output {
        config.output = out_path.to_string_lossy().to_string();
    }

    if config.input.is_empty() {
        error!("No input file specified in config or arguments");
        std::process::exit(1);
    }

    let input_path = PathBuf::from(&config.input);
    
    // 1. Parse
    info!("Parsing circuit: {}", input_path.display());
    let ext = input_path.extension().unwrap_or_default().to_str().unwrap_or("");
    
    let circuit = if ext == "blif" || config.format == "blif" {
        blif::parse(&input_path).unwrap_or_else(|e| {
            error!("BLIF parse error: {}", e);
            std::process::exit(1);
        })
    } else {
        error!("Unsupported format (only BLIF supported in this demo)");
        std::process::exit(1);
    };

    if circuit.num_gates == 0 {
        error!("Circuit has 0 gates. Exiting.");
        std::process::exit(1);
    }

    // 2. Partition
    info!("Phase 1: Hierarchical Partitioning");
    let p_config = partitioner::PartitionConfig {
        max_leaf_size: config.partitioning.max_leaf_size,
        min_cluster_size: config.partitioning.min_cluster_size,
        balance_factor: config.partitioning.balance_factor,
        power_iterations: 30,
    };
    let mut tree = partitioner::build_hierarchy(&circuit, &p_config);

    // 3. Attention
    info!("Phase 2: Attention Scoring");
    let a_config = attention::AttentionConfig {
        top_k: config.attention.top_k,
    };
    let attention_map = attention::compute_all_attention(&tree, &circuit, &a_config);

    // 4. Place
    info!("Phase 3: Multi-level Placement");
    let pl_config = placer::PlacerConfig {
        force_iterations: config.placement.force_iterations,
        attract_weight: config.placement.attract_weight,
        repel_weight: config.placement.repel_weight,
        damping: config.placement.damping,
        dt: config.placement.dt,
    };
    let mut placement = placer::hapr_placement(&circuit, &mut tree, &attention_map, &pl_config);

    // 5. Refine (Congestion-aware)
    info!("Phase 4: Congestion Refinement");
    let r_config = refiner::RefineConfig {
        max_iterations: config.congestion.refinement_iterations,
        move_distance: config.congestion.move_distance,
        overflow_threshold: config.congestion.overflow_threshold,
        grid_cols: config.congestion.grid_cols,
        grid_rows: config.congestion.grid_rows,
    };
    let _grid = refiner::refine_placement(&mut placement, &circuit, &r_config);

    // 6. Export
    info!("Phase 5: Export");
    let out_path = PathBuf::from(&config.output);
    if let Err(e) = def::write_def(&out_path, &circuit, &placement) {
        error!("Export error: {}", e);
        std::process::exit(1);
    }

    info!("Done! HPWL: {:.1}", placement.total_wirelength);
}
