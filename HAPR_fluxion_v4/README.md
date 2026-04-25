# HAPR Engine Blueprint

## What Is It
A chip placement engine. Takes a circuit file, places all gates on a chip, outputs the placement. Competes with Cadence Innovus and Synopsys ICC2.

## The Algorithm: HAPR (Hierarchical Attention Placement with Routing)
6 steps, that's the entire algorithm:

1. **PARSE**: Read circuit file (BLIF or DEF)
2. **PARTITION**: Split gates into clusters recursively until each group has ~64 gates
3. **ATTEND**: For each cluster pair, score how connected they are (0 to 1)
4. **PLACE**: Big clusters first, small clusters inside, individual gates last
5. **ROUTE**: Predict wire congestion, push gates away from crowded areas
6. **LEGALIZE**: Snap to grid using Z3 solver, no overlaps

## Three-Level Architecture

### LEVEL 1 — Python (User Interface)
CLI, config loading, visualization, Z3 legalization
Thin layer. No heavy math here.
Files: `cli.py`, `config.py`, `visualizer.py`, `legalizer.py`

### LEVEL 2 — Rust (Engine Core)
Parsing, partitioning, attention scoring, placement logic, congestion prediction, refinement loop, I/O
This is where the algorithm lives.
Files: `parser.rs`, `partitioner.rs`, `attention.rs`, `placer.rs`, `congestion.rs`, `refiner.rs`, `exporter.rs`

### LEVEL 3 — GPU (Compute)
Force computation, congestion grid counting, parallel sort
Uses WebGPU (wgpu). Runs on ANY GPU: NVIDIA, AMD, Intel, Apple.
Files: `forces.wgsl`, `congestion.wgsl`, `sort.wgsl`

## How Partitioning Works
Take all gates (e.g., 1 million)
Split into 2 groups based on connectivity (connected gates together)
Split each group into 2 again
Repeat until each group has ~64 gates

**Result:** A tree
1,000,000 gates → 500 clusters → 1000 subclusters → 16000 leaves

**Why:** Never place 1M gates at once. Place 500 clusters, then refine.

## How Attention Works
For every pair of clusters, ask: "How many wires connect us?"

Cluster A has 100 wires. 60 go to Cluster B. 20 to C. 20 to D.
- `Attention(A→B) = 0.6` → place B next to A
- `Attention(A→C) = 0.2` → C can be farther
- `Attention(A→D) = 0.2` → D can be farther

One formula. No neural network. Just counting wires.

## How Placement Works (Top to Bottom)
- **Level 0**: Place 500 clusters on chip (using attention scores)
- **Level 1**: Place subclusters inside each cluster (using attention)
- **Level 2**: Place individual gates inside each subcluster (using 2 forces)

At every level, only 2 forces exist:
- **ATTRACT** — connected things pull together
- **REPEL** — overlapping things push apart

## How Routing Awareness Works
Before finalizing placement:
1. Draw virtual wires between connected gates
2. Count wire crossings per grid cell
3. Find overcrowded cells
4. Move gates out of overcrowded cells
5. Repeat until no overcrowding

This prevents the #1 real-world placement failure: unroutable designs.

## How Legalization Works
After placement, gate positions are floating point (x=100.3, y=55.7). Real chips need integer grid positions with no overlaps.

**Z3 solver:**
- **Input:** floating point positions
- **Constraints:** integer grid, no overlaps, within boundaries, row-aligned
- **Objective:** minimize movement from original positions
- **Output:** legal placement

## File Structure
```text
fluxion-v4/
├── src/                          # Rust core
│   ├── parser/                   # Read BLIF, DEF, Bookshelf
│   ├── algorithm/
│   │   ├── partitioner.rs        # Build hierarchy tree
│   │   ├── attention.rs          # Score cluster connections
│   │   ├── placer.rs             # Multi-level placement
│   │   ├── congestion.rs         # Routing prediction
│   │   └── refiner.rs            # Congestion-aware refinement
│   ├── gpu/
│   │   ├── context.rs            # WebGPU init (any GPU)
│   │   ├── buffers.rs            # GPU memory management
│   │   └── dispatch.rs           # Run shaders
│   ├── data/
│   │   ├── circuit.rs            # Gate, Net, Pin structs
│   │   ├── hierarchy.rs          # Cluster tree
│   │   ├── placement.rs          # Result struct
│   │   └── grid.rs               # Routing grid
│   ├── config.rs                 # Load YAML config
│   ├── lib.rs                    # Public API
│   └── main.rs                   # CLI entry
│
├── shaders/                      # GPU shaders (WGSL)
│   ├── forces.wgsl               # Attract + Repel
│   ├── congestion.wgsl           # Wire crossing count
│   └── sort.wgsl                 # Parallel sort
│
├── python/                       # Python wrapper
│   ├── fluxion/
│   │   ├── engine.py             # API (calls Rust via PyO3)
│   │   ├── legalizer.py          # Z3 legalization
│   │   └── visualizer.py         # Draw placement
│   └── cli.py                    # Command line interface
│
├── config/
│   └── default.yaml              # All parameters
│
├── Cargo.toml
├── pyproject.toml
└── README.md
```

## Cloud GPU Execution (Kaggle / Colab)
Because FLUXION v4 uses WebGPU (`wgpu`), it runs natively on cloud Linux instances via Vulkan without any CUDA setup.

You can easily run benchmarks and tests on a **Kaggle** or **Google Colab** notebook.
1. Upload the repository or `notebooks/kaggle_quickstart.ipynb` to Kaggle/Colab.
2. Run the cells to automatically install Rust and compile the engine.

### Running Benchmarks
To run the engine against official ISPD/ICCAD benchmarks, use the included Python script:
```bash
python benchmarks/run_benchmarks.py --circuit benchmarks/ispd2005/adaptec1.def
```

### Running the Test Suite
To verify the engine end-to-end:
```bash
# Rust Unit Tests
cargo test

# Python Integration Tests
pytest tests/test_integration.py
```

