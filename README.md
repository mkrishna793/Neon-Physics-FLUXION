# FLUXION v4: HAPR Engine

![FLUXION](https://img.shields.io/badge/FLUXION-v4-blue) ![HAPR](https://img.shields.io/badge/Algorithm-HAPR-orange) ![WebGPU](https://img.shields.io/badge/Compute-WebGPU-brightgreen)

A massive-scale, high-performance chip placement engine. FLUXION takes a raw circuit file, calculates the optimal physical location for every logic gate on a silicon chip to minimize wire length and congestion, and outputs the final placement coordinates.    you can open the main github of this HAPR Fluxion :- https://github.com/mkrishna793/Neon-FLUXION.git

It is built to compete with commercial EDA physical synthesis tools (like Cadence Innovus and Synopsys ICC2) through modern algorithms and hardware-agnostic GPU acceleration.

---

## 🧠 The Algorithm: HAPR

**HAPR (Hierarchical Attention Placement with Routing)** is a novel physical synthesis algorithm designed to place millions of gates without falling into local minima or taking days to compute.

It operates strictly in 6 sequential steps:

### 1. PARSE
The engine ingests standard EDA circuit files (like BLIF or DEF). It constructs a highly optimized, flat-array structure in memory where gates, pins, and nets are represented as SIMD-friendly indices.

### 2. PARTITION (The "Camera Zoom")
You cannot place 1 million gates simultaneously. HAPR solves this by recursively slicing the circuit graph into smaller, highly-connected clusters.
* **Mechanism**: It constructs a graph Laplacian matrix and uses Power Iteration to find the Fiedler Vector. Splitting on the Fiedler value perfectly bisects the graph while cutting the minimum number of wires.
* **Result**: A tree hierarchy where 1,000,000 gates → 500 clusters → 10,000 subclusters → 64-gate leaves.

### 3. ATTEND (The Attention Mechanism)
Instead of a neural network, HAPR uses a deterministic graph-attention formula to establish spatial relationships between clusters.
* **Formula**: `Attention(A→B) = shared_wires / total_wires`.
* If Cluster A shares 60% of its wires with Cluster B, it has an attention score of `0.6`. This score acts as a rigid attractive spring during physical placement, guaranteeing that heavily connected logic is kept physically close.

### 4. PLACE (Top-Down Physical Embedding)
Placement occurs hierarchically:
* **Level 0 (Macro)**: The top 500 clusters are placed onto the chip bounds using Spectral Embedding based on their attention scores.
* **Level N (Micro)**: Individual gates within the smallest clusters are placed using a **Force-Directed** algorithm. 
  * *ATTRACT*: Connected gates pull each other together.
  * *REPEL*: Overlapping gates push each other apart.

### 5. ROUTE (Congestion Prediction)
Before finalizing the layout, HAPR predicts where wires will actually be routed.
* **Mechanism**: It projects virtual wires between connected gates onto a 2D routing grid, counting bounding-box crossings per cell. 
* **Refinement**: Overcrowded cells (hotspots) are identified, and gates inside them are pushed outward toward cooler neighbor cells. This prevents the #1 failure mode in chip design: unroutable congestion.

### 6. LEGALIZE
Floating-point coordinates (`x=100.34`, `y=55.71`) are useless to a silicon foundry. 
* **Mechanism**: A Z3 solver or bipartite matching algorithm snaps the floating-point gates to precise integer coordinates on the standard cell rows, ensuring absolutely zero overlap while minimizing displacement from the ideal HAPR layout.

---

## 🏗️ Three-Level Architecture

FLUXION v4 is split into three strict layers to isolate complexity.

### LEVEL 1 — Python (User Interface)
* **Purpose**: CLI, Configuration, Legalization, Visualization.
* **Why**: Python is excellent for interacting with users, parsing YAML configs, and interfacing with tools like the Z3 solver or Matplotlib. No heavy math happens here.
* **Files**: `engine.py`, `legalizer.py`, `cli.py`

### LEVEL 2 — Rust (Engine Core)
* **Purpose**: The HAPR algorithm lives here. Parsing, partitioning, attention scoring, and orchestration.
* **Why**: Rust provides C-level performance with total memory safety. By avoiding Python's GIL, Rust can partition a 1M gate circuit across 32 CPU cores instantly.
* **Files**: `partitioner.rs`, `attention.rs`, `placer.rs`, `congestion.rs`

### LEVEL 3 — GPU (Compute)
* **Purpose**: Massive parallel math (Forces, Congestion Grid Counting, Bitonic Sorts).
* **Why**: Uses `wgpu` (WebGPU standard). This allows FLUXION to run on **ANY GPU** without CUDA. It compiles down to Vulkan (NVIDIA/AMD/Linux), DX12 (Windows), or Metal (Apple M1/M2/M3).
* **Files**: `forces.wgsl`, `congestion.wgsl`, `sort.wgsl`

---

## 🚀 Quick Start & Testing

You do not need a complex cloud environment to test the engine. 

### 1. Prerequisites
Ensure you have the Rust compiler installed on your machine.
* Install via: `curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.js | sh`
* Ensure you have C++ build tools installed on your OS (MSVC for Windows, GCC for Linux).

### 2. Run the Verification Script
We provide a simple Python script to automatically compile the engine, run the unit tests, and execute a full end-to-end placement on a test circuit.

```bash
# From the project root
python run_tests.py
```

### 3. Run Benchmarks manually
To run the engine against a specific ISPD/ICCAD benchmark file:

```bash
python benchmarks/run_benchmarks.py --circuit tests/fixtures/small.blif
```
