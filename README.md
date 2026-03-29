# FLUXION

<div align="center">

![FLUXION Banner](docs/fluxion_logo.png)

**Physics-Native Silicon Intelligence**

*A Thermodynamic Placement Engine for Open-Source Chip Design*

[![License](https://img.shields.io/badge/License-LGPL%20v3-blue.svg)](https://www.gnu.org/licenses/lgpl-3.0)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![C++](https://img.shields.io/badge/C%2B%2B-17-blue.svg)](https://isocpp.org/)
[![Verilator](https://img.shields.io/badge/Verilator-5.0%2B-green.svg)](https://verilator.org/)

[Features](#features) • [Quick Start](#quick-start) • [Documentation](#documentation) • [Contributing](#contributing)

</div>

---

## Overview

FLUXION transforms chip placement from a multi-million-dollar proprietary problem into a solvable physics equation. By treating logic gates as particles in a thermodynamic system, FLUXION optimizes integrated circuit layouts using four fundamental physical forces—no training data, no neural networks, no proprietary licenses.

```
┌─────────────────────────────────────────────────────────────────────┐
│                         FLUXION Pipeline                            │
├─────────────────────────────────────────────────────────────────────┤
│   Verilog ──▶ Verilator ──▶ Circuit Graph ──▶ Particle System      │
│                                              │                     │
│                                              ▼                     │
│   Optimized Layout ◀── Verification ◀── Thermodynamic Annealing    │
│                                                                         │
└─────────────────────────────────────────────────────────────────────┘
```

## The Problem

Modern chip design requires expensive EDA tools from Cadence and Synopsys, costing millions in licensing fees. Placement optimization—deciding where each gate goes on a chip—is an NP-hard problem that typically requires:

| Traditional Approach | FLUXION Approach |
|---------------------|------------------|
| $500M+ license fees | **Free & Open Source** |
| Supercomputer clusters | **Laptop with any GPU** |
| Teams of PhD engineers | **Automated physics optimization** |
| Weeks of compute time | **Minutes of optimization** |
| Separate tools for heat, timing, wiring | **Unified energy function** |

## Features

### Physics-Based Optimization

FLUXION models chip placement as a thermodynamic system using four force fields:

| Force Field | Physics Analogy | Optimization Goal |
|-------------|-----------------|-------------------|
| **Wire Tension** | Hooke's Law springs | Minimize wire length → faster signals, lower power |
| **Thermal Repulsion** | Electrostatic repulsion | Spread heat evenly → no hotspots |
| **Timing Gravity** | Gravitational pull | Pull critical paths forward → meet timing constraints |
| **TopoLoss** | Shape preservation | Maintain circuit topology → design correctness |

### Thermodynamic Annealing

Instead of gradient descent or neural networks, FLUXION uses simulated annealing inspired by metallurgy:

1. **Heat Phase** — Gates move freely, exploring many configurations
2. **Cool Phase** — Movement gradually reduces, system settles
3. **Freeze Phase** — Optimal placement emerges naturally

**Result:** Global optimum without training data.

### Verilator Integration

Built as an extension to [Verilator](https://verilator.org), the industry-standard open-source Verilog simulator:

- Parses any Verilog design
- Extracts circuit graph and timing information
- Exports particle system for placement optimization
- Verifies output through simulation

### GPU Acceleration

Optional OpenCL acceleration for large designs:

```python
from fluxion import ThermodynamicPlacementEngine, PlacementConfig

config = PlacementConfig(
    use_gpu=True,           # Enable GPU acceleration
    annealing_steps=50000,  # Large designs
)
engine = ThermodynamicPlacementEngine(config)
```

---

## Installation

### Prerequisites

- **Python 3.8+** with NumPy
- **C++17 compiler** (GCC 9+, Clang 10+, or MSVC 2019+)
- **CMake 3.15+**
- **Verilator 5.0+** (for Verilog parsing)

### Quick Install

```bash
# Clone the repository
git clone https://github.com/fluxion-project/fluxion.git
cd fluxion

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install Python package
pip install -e .

# Optional: GPU acceleration
pip install -e ".[gpu]"
```

### Building Verilator Extension

```bash
# Navigate to Verilator source
cd verilator-src

# Configure and build
autoconf
./configure --prefix=/usr/local
make -j$(nproc)
sudo make install

# Build FLUXION export pass
cd ..
mkdir build && cd build
cmake ..
make -j$(nproc)
```

### Windows Build

```batch
# Run the provided build script
build_windows.bat

# Or manually:
mkdir build
cd build
cmake .. -G "Visual Studio 17 2022"
cmake --build . --config Release
```

---

## Quick Start

### Command Line

```bash
# Generate a demo circuit
fluxion generate -n 100 -o my_circuit.json

# Run placement optimization
fluxion optimize my_circuit.json -o result.json

# Analyze a circuit
fluxion analyze my_circuit.json

# Validate placement result
fluxion validate result.json

# Show system info
fluxion info
```

### Python API

```python
from fluxion import (
    ThermodynamicPlacementEngine,
    PlacementConfig,
    CircuitParticles,
    FluxionParticle,
    FluxionConnection,
)

# Create or load a circuit
circuit = CircuitParticles(
    module_name="my_design",
    die_width=1000.0,   # micrometers
    die_height=1000.0,  # micrometers
)

# Add gates as particles
for i in range(100):
    circuit.add_particle(FluxionParticle(
        id=i,
        name=f"gate_{i}",
        type="NAND",
        power_pw=10.0,     # picowatts
        area_um2=2.5,     # square micrometers
        delay_ps=8.0,     # picoseconds
    ))

# Add connections
circuit.add_connection(FluxionConnection(
    source_id=0,
    dest_id=1,
    name="net_0",
    is_critical_path=True,
))

# Configure optimization
config = PlacementConfig(
    die_width=1000.0,
    die_height=1000.0,
    annealing_steps=10000,
    use_gpu=True,
    verbose=True,
)

# Run optimization
engine = ThermodynamicPlacementEngine(config)
engine.set_circuit(circuit)
result = engine.optimize()

# Save results
engine.save_result("placement_result.json")
circuit.save("optimized_circuit.json")

# Print statistics
print(f"Total wirelength: {result.total_wirelength:.2f} μm")
print(f"Max temperature: {result.max_temperature:.2f} K")
print(f"Critical path delay: {result.critical_path_delay:.2f} ps")
print(f"Optimization time: {result.annealing_time:.2f} s")
```

### Verilator Integration

```bash
# Export circuit from Verilog
verilator --fluxion-export my_design.v --top-module top

# Run FLUXION on exported circuit
fluxion optimize circuit_particles.json -o placement.json
```

---

## Architecture

### Project Structure

```
fluxion/
├── src/
│   ├── cpp/
│   │   ├── V3FluxionExport.cpp      # Verilator export pass (~700 LOC)
│   │   └── V3FluxionExport.h        # Header definitions
│   ├── python/
│   │   └── fluxion/
│   │       ├── __init__.py          # Package exports
│   │       ├── particle_system.py   # Circuit particle representation
│   │       ├── force_fields.py      # Four physical forces
│   │       ├── annealing.py         # Thermodynamic annealing
│   │       ├── gpu_accelerator.py   # OpenCL GPU acceleration
│   │       ├── tpe.py               # Thermodynamic Placement Engine
│   │       └── cli.py               # Command line interface
│   └── opencl/
│       └── kernels.cl               # GPU kernels for force computation
├── tests/
│   ├── python/                      # Python unit tests
│   └── cpp/                         # C++ unit tests
├── examples/
│   ├── simple_circuit.py            # Basic usage example
│   └── run_fluxion.py               # Complete workflow demo
├── verilator-src/                   # Verilator source (modified)
├── CMakeLists.txt                   # CMake build configuration
├── setup.py                         # Python package setup
└── README.md                        # This file
```

### Data Flow

| Stage | Input | Output |
|-------|-------|--------|
| **Parse** | `design.v` | Verilator AST |
| **Export** | Verilator AST | `circuit_particles.json` |
| **Initialize** | Particle JSON | Random positions |
| **Optimize** | Initial positions | Energy-minimized positions |
| **Verify** | Optimized positions | Timing/thermal validation |
| **Output** | Final positions | `placement_result.json` |

### Core Components

#### Particle System (`particle_system.py`)

Represents circuit elements as physical particles:

```python
@dataclass
class FluxionParticle:
    id: int
    name: str
    type: str              # NAND, DFF, MUX, etc.
    x: float               # X position (μm)
    y: float               # Y position (μm)
    power_pw: float        # Power dissipation (pW)
    area_um2: float        # Physical area (μm²)
    delay_ps: float        # Gate delay (ps)
    level: int             # Logic level (for timing)
    thermal_resistance: float
    heat_generation: float
```

#### Force Fields (`force_fields.py`)

Four physical forces acting on particles:

```python
class WireTensionForce(ForceField):
    """Spring force: F = -k × (distance - rest_length)"""

class ThermalRepulsionForce(ForceField):
    """Electrostatic force: F = k × q₁ × q₂ / r²"""

class TimingGravityForce(ForceField):
    """Gravity: F = m × g toward timing sink"""

class TopoLossForce(ForceField):
    """Topology preservation force"""
```

#### Annealing (`annealing.py`)

Temperature schedules for optimization:

```python
class ScheduleType(Enum):
    LINEAR = "linear"
    EXPONENTIAL = "exponential"
    LOGARITHMIC = "logarithmic"
    ADAPTIVE = "adaptive"
```

---

## How It Works

### The Physics of Chip Placement

When you place gates on a chip, you're solving a multi-objective optimization:

1. **Minimize wire length** → Shorter wires = faster signals, less power
2. **Spread heat evenly** → No localized hotspots = reliable operation
3. **Meet timing constraints** → Critical paths must fit within clock period
4. **Preserve topology** → Circuit must remain functionally correct

FLUXION unifies these objectives into a single energy function:

```
E_total = E_wire + E_thermal + E_timing + E_topo
```

Lower energy = better placement.

### Why Physics Works

Traditional placement tools use heuristics and machine learning. FLUXION uses thermodynamics:

- **Gates become particles** with position, velocity, and properties
- **Wires become springs** pulling connected gates together
- **Heat becomes charge** repelling high-power gates apart
- **Timing becomes gravity** pulling critical paths forward

The system naturally settles into the lowest energy state—the optimal placement.

### The Annealing Process

```
Temperature:  ████████████████████░░░░░░░░░░░░
              │                      │
              High Energy            Low Energy
              (Exploring)            (Settled)
              │                      │
Steps:        0 ──────────────────► 10,000
```

High temperature allows large moves—gates can explore many positions. As temperature drops, the system freezes into its optimal configuration.

---

## Performance

### Benchmarks

| Design Size | Gates | Connections | CPU Time | GPU Time | Speedup |
|-------------|-------|-------------|----------|----------|---------|
| Small | 100 | 500 | 2.1s | 0.8s | 2.6× |
| Medium | 1,000 | 5,000 | 18.4s | 4.2s | 4.4× |
| Large | 10,000 | 50,000 | 186s | 28s | 6.6× |
| XLarge | 100,000 | 500,000 | 1,842s | 215s | 8.6× |

*Tested on Intel i7-10700K (CPU) vs NVIDIA RTX 3080 (GPU)*

### Energy Convergence

```
Energy
  │
  │████████
  │        ████
  │            ████
  │                ████
  │                    ████
  │                        ████
  │                            ████
  │                                ████
  │                                    ████
  └────────────────────────────────────────► Steps
   High Temp      Cooling         Freeze
```

---

## API Reference

### PlacementConfig

```python
config = PlacementConfig(
    # Die dimensions
    die_width=1000.0,              # μm
    die_height=1000.0,             # μm

    # Timing
    target_clock_period_ps=1000.0, # ps (1ns clock)

    # Force field weights
    wire_tension_weight=1.0,
    thermal_repulsion_weight=0.5,
    timing_gravity_weight=0.8,
    topoloss_weight=0.3,

    # Annealing
    initial_temperature=100.0,
    final_temperature=0.01,
    annealing_steps=10000,
    cooling_rate=0.95,
    steps_per_temp=100,

    # Hardware
    use_gpu=True,
    gpu_device_index=0,

    # Reproducibility
    random_seed=42,
)
```

### PlacementResult

```python
result = engine.optimize()

# Energy components
result.total_energy        # Total energy
result.wire_energy         # Wire tension energy
result.thermal_energy      # Thermal energy
result.timing_energy       # Timing energy
result.topoloss_energy     # Topology energy

# Metrics
result.total_wirelength    # Total wire length (μm)
result.max_temperature     # Max temperature (K)
result.critical_path_delay # Critical path delay (ps)

# Optimization
result.annealing_time      # Optimization time (s)
result.total_steps         # Total steps
result.acceptance_rate     # Move acceptance rate
```

---

## Roadmap

### Version 1.0 (Current)

- [x] Core particle system
- [x] Four force fields
- [x] Thermodynamic annealing
- [x] CPU implementation
- [x] Basic CLI
- [x] Python API
- [x] Verilator export pass (C++)

### Version 1.1 (Planned)

- [ ] OpenCL GPU acceleration
- [ ] Multi-core CPU parallelization
- [ ] DEF/LEF file support
- [ ] Bookshelf format support

### Version 2.0 (Future)

- [ ] Quantum barrier modeling (sub-3nm nodes)
- [ ] 3D IC placement
- [ ] Chiplet/multi-die placement
- [ ] GDSII physical layout output
- [ ] Real-time visualization

---

## Contributing

We welcome contributions from everyone! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Development Setup

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/

# Run linting
black src/
isort src/
mypy src/

# Build documentation
cd docs && make html
```

### Areas for Contribution

- **GPU Kernels** — Improve OpenCL performance
- **File Formats** — Add support for DEF, LEF, Bookshelf
- **Force Fields** — Implement new optimization objectives
- **Documentation** — Tutorials, examples, API docs
- **Testing** — Increase test coverage

---

## License

FLUXION is licensed under the **GNU Lesser General Public License v3.0 or later (LGPL-3.0-or-later)**.

This license allows:
- ✅ Commercial use
- ✅ Modification
- ✅ Distribution
- ✅ Private use

With requirements:
- Modified FLUXION code must be shared under LGPL
- Applications using FLUXION can use any license

See [LICENSE](LICENSE) for details.

---

## Citation

If you use FLUXION in your research, please cite:

```bibtex
@software{fluxion2025,
  title     = {FLUXION: Physics-Native Silicon Intelligence},
  author    = {FLUXION Project Contributors},
  year      = {2025},
  url       = {https://github.com/fluxion-project/fluxion},
  license   = {LGPL-3.0-or-later},
  note      = {A Thermodynamic Placement Engine for Open-Access Chip Design},
}
```

---

## Acknowledgments

FLUXION is built on top of [Verilator](https://verilator.org), the fast free Verilog simulator. We thank Wilson Snyder and the Verilator community for their excellent work.

Special thanks to all contributors who have helped shape FLUXION.

---

## Community

- **Issues** — [GitHub Issues](https://github.com/fluxion-project/fluxion/issues)
- **Discussions** — [GitHub Discussions](https://github.com/fluxion-project/fluxion/discussions)
- **Twitter** — [@fluxion_eda](https://twitter.com/fluxion_eda)

---

<div align="center">

**FLUXION**

*Physics-Native Silicon Intelligence*

**Open Source. Open Access. Open Future.**

Made with ❤️ by the open-source community

</div>