# FLUXION

<div align="center">

**Physics-Native Silicon Intelligence**

*A Thermodynamic Placement Engine for Chip Design*

[![License](https://img.shields.io/badge/License-AGPL%20v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![C++](https://img.shields.io/badge/C%2B%2B-17-blue.svg)](https://isocpp.org/)
[![Status](https://img.shields.io/badge/Status-Research%20Prototype-orange.svg)]()

[Overview](#overview) • [Installation](#installation) • [Quick Start](#quick-start) • [Documentation](#documentation)

</div>

---

## The Core Idea

**FLUXION explores a new question:**

> Can four unified physics force fields — wire tension, thermal repulsion, timing gravity, and topology preservation — solve chip placement without training data?

Instead of machine learning or heuristics, FLUXION models the circuit as a physical system:

| Force Field | Physics Analogy | What It Optimizes |
|-------------|-----------------|-------------------|
| **Wire Tension** | Hooke's Law springs | Minimize wire length → faster signals |
| **Thermal Repulsion** | Electrostatic repulsion | Spread heat evenly → no hotspots |
| **Timing Gravity** | Gravitational pull | Pull critical paths → meet timing |
| **TopoLoss** | Shape preservation | Maintain circuit correctness |

All four forces operate simultaneously, unified in a single energy function. The system finds equilibrium through thermodynamic annealing — no training data, no neural networks.

---

## Overview

FLUXION is a **research prototype** for physics-based integrated circuit placement. It takes a Verilog design and optimizes gate positions using thermodynamic principles.

```
┌─────────────────────────────────────────────────────────────────────┐
│                         FLUXION Pipeline                            │
├─────────────────────────────────────────────────────────────────────┤
│   Verilog ──▶ Verilator ──▶ Circuit Graph ──▶ Particle System      │
│                                              │                     │
│                                              ▼                     │
│   Optimized Layout ◀── Verification ◀── Thermodynamic Annealing    │
└─────────────────────────────────────────────────────────────────────┘
```

### What Makes This Different

| Traditional Approaches | FLUXION Approach |
|-----------------------|------------------|
| Analytical solvers | Unified physics simulation |
| Machine learning (needs training data) | No training data required |
| Separate objectives (wire, timing, thermal) | Single energy function |
| Proprietary or expensive | Open source (AGPL-3.0) |

---

## Installation

### Prerequisites

- Python 3.8+
- NumPy
- (Optional) OpenCL for GPU acceleration

### Quick Install

```bash
# Clone the repository
git clone https://github.com/mkrishna793/Neon-Physics-FLUXION.git
cd Neon-Physics-FLUXION

# Install dependencies
pip install -r requirements.txt

# Install package
pip install -e .
```

---

## Quick Start

### Python API

```python
from fluxion import (
    ThermodynamicPlacementEngine,
    PlacementConfig,
    CircuitParticles,
    FluxionParticle,
    FluxionConnection,
)

# Create a circuit
circuit = CircuitParticles(
    module_name="example",
    die_width=100.0,
    die_height=100.0,
)

# Add gates
for i in range(50):
    circuit.add_particle(FluxionParticle(
        id=i,
        name=f"gate_{i}",
        type="NAND",
        power_pw=10.0,
        area_um2=2.5,
        delay_ps=8.0,
    ))

# Configure and run
config = PlacementConfig(annealing_steps=5000, verbose=True)

engine = ThermodynamicPlacementEngine(config)
engine.set_circuit(circuit)
result = engine.optimize()

print(f"Total wirelength: {result.total_wirelength:.2f} um")
print(f"Critical path delay: {result.critical_path_delay:.2f} ps")
```

### Command Line

```bash
python examples/run_fluxion.py --num-gates 100 --steps 5000
```

---

## Architecture

### Project Structure

```
fluxion/
├── src/
│   ├── python/fluxion/
│   │   ├── particle_system.py   # Circuit as particles
│   │   ├── force_fields.py      # Four physics forces
│   │   ├── annealing.py        # Thermodynamic annealing
│   │   ├── tpe.py              # Main placement engine
│   │   └── cli.py              # Command line interface
│   └── cpp/
│       └── V3FluxionExport.cpp  # Verilator export pass
├── examples/
│   └── run_fluxion.py           # Demo script
├── tests/
│   └── test_fluxion.py          # Unit tests
└── README.md
```

### Core Components

#### Force Fields (`force_fields.py`)

Each force field calculates energy and gradients:

```python
class WireTensionForce(ForceField):
    """Spring force: F = -k × (distance - rest_length)"""

class ThermalRepulsionForce(ForceField):
    """Electrostatic: F = k × q₁ × q₂ / r²"""

class TimingGravityForce(ForceField):
    """Gravity: F = m × g toward timing sink"""

class TopoLossForce(ForceField):
    """Topology preservation force"""
```

#### Thermodynamic Annealing (`annealing.py`)

Temperature schedules for optimization:

```python
class ScheduleType(Enum):
    LINEAR = "linear"
    EXPONENTIAL = "exponential"
    LOGARITHMIC = "logarithmic"
    ADAPTIVE = "adaptive"
```

---

## Documentation

### PlacementConfig

```python
config = PlacementConfig(
    # Die dimensions (micrometers)
    die_width=1000.0,
    die_height=1000.0,

    # Timing target (picoseconds)
    target_clock_period_ps=1000.0,

    # Force field weights
    wire_tension_weight=1.0,
    thermal_repulsion_weight=0.5,
    timing_gravity_weight=0.8,
    topoloss_weight=0.3,

    # Annealing
    initial_temperature=100.0,
    final_temperature=0.01,
    annealing_steps=10000,

    # Hardware
    use_gpu=True,
)
```

### PlacementResult

```python
result = engine.optimize()

# Energy components
result.total_energy        # Total energy
result.wire_energy        # Wire tension energy
result.thermal_energy     # Thermal energy
result.timing_energy      # Timing energy
result.topoloss_energy    # Topology energy

# Metrics
result.total_wirelength   # Total wire length (μm)
result.max_temperature    # Max temperature (K)
result.critical_path_delay # Critical path (ps)

# Optimization info
result.annealing_time     # Time (seconds)
result.acceptance_rate    # Move acceptance rate
```

---

## Current Status

### What Works

- Core particle system for circuit representation
- Four unified force fields
- Thermodynamic annealing with multiple schedules
- Python API and CLI interface
- Verilator export pass (C++)

### Development Roadmap

| Feature | Status |
|---------|--------|
| Core placement engine | ✅ Working |
| Python API | ✅ Working |
| Four force fields | ✅ Working |
| GPU acceleration (OpenCL) | 🔧 In progress |
| Industry benchmarks (ISPD) | 📋 Planned |
| Comparison vs OpenROAD | 📋 Planned |
| Multi-corner timing | 📋 Planned |

---

## Related Work

Simulated annealing has been used in EDA since TimberWolf (1984). FLUXION builds on this foundation with a unified four-field physics model.

| Tool | Approach | Notes |
|------|----------|-------|
| TimberWolf | Simulated annealing | Classic SA placer |
| OpenROAD | Analytical + SA | Full open-source flow |
| DREAMPlace | Deep learning + analytical | GPU-accelerated |
| RePlAce | Analytical global placement | Electrostatic formulation |
| **FLUXION** | Four-field physics + SA | No training data |

---

## Contributing

Contributions welcome! Priority areas:

1. **Benchmarks**: Run on ISPD/ICCAD datasets
2. **Comparisons**: Compare against existing tools
3. **GPU kernels**: Improve OpenCL performance
4. **Documentation**: Tutorials and examples

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

## License

FLUXION is licensed under the **GNU Affero General Public License v3.0 or later (AGPL-3.0-or-later)**.

This ensures:
- Free and open source forever
- Modifications must be shared back to the community
- Network use (SaaS) requires providing source code

See [LICENSE](LICENSE) for details.

---

## Citation

If you use FLUXION in your research, please cite:

```bibtex
@software{fluxion2025,
  title     = {FLUXION: Physics-Native Silicon Intelligence},
  author    = {Krishna, M.},
  year      = {2025},
  url       = {https://github.com/mkrishna793/Neon-Physics-FLUXION},
  license   = {AGPL-3.0-or-later},
  note      = {A Thermodynamic Placement Engine},
}
```

---

## Acknowledgments

- [Verilator](https://verilator.org) — Verilog simulator framework
- The physical design research community
- Open-source EDA contributors worldwide

---

<div align="center">

**FLUXION**

*Four unified force fields. No training data. Physics-first placement.*

[GitHub](https://github.com/mkrishna793/Neon-Physics-FLUXION) • [Issues](https://github.com/mkrishna793/Neon-Physics-FLUXION/issues) • [Discussions](https://github.com/mkrishna793/Neon-Physics-FLUXION/discussions)

</div>