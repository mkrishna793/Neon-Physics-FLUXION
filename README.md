# FLUXION

<div align="center">

**Physics-Native Silicon Intelligence**

*A Thermodynamic Placement Engine for Chip Design Research*

[![License](https://img.shields.io/badge/License-AGPL%20v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![C++](https://img.shields.io/badge/C%2B%2B-17-blue.svg)](https://isocpp.org/)
[![Status](https://img.shields.io/badge/Status-Research%20Prototype-orange.svg)]()

[Overview](#overview) • [Installation](#installation) • [Quick Start](#quick-start) • [Limitations](#limitations)

</div>

---

## Overview

FLUXION is a **research prototype** exploring physics-based approaches to integrated circuit placement. It treats logic gates as particles in a thermodynamic system and uses simulated annealing with custom force fields to optimize placement.

### What This Is

- A **research project** exploring physics-based placement optimization
- An **educational tool** for understanding placement algorithms
- A **prototype implementation** of thermodynamic annealing for circuits

### What This Is NOT

- A production-ready EDA tool
- A replacement for commercial tools (Cadence, Synopsys)
- Proven to achieve optimal or near-optimal solutions
- Benchmarked against industry standards (yet)

---

## How It Works

FLUXION models circuit placement as an energy minimization problem using four force fields:

| Force Field | Description | Optimization Goal |
|-------------|-------------|-------------------|
| **Wire Tension** | Springs between connected gates | Reduce wire length |
| **Thermal Repulsion** | Repulsion between high-power gates | Spread heat distribution |
| **Timing Gravity** | Pull critical path components | Improve timing slack |
| **TopoLoss** | Preserve circuit topology | Maintain structural integrity |

The placement is optimized using **simulated annealing**, a well-known technique in EDA since the 1980s (see [TimberWolf](https://doi.org/10.1109/TCAD.1984.1270038)).

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
config = PlacementConfig(
    annealing_steps=5000,
    verbose=True,
)

engine = ThermodynamicPlacementEngine(config)
engine.set_circuit(circuit)
result = engine.optimize()

print(f"Total wirelength: {result.total_wirelength:.2f} um")
print(f"Critical path delay: {result.critical_path_delay:.2f} ps")
```

### Command Line

```bash
# Generate and optimize a demo circuit
python examples/run_fluxion.py --num-gates 100 --steps 5000

# Output saved to fluxion_output/
```

---

## Limitations

**Important: This is a research prototype with significant limitations.**

### What's Missing

| Feature | Status |
|---------|--------|
| Clock tree synthesis | ❌ Not implemented |
| Power grid routing | ❌ Not implemented |
| Multi-corner timing | ❌ Not implemented |
| Routing congestion | ❌ Not implemented |
| Industry benchmarks | ❌ Not evaluated |
| Comparison vs OpenROAD | ❌ Not done |
| Comparison vs DREAMPlace | ❌ Not done |

### Known Issues

1. **Simplified physical model**: Real chips have constraints not modeled here (metal layers, via restrictions, design rules)
2. **No timing closure loop**: Critical path optimization is approximate
3. **No actual Verilator integration**: The C++ export pass exists but requires building Verilator from source
4. **Synthetic benchmarks only**: Tested on generated circuits, not real designs
5. **Performance**: Not optimized for large designs (>10K gates)

### Simulated Annealing Context

Simulated annealing for placement is not novel:
- TimberWolf (1984) — first major simulated annealing placer
- Commercial tools moved away from pure SA due to scalability issues
- Modern tools (OpenROAD, DREAMPlace) use analytical methods + SA refinement

This project explores whether a **physics-based formulation** can provide insights, but does not claim superiority over existing methods.

---

## Project Structure

```
fluxion/
├── src/
│   ├── python/fluxion/
│   │   ├── particle_system.py   # Circuit representation
│   │   ├── force_fields.py      # Four force fields
│   │   ├── annealing.py         # Temperature schedules
│   │   ├── tpe.py               # Main engine
│   │   └── cli.py               # Command line
│   └── cpp/
│       └── V3FluxionExport.cpp  # Verilator export pass
├── examples/
│   └── run_fluxion.py           # Demo script
├── tests/
│   └── test_fluxion.py          # Unit tests
├── README.md
├── LICENSE
└── requirements.txt
```

---

## Roadmap

### Current Status (v0.1.0)

- [x] Basic particle system
- [x] Four force fields implementation
- [x] Simulated annealing engine
- [x] Python API
- [x] Demo script

### Future Work (Contributions Welcome)

- [ ] ISPD/ICCAD benchmark evaluation
- [ ] Comparison with OpenROAD, DREAMPlace
- [ ] Real Verilator integration
- [ ] Multi-corner timing analysis
- [ ] Clock tree awareness
- [ ] Scalability improvements

---

## Related Work

This project builds on decades of research in physical design:

| Tool/Method | Year | Approach |
|-------------|------|----------|
| TimberWolf | 1984 | Simulated annealing |
| FastPlace | 2005 | Analytical quadratic |
| SimPL | 2012 | Analytical + lookahead |
| OpenROAD | 2019 | Open-source full flow |
| DREAMPlace | 2019 | Deep learning + analytical |
| RePlAce | 2020 | Analytical global placement |

For production chip design, use established tools like [OpenROAD](https://github.com/The-OpenROAD-Project/OpenROAD).

---

## Contributing

Contributions are welcome! Priority areas:

1. **Benchmarks**: Evaluate on ISPD contest datasets
2. **Comparisons**: Compare against OpenROAD placer
3. **Documentation**: Improve clarity of implementation
4. **Testing**: Add unit tests and integration tests

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

## License

FLUXION is licensed under the **GNU Affero General Public License v3.0 or later (AGPL-3.0-or-later)**.

This means:
- Free and open source
- Modifications must be shared back
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
  note      = {A Thermodynamic Placement Engine Research Prototype},
}
```

---

## Acknowledgments

- [Verilator](https://verilator.org) — Verilog simulator framework
- The physical design research community
- OpenROAD Project — for setting the standard in open-source EDA

---

<div align="center">

**FLUXION** — A Research Prototype for Physics-Based Placement

*Not production-ready. For real designs, use OpenROAD.*

</div>