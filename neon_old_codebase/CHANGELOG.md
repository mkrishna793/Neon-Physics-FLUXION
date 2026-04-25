# Changelog

All notable changes to FLUXION will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2025-03-29

### Added
- Initial release of FLUXION Thermodynamic Placement Engine
- Core particle system for circuit representation
- Four force fields: Wire Tension, Thermal Repulsion, Timing Gravity, TopoLoss
- Thermodynamic annealing scheduler with multiple temperature schedules
- Python API and CLI interface
- Verilator export pass (C++) for Verilog parsing
- GPU acceleration support via OpenCL
- Basic test suite
- Example scripts and documentation

### Features
- `ThermodynamicPlacementEngine` for placement optimization
- `CircuitParticles` for loading and saving circuit data
- `PlacementConfig` for configuration options
- Support for linear, exponential, logarithmic, and adaptive annealing schedules
- Energy and force computation for all four force fields
- Critical path identification and timing analysis
- Thermal hotspot detection

### Documentation
- Comprehensive README with examples
- Contributing guidelines
- LGPL-3.0 license

---

## Future Releases

### [1.1.0] - Planned
- OpenCL GPU kernel optimization
- Multi-core CPU parallelization
- DEF/LEF file format support
- Bookshelf format support
- Improved documentation and tutorials

### [2.0.0] - Future
- Quantum barrier modeling for sub-3nm nodes
- 3D IC placement support
- Chiplet/multi-die placement
- GDSII physical layout output
- Real-time visualization

---

[1.0.0]: https://github.com/mkrishna793/FLUXION/releases/tag/v1.0.0