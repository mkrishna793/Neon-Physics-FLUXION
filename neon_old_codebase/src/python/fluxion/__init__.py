"""
FLUXION Physics-Native Silicon Intelligence
A Thermodynamic Placement Engine for Open-Access Chip Design

This package provides the core components for FLUXION:
- Particle system for circuit representation
- Force fields for physical placement optimization
- Thermodynamic annealing for global optimization
- GPU acceleration via OpenCL
- Verification loop through Verilator

Version: 1.0.0
"""

__version__ = "1.0.0"
__author__ = "FLUXION Project"

from .particle_system import (
    FluxionParticle,
    FluxionConnection,
    CriticalPath,
    FluxionParticleSystem,
    CircuitParticles,
    load_circuit_particles,
)

from .force_fields import (
    ForceField,
    WireTensionForce,
    ThermalRepulsionForce,
    TimingGravityForce,
    TopoLossForce,
    CompositeForceField,
)

from .force_density import DensityEqualizationForce
from .force_electrostatic import ElectrostaticSmoothingForce

from .barnes_hut import BarnesHutTree
from .spatial_hash import SpatialHashGrid

from .def_exporter import DEFExporter, export_def
from .lef_library import LEFLibrary
from .grid import PlacementGrid
from .legalizer import HybridLegalizer, TetrisLegalizer, Z3HotspotSolver

from .annealing import (
    ThermodynamicAnnealing,
    TemperatureSchedule,
    ScheduleType,
)

from .gpu_accelerator import (
    GPUAccelerator,
    OpenCLAccelerator,
)

from .tpe import (
    ThermodynamicPlacementEngine,
    PlacementResult,
    PlacementConfig,
)

__all__ = [
    # Particle system
    "FluxionParticle",
    "FluxionConnection",
    "CriticalPath",
    "FluxionParticleSystem",
    "CircuitParticles",
    "load_circuit_particles",
    # Force fields
    "ForceField",
    "WireTensionForce",
    "ThermalRepulsionForce",
    "TimingGravityForce",
    "TopoLossForce",
    "CompositeForceField",
    "DensityEqualizationForce",
    "ElectrostaticSmoothingForce",
    "BarnesHutTree",
    "SpatialHashGrid",
    "DEFExporter",
    "export_def",
    "LEFLibrary",
    "PlacementGrid",
    "HybridLegalizer",
    "TetrisLegalizer",
    "Z3HotspotSolver",
    # Annealing
    "ThermodynamicAnnealing",
    "TemperatureSchedule",
    "ScheduleType",
    # GPU
    "GPUAccelerator",
    "OpenCLAccelerator",
    # TPE
    "ThermodynamicPlacementEngine",
    "PlacementResult",
    "PlacementConfig",
]