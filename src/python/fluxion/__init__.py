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

from .annealing import (
    ThermodynamicAnnealing,
    TemperatureSchedule,
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
    # Annealing
    "ThermodynamicAnnealing",
    "TemperatureSchedule",
    # GPU
    "GPUAccelerator",
    "OpenCLAccelerator",
    # TPE
    "ThermodynamicPlacementEngine",
    "PlacementResult",
    "PlacementConfig",
]