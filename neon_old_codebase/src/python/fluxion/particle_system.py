"""
FLUXION Particle System

Represents a digital circuit as a system of physical particles,
where each gate/module is a particle and wires create force
connections between particles.

This module provides:
- FluxionParticle: A single gate/module as a physical particle
- FluxionParticleSystem: Collection of particles with interactions
- CircuitParticles: Circuit-specific particle representation
"""

import json
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set
from pathlib import Path


@dataclass
class FluxionParticle:
    """
    A single gate/module represented as a physical particle.

    Each particle has:
    - Position (x, y) on the die
    - Physical properties (power, area, delay)
    - Connectivity (inputs, outputs)
    - Thermal properties
    """

    id: int
    name: str
    type: str
    gate_count: int = 1
    power_pw: float = 0.0  # Power in picowatts
    area_um2: float = 0.0  # Area in square micrometers
    x: float = 0.0  # X position in micrometers
    y: float = 0.0  # Y position in micrometers
    delay_ps: float = 0.0  # Gate delay in picoseconds
    level: int = 0  # Logic level from primary inputs
    thermal_resistance: float = 1.0  # Thermal resistance in K/W
    heat_generation: float = 0.0  # Heat generation in Watts
    inputs: List[int] = field(default_factory=list)  # IDs of input particles
    outputs: List[int] = field(default_factory=list)  # IDs of output particles

    @property
    def mass(self) -> float:
        """
        Particle mass based on gate count.
        Used in force calculations for inertia.
        """
        return self.gate_count * 1.0

    @property
    def radius(self) -> float:
        """
        Particle radius based on area.
        Used for collision detection and visualization.
        """
        return np.sqrt(self.area_um2 / np.pi) if self.area_um2 > 0 else 1.0

    def distance_to(self, other: 'FluxionParticle') -> float:
        """Calculate Euclidean distance to another particle."""
        return np.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)

    def kinetic_energy(self, vx: float, vy: float) -> float:
        """Calculate kinetic energy given velocity components."""
        return 0.5 * self.mass * (vx**2 + vy**2)

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            'id': self.id,
            'name': self.name,
            'type': self.type,
            'gate_count': self.gate_count,
            'power_pw': self.power_pw,
            'area_um2': self.area_um2,
            'x': self.x,
            'y': self.y,
            'delay_ps': self.delay_ps,
            'level': self.level,
            'thermal_resistance': self.thermal_resistance,
            'heat_generation': self.heat_generation,
            'inputs': self.inputs,
            'outputs': self.outputs,
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'FluxionParticle':
        """Create from dictionary."""
        return cls(
            id=data['id'],
            name=data['name'],
            type=data['type'],
            gate_count=data.get('gate_count', 1),
            power_pw=data.get('power_pw', 0.0),
            area_um2=data.get('area_um2', 0.0),
            x=data.get('x', 0.0),
            y=data.get('y', 0.0),
            delay_ps=data.get('delay_ps', 0.0),
            level=data.get('level', 0),
            thermal_resistance=data.get('thermal_resistance', 1.0),
            heat_generation=data.get('heat_generation', 0.0),
            inputs=data.get('inputs', []),
            outputs=data.get('outputs', []),
        )


@dataclass
class FluxionConnection:
    """
    A connection between two particles (wire).

    Represents a signal wire from one gate output to another gate input.
    """
    source_id: int
    dest_id: int
    name: str = ""
    bit_width: int = 1
    is_critical_path: bool = False
    estimated_length: float = 0.0  # Will be calculated during placement
    capacitance: float = 0.0  # Wire capacitance in femtofarads

    def to_dict(self) -> dict:
        return {
            'source_id': self.source_id,
            'dest_id': self.dest_id,
            'name': self.name,
            'bit_width': self.bit_width,
            'is_critical_path': self.is_critical_path,
            'estimated_length': self.estimated_length,
            'capacitance': self.capacitance,
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'FluxionConnection':
        return cls(
            source_id=data['source_id'],
            dest_id=data['dest_id'],
            name=data.get('name', ''),
            bit_width=data.get('bit_width', 1),
            is_critical_path=data.get('is_critical_path', False),
            estimated_length=data.get('estimated_length', 0.0),
            capacitance=data.get('capacitance', 0.0),
        )


@dataclass
class CriticalPath:
    """
    A timing-critical path through the circuit.
    """
    node_ids: List[int]
    total_delay_ps: float
    slack_ps: float = 0.0
    start_clock: str = ""
    end_clock: str = ""

    def to_dict(self) -> dict:
        return {
            'node_ids': self.node_ids,
            'total_delay_ps': self.total_delay_ps,
            'slack_ps': self.slack_ps,
            'start_clock': self.start_clock,
            'end_clock': self.end_clock,
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'CriticalPath':
        return cls(
            node_ids=data['node_ids'],
            total_delay_ps=data['total_delay_ps'],
            slack_ps=data.get('slack_ps', 0.0),
            start_clock=data.get('start_clock', ''),
            end_clock=data.get('end_clock', ''),
        )


class FluxionParticleSystem:
    """
    A system of particles representing a circuit.

    This is the core data structure for FLUXION's physics-based placement.
    It manages:
    - Particle positions and velocities
    - Connectivity graph
    - Force calculations
    - Optimization state
    """

    def __init__(self, die_width: float = 1000.0, die_height: float = 1000.0):
        """
        Initialize particle system.

        Args:
            die_width: Width of the die in micrometers
            die_height: Height of the die in micrometers
        """
        self.die_width = die_width
        self.die_height = die_height

        self.particles: Dict[int, FluxionParticle] = {}
        self.connections: List[FluxionConnection] = []
        self.critical_paths: List[CriticalPath] = []

        # Optimization state
        self.velocities: Dict[int, Tuple[float, float]] = {}
        self.forces: Dict[int, Tuple[float, float]] = {}

        # Physical constraints
        self.target_clock_period_ps: float = 1000.0  # 1ns default
        self.target_temperature_K: float = 350.0  # 77°C max junction temp

    def add_particle(self, particle: FluxionParticle) -> None:
        """Add a particle to the system."""
        self.particles[particle.id] = particle
        self.velocities[particle.id] = (0.0, 0.0)
        self.forces[particle.id] = (0.0, 0.0)

    def add_connection(self, connection: FluxionConnection) -> None:
        """Add a connection between particles."""
        self.connections.append(connection)

    def add_critical_path(self, path: CriticalPath) -> None:
        """Add a critical timing path."""
        self.critical_paths.append(path)

    def get_positions(self) -> np.ndarray:
        """Get all particle positions as Nx2 array."""
        positions = np.zeros((len(self.particles), 2))
        for i, (pid, particle) in enumerate(self.particles.items()):
            positions[i] = [particle.x, particle.y]
        return positions

    def set_positions(self, positions: np.ndarray) -> None:
        """Set all particle positions from Nx2 array."""
        for i, (pid, particle) in enumerate(self.particles.items()):
            particle.x = positions[i, 0]
            particle.y = positions[i, 1]

    def get_velocities(self) -> np.ndarray:
        """Get all particle velocities as Nx2 array."""
        velocities = np.zeros((len(self.particles), 2))
        for i, pid in enumerate(self.particles.keys()):
            vx, vy = self.velocities.get(pid, (0.0, 0.0))
            velocities[i] = [vx, vy]
        return velocities

    def set_velocities(self, velocities: np.ndarray) -> None:
        """Set all particle velocities from Nx2 array."""
        for i, pid in enumerate(self.particles.keys()):
            self.velocities[pid] = (velocities[i, 0], velocities[i, 1])

    def get_connection_matrix(self) -> np.ndarray:
        """
        Get connectivity as sparse adjacency matrix.

        Returns:
            NxN adjacency matrix where entry (i,j) is the number of
            connections between particles i and j.
        """
        n = len(self.particles)
        id_to_idx = {pid: idx for idx, pid in enumerate(self.particles.keys())}

        matrix = np.zeros((n, n), dtype=np.float64)

        for conn in self.connections:
            if conn.source_id in id_to_idx and conn.dest_id in id_to_idx:
                i = id_to_idx[conn.source_id]
                j = id_to_idx[conn.dest_id]
                # Weight by criticality
                weight = 2.0 if conn.is_critical_path else 1.0
                matrix[i, j] += weight
                matrix[j, i] += weight  # Symmetric for undirected force

        return matrix

    def total_wirelength(self) -> float:
        """Calculate total wire length (HPWL - Half Perimeter Wire Length)."""
        total = 0.0
        id_to_particle = self.particles

        for conn in self.connections:
            if conn.source_id in id_to_particle and conn.dest_id in id_to_particle:
                src = id_to_particle[conn.source_id]
                dst = id_to_particle[conn.dest_id]
                length = abs(src.x - dst.x) + abs(src.y - dst.y)  # HPWL
                total += length

        return total

    def max_temperature(self) -> float:
        """Estimate maximum junction temperature."""
        # Simplified thermal model
        # In practice, this would use thermal simulation
        temperatures = []

        for particle in self.particles.values():
            # Heat dissipation increases with local density
            local_density = self._local_density(particle)
            temp = self.target_temperature_K * (1.0 + 0.1 * local_density)
            temp += particle.heat_generation * particle.thermal_resistance * 0.001
            temperatures.append(temp)

        return max(temperatures) if temperatures else self.target_temperature_K

    def _local_density(self, particle: FluxionParticle, radius: float = 50.0) -> float:
        """Calculate local particle density around a particle."""
        count = 0
        for other in self.particles.values():
            if other.id != particle.id:
                dist = particle.distance_to(other)
                if dist < radius:
                    count += 1
        return count / (np.pi * radius**2 / 10000)  # Normalized density

    def update_timing(self, wire_delay_per_um: float = 0.5) -> None:
        """Update critical paths and compute delays based on physical geometry."""
        crit_conns = [c for c in self.connections if c.is_critical_path]
        if not crit_conns:
            return
            
        # Build adjacency for critical subgraph
        adj = {}
        for c in crit_conns:
            if c.source_id not in adj: adj[c.source_id] = []
            adj[c.source_id].append(c.dest_id)
            
        # Find roots (nodes with no incoming critical edges)
        has_incoming = {c.dest_id for c in crit_conns}
        roots = [c.source_id for c in crit_conns if c.source_id not in has_incoming]
        
        # Fallback if cyclic
        if not roots:
            roots = list(adj.keys())
            
        self.critical_paths = []
        
        from typing import List
        def dfs(node_id: int, current_path: List[int], current_delay: float):
            if node_id not in self.particles:
                return
            particle = self.particles[node_id]
            node_delay = particle.delay_ps
            
            if node_id not in adj or not adj[node_id]:
                self.critical_paths.append(CriticalPath(
                    node_ids=current_path.copy() + [node_id],
                    total_delay_ps=current_delay + node_delay
                ))
                return
                
            for next_id in adj[node_id]:
                if next_id not in self.particles or next_id in current_path:
                    continue  # Prevent cycles
                next_particle = self.particles[next_id]
                dist = particle.distance_to(next_particle)
                wire_delay = dist * wire_delay_per_um
                dfs(next_id, current_path + [node_id], current_delay + node_delay + wire_delay)
                
        for root in roots:
            dfs(root, [], 0.0)

    def critical_path_delay(self) -> float:
        """Calculate delay along critical paths."""
        if not self.critical_paths:
            return 0.0
        return max(path.total_delay_ps for path in self.critical_paths)

    def randomize_positions(self, seed: int = None) -> None:
        """Randomize particle positions within die boundaries."""
        if seed is not None:
            np.random.seed(seed)

        for particle in self.particles.values():
            particle.x = np.random.uniform(0, self.die_width)
            particle.y = np.random.uniform(0, self.die_height)

    def enforce_boundaries(self) -> None:
        """Ensure all particles are within die boundaries."""
        for particle in self.particles.values():
            particle.x = np.clip(particle.x, 0, self.die_width)
            particle.y = np.clip(particle.y, 0, self.die_height)

    def get_particle_by_name(self, name: str) -> Optional[FluxionParticle]:
        """Get particle by name."""
        for particle in self.particles.values():
            if particle.name == name:
                return particle
        return None

    def get_neighbors(self, particle_id: int) -> Set[int]:
        """Get IDs of all particles connected to this one."""
        neighbors = set()
        for conn in self.connections:
            if conn.source_id == particle_id:
                neighbors.add(conn.dest_id)
            elif conn.dest_id == particle_id:
                neighbors.add(conn.source_id)
        return neighbors

    def compute_logic_levels(self) -> None:
        """Compute logic level for each particle (distance from primary inputs)."""
        # Start with particles that have no inputs (primary inputs)
        levels = {}

        # Initialize all levels to -1
        for pid in self.particles:
            levels[pid] = -1

        # Primary inputs have level 0
        for particle in self.particles.values():
            if not particle.inputs:
                levels[particle.id] = 0

        # Propagate levels using BFS
        changed = True
        while changed:
            changed = False
            for particle in self.particles.values():
                if levels[particle.id] == -1 and particle.inputs:
                    # Check if all inputs have levels assigned
                    input_levels = [levels[i] for i in particle.inputs if levels[i] >= 0]
                    if len(input_levels) == len(particle.inputs):
                        levels[particle.id] = max(input_levels) + 1
                        particle.level = levels[particle.id]
                        changed = True

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            'die_width': self.die_width,
            'die_height': self.die_height,
            'target_clock_period_ps': self.target_clock_period_ps,
            'target_temperature_K': self.target_temperature_K,
            'particles': [p.to_dict() for p in self.particles.values()],
            'connections': [c.to_dict() for c in self.connections],
            'critical_paths': [p.to_dict() for p in self.critical_paths],
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'FluxionParticleSystem':
        """Create from dictionary."""
        system = cls(
            die_width=data.get('die_width', 1000.0),
            die_height=data.get('die_height', 1000.0),
        )

        system.target_clock_period_ps = data.get('target_clock_period_ps', 1000.0)
        system.target_temperature_K = data.get('target_temperature_K', 350.0)

        for p_data in data.get('particles', []):
            system.add_particle(FluxionParticle.from_dict(p_data))

        for c_data in data.get('connections', []):
            system.add_connection(FluxionConnection.from_dict(c_data))

        for path_data in data.get('critical_paths', []):
            system.add_critical_path(CriticalPath.from_dict(path_data))

        return system


class CircuitParticles(FluxionParticleSystem):
    """
    Circuit-specific particle system loaded from FLUXION JSON.

    This is the main class for working with circuits in FLUXION.
    """

    def __init__(self, module_name: str = "top", **kwargs):
        super().__init__(**kwargs)
        self.module_name = module_name
        self.total_gates = 0
        self.total_nets = 0
        self.total_power_pw = 0.0
        self.total_area_um2 = 0.0
        self.max_logic_level = 0

    @classmethod
    def load(cls, filepath: str) -> 'CircuitParticles':
        """Load circuit from FLUXION JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)

    def save(self, filepath: str) -> None:
        """Save circuit to FLUXION JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_dict(cls, data: dict) -> 'CircuitParticles':
        """Create from dictionary."""
        circuit = cls(
            module_name=data.get('module', 'top'),
            die_width=data.get('physical_constraints', {}).get('die_width_um', 1000.0),
            die_height=data.get('physical_constraints', {}).get('die_height_um', 1000.0),
        )

        # Load statistics
        stats = data.get('statistics', {})
        circuit.total_gates = stats.get('total_gates', 0)
        circuit.total_nets = stats.get('total_nets', 0)
        circuit.total_power_pw = stats.get('total_power_pw', 0.0)
        circuit.total_area_um2 = stats.get('total_area_um2', 0.0)
        circuit.max_logic_level = stats.get('max_logic_level', 0)

        # Load target clock period
        constraints = data.get('physical_constraints', {})
        circuit.target_clock_period_ps = constraints.get('target_clock_period_ps', 1000.0)

        # Load particles
        for p_data in data.get('nodes', []):
            circuit.add_particle(FluxionParticle.from_dict(p_data))

        # Load connections
        for c_data in data.get('connections', []):
            circuit.add_connection(FluxionConnection.from_dict(c_data))

        # Load critical paths
        for path_data in data.get('critical_paths', []):
            circuit.add_critical_path(CriticalPath.from_dict(path_data))

        return circuit

    def compute_statistics(self) -> None:
        """Compute and cache aggregate statistics for this circuit."""
        self.total_gates = len(self.particles)
        self.total_nets = len(self.connections)
        self.total_power_pw = sum(p.power_pw for p in self.particles.values())
        self.total_area_um2 = sum(p.area_um2 for p in self.particles.values())
        self.compute_logic_levels()
        self.max_logic_level = max(
            (p.level for p in self.particles.values()), default=0
        )

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            'fluxion_version': '1.0.0',
            'format': 'circuit_particles',
            'module': self.module_name,
            'statistics': {
                'total_gates': self.total_gates,
                'total_nets': self.total_nets,
                'total_power_pw': self.total_power_pw,
                'total_area_um2': self.total_area_um2,
                'max_logic_level': self.max_logic_level,
                'critical_paths': len(self.critical_paths),
            },
            'physical_constraints': {
                'die_width_um': self.die_width,
                'die_height_um': self.die_height,
                'target_clock_period_ps': self.target_clock_period_ps,
            },
            'nodes': [p.to_dict() for p in self.particles.values()],
            'connections': [c.to_dict() for c in self.connections],
            'critical_paths': [p.to_dict() for p in self.critical_paths],
        }


def load_circuit_particles(filepath: str) -> CircuitParticles:
    """
    Convenience function to load circuit particles from file.

    Args:
        filepath: Path to FLUXION JSON file

    Returns:
        CircuitParticles object
    """
    return CircuitParticles.load(filepath)