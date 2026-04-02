"""
FLUXION Force Fields

Implements the four physical forces used in the Thermodynamic Placement Engine:
1. Wire Tension - Springs pulling connected gates together (shorter wires)
2. Thermal Repulsion - Charged particles pushing hot gates apart (heat spreading)
3. Timing Gravity - Gravity pulling critical path gates forward (faster paths)
4. TopoLoss - Shape preservation force (circuit topology integrity)

These forces are combined to create the energy landscape that the annealing
process optimizes.
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

from .percolation import ThermalPercolationChecker
from .barnes_hut import BarnesHutTree


@dataclass
class ForceResult:
    """Result of force calculation."""
    forces: np.ndarray  # Nx2 array of force vectors
    energy: float  # Total potential energy
    max_force: float  # Maximum force magnitude
    force_details: Optional[Dict] = None  # Detailed breakdown


class ForceField(ABC):
    """
    Abstract base class for force fields.

    Each force field calculates forces on particles based on different
    physical analogies. Forces are combined to create the total force
    landscape for optimization.
    """

    def __init__(self, name: str, weight: float = 1.0):
        """
        Initialize force field.

        Args:
            name: Name of the force field
            weight: Weight factor for this force in the composite
        """
        self.name = name
        self.weight = weight
        self.enabled = True

    @abstractmethod
    def calculate(self, system: FluxionParticleSystem) -> ForceResult:
        """
        Calculate forces on all particles.

        Args:
            system: The particle system

        Returns:
            ForceResult with forces, energy, and statistics
        """
        pass

    @abstractmethod
    def calculate_energy(self, system: FluxionParticleSystem) -> float:
        """
        Calculate total potential energy for this force field.

        Args:
            system: The particle system

        Returns:
            Total potential energy
        """
        pass

    def set_weight(self, weight: float) -> None:
        """Set the weight factor for this force."""
        self.weight = weight

    def enable(self) -> None:
        """Enable this force field."""
        self.enabled = True

    def disable(self) -> None:
        """Disable this force field."""
        self.enabled = False


class WireTensionForce(ForceField):
    """
    Wire Tension Force - Springs pulling connected gates together.

    This force models wire connections as springs, pulling connected
    gates closer together. The goal is to minimize wire length, which
    reduces signal delay, power consumption, and routing congestion.

    Physics analogy: Hooke's Law spring force
    F = -k * (distance - rest_length)

    Energy: E = 0.5 * k * sum((d_ij - L_ij)^2) for all connected pairs
    Force: F_i = -k * sum((x_i - x_j) * (1 - L_ij/d_ij)) for neighbors j
    """

    def __init__(self, weight: float = 1.0, spring_constant: float = 1.0,
                 rest_length: float = 10.0, critical_weight: float = 2.0):
        """
        Initialize wire tension force.

        Args:
            weight: Overall weight factor
            spring_constant: Spring constant k (force per unit length)
            rest_length: Natural spring length in micrometers
            critical_weight: Additional weight for critical path connections
        """
        super().__init__("WireTension", weight)
        self.spring_constant = spring_constant
        self.rest_length = rest_length
        self.critical_weight = critical_weight

    def calculate(self, system: FluxionParticleSystem) -> ForceResult:
        """Calculate wire tension forces on all particles."""
        n = len(system.particles)
        forces = np.zeros((n, 2))
        total_energy = 0.0
        max_force = 0.0

        # Create index mapping
        id_to_idx = {pid: idx for idx, pid in enumerate(system.particles.keys())}
        particles = list(system.particles.values())

        # Calculate forces for each connection
        for conn in system.connections:
            if conn.source_id not in id_to_idx or conn.dest_id not in id_to_idx:
                continue

            i = id_to_idx[conn.source_id]
            j = id_to_idx[conn.dest_id]

            p_i = particles[i]
            p_j = particles[j]

            # Calculate distance
            dx = p_j.x - p_i.x
            dy = p_j.y - p_i.y
            distance = np.sqrt(dx**2 + dy**2)

            if distance < 1e-10:
                continue  # Avoid division by zero

            # Calculate spring force magnitude
            # For short connections, we want strong attraction
            # For long connections, we want to minimize length
            effective_k = self.spring_constant
            if conn.is_critical_path:
                effective_k *= self.critical_weight

            # Force magnitude: F = k * (distance - rest_length)
            # This pulls particles together when distance > rest_length
            # But we always want to minimize wire length, so we use:
            force_magnitude = effective_k * distance

            # Unit vector from i to j
            ux = dx / distance
            uy = dy / distance

            # Force on particle i (towards j)
            forces[i, 0] += force_magnitude * ux
            forces[i, 1] += force_magnitude * uy

            # Force on particle j (towards i, Newton's 3rd law)
            forces[j, 0] -= force_magnitude * ux
            forces[j, 1] -= force_magnitude * uy

            # Energy: 0.5 * k * distance^2 (we want to minimize total wirelength)
            total_energy += 0.5 * effective_k * distance**2

            # Track max force
            max_force = max(max_force, force_magnitude)

        return ForceResult(
            forces=forces * self.weight,
            energy=total_energy * self.weight,
            max_force=max_force * self.weight,
            force_details={'total_wirelength': system.total_wirelength()}
        )

    def calculate_energy(self, system: FluxionParticleSystem) -> float:
        """Calculate total wire length energy."""
        total_energy = 0.0
        particles = system.particles

        for conn in system.connections:
            if conn.source_id in particles and conn.dest_id in particles:
                p_src = particles[conn.source_id]
                p_dst = particles[conn.dest_id]

                distance = np.sqrt((p_src.x - p_dst.x)**2 + (p_src.y - p_dst.y)**2)

                effective_k = self.spring_constant
                if conn.is_critical_path:
                    effective_k *= self.critical_weight

                total_energy += 0.5 * effective_k * distance**2

        return total_energy * self.weight


class ThermalRepulsionForce(ForceField):
    """
    Thermal Repulsion Force - Charged particles pushing hot gates apart.

    This force prevents thermal hotspots by pushing high-power gates
    apart. It models thermal spreading using electrostatic repulsion
    analogy.

    Physics analogy: Coulomb repulsion between charged particles
    F = k_e * q1 * q2 / r^2

    For thermal: F_ij = k_thermal * P_i * P_j / r_ij^2
    where P_i is the power dissipation of gate i

    This spreads heat-generating gates across the die.
    """

    def __init__(self, weight: float = 1.0, thermal_constant: float = 100.0,
                 min_distance: float = 5.0):
        """
        Initialize thermal repulsion force.

        Args:
            weight: Overall weight factor
            thermal_constant: Thermal repulsion constant
            min_distance: Minimum distance to avoid singularity
        """
        super().__init__("ThermalRepulsion", weight)
        self.thermal_constant = thermal_constant
        self.percolation_checker = ThermalPercolationChecker()
        self.barnes_hut_cache = None

    def calculate(self, system: FluxionParticleSystem) -> ForceResult:
        """Calculate thermal repulsion forces on all particles."""
        n = len(system.particles)
        forces = np.zeros((n, 2))
        total_energy = 0.0
        # -------- VECTORIZED N^2 --------
        if n == 0:
            return ForceResult(
                forces=forces * self.weight,
                energy=0.0,
                max_force=0.0,
                force_details={'max_temperature': 0, 'percolation_risk': 0.0, 'is_percolating': False}
            )

        particles = list(system.particles.values())
        positions = np.array([[p.x, p.y] for p in particles])
        powers = np.array([p.power_pw for p in particles])
        charges = np.sqrt(powers + 1)
        q = charges

        # For large designs (N > 5000), use O(N log N) Barnes-Hut
        if n > 5000:
            if self.barnes_hut_cache is None:
                self.barnes_hut_cache = BarnesHutTree(theta=0.5)
            
            self.barnes_hut_cache.build(positions, charges)
            forces, total_energy = self.barnes_hut_cache.compute_repulsion_forces(
                positions, charges, self.thermal_constant, self.min_distance
            )
            max_force = np.max(np.linalg.norm(forces, axis=1)) if len(forces) > 0 else 0.0
            
            perc_result = self.percolation_checker.analyze(system)
            return ForceResult(
                forces=forces * self.weight,
                energy=total_energy * self.weight,
                max_force=max_force * self.weight,
                force_details={
                    'max_temperature': system.max_temperature(),
                    'using_barnes_hut': True,
                    'percolation_risk': perc_result.percolation_risk,
                    'is_percolating': perc_result.is_percolating
                }
            )

        diffs = positions[np.newaxis, :, :] - positions[:, np.newaxis, :]  # Shape: (N, N, 2)
        dist_sq = np.sum(diffs**2, axis=2)
        
        # Set diagonal to infinity to avoid self-repulsion division by zero
        np.fill_diagonal(dist_sq, np.inf)

        dist = np.sqrt(dist_sq)
        dist = np.maximum(dist, self.min_distance)

        q_prod = q[:, np.newaxis] * q[np.newaxis, :]
        force_mag = self.thermal_constant * q_prod / (dist**2)

        # Force direction: -diffs / dist
        forces[:, 0] = -np.sum(force_mag * diffs[:, :, 0] / dist, axis=1)
        forces[:, 1] = -np.sum(force_mag * diffs[:, :, 1] / dist, axis=1)

        total_energy = np.sum(self.thermal_constant * q_prod / dist) / 2.0
        max_force = np.max(force_mag) if n > 1 else 0.0

        perc_result = self.percolation_checker.analyze(system)

        return ForceResult(
            forces=forces * self.weight,
            energy=total_energy * self.weight,
            max_force=max_force * self.weight,
            force_details={
                'max_temperature': system.max_temperature(),
                'percolation_risk': perc_result.percolation_risk,
                'is_percolating': perc_result.is_percolating
            }
        )

    def calculate_energy(self, system: FluxionParticleSystem) -> float:
        """Calculate total thermal energy."""
        total_energy = 0.0
        particles = list(system.particles.values())
        if n == 0: return 0.0

        positions = np.array([[p.x, p.y] for p in particles])
        charges = np.array([np.sqrt(p.power_pw + 1) for p in particles])

        if n > 5000:
            if self.barnes_hut_cache is None:
                self.barnes_hut_cache = BarnesHutTree(theta=0.5)
            self.barnes_hut_cache.build(positions, charges)
            _, total_energy = self.barnes_hut_cache.compute_repulsion_forces(
                positions, charges, self.thermal_constant, self.min_distance
            )
            return total_energy * self.weight

        if n < 2:
            return 0.0
            
        positions = np.array([[p.x, p.y] for p in particles])
        powers = np.array([p.power_pw for p in particles])
        q = np.sqrt(powers + 1)

        diffs = positions[np.newaxis, :, :] - positions[:, np.newaxis, :]
        dist_sq = np.sum(diffs**2, axis=2)
        np.fill_diagonal(dist_sq, np.inf)

        dist = np.sqrt(dist_sq)
        dist = np.maximum(dist, self.min_distance)

        q_prod = q[:, np.newaxis] * q[np.newaxis, :]
        total_energy = np.sum(self.thermal_constant * q_prod / dist) / 2.0

        return total_energy * self.weight


class TimingGravityForce(ForceField):
    """
    Timing Gravity Force - Gravity pulling critical path gates forward.

    This force arranges critical timing paths to minimize delay.
    Gates on critical paths are pulled in a "forward" direction based
    on their timing slack, creating a layout where signals flow naturally.

    Physics analogy: Gravitational force towards optimal timing position
    F_i = -k_timing * slack_i * direction_i

    Gates with negative slack (timing violations) are pulled more strongly.
    """

    def __init__(self, weight: float = 1.0, timing_constant: float = 0.5,
                 target_clock_ps: float = 1000.0):
        """
        Initialize timing gravity force.

        Args:
            weight: Overall weight factor
            timing_constant: Timing optimization strength
            target_clock_ps: Target clock period in picoseconds
        """
        super().__init__("TimingGravity", weight)
        self.timing_constant = timing_constant
        self.target_clock_ps = target_clock_ps

    def calculate(self, system: FluxionParticleSystem) -> ForceResult:
        """Calculate timing gravity forces on all particles."""
        n = len(system.particles)
        forces = np.zeros((n, 2))
        total_energy = 0.0
        max_force = 0.0

        particles = list(system.particles.values())

        # Calculate logic levels if not already computed
        system.compute_logic_levels()

        # Find max logic level
        max_level = max(p.level for p in particles) if particles else 1

        # Force direction is typically left-to-right (towards outputs)
        # but can be customized based on design flow

        for i, particle in enumerate(particles):
            # Calculate slack for this path
            # Negative slack means timing violation
            path_delay = particle.delay_ps
            slack = self.target_clock_ps - path_delay

            # Force proportional to timing criticality
            # Pull towards position corresponding to logic level
            target_x = (particle.level / max_level) * system.die_width if max_level > 0 else 0

            # Current position vs target
            dx = target_x - particle.x

            # Force towards target, weighted by timing criticality
            timing_weight = 1.0 + max(0, -slack / self.target_clock_ps)

            force_x = self.timing_constant * timing_weight * dx
            force_y = 0.0  # No vertical timing force

            forces[i, 0] += force_x
            forces[i, 1] += force_y

            # Energy: potential energy of being away from optimal timing position
            total_energy += 0.5 * self.timing_constant * timing_weight * (dx**2)

            max_force = max(max_force, abs(force_x))

        # Additional: pull connected critical path gates closer
        id_to_idx = {p.id: idx for idx, p in enumerate(particles)}

        for path in system.critical_paths:
            # Each consecutive pair in critical path should be close
            for k in range(len(path.node_ids) - 1):
                id1 = path.node_ids[k]
                id2 = path.node_ids[k + 1]

                if id1 in id_to_idx and id2 in id_to_idx:
                    i = id_to_idx[id1]
                    j = id_to_idx[id2]

                    p1 = particles[i]
                    p2 = particles[j]

                    dx = p2.x - p1.x
                    dy = p2.y - p1.y
                    distance = np.sqrt(dx**2 + dy**2)

                    if distance < 1e-10:
                        continue

                    # Stronger attraction for critical path gates
                    force_mag = self.timing_constant * 2.0 * distance

                    ux = dx / distance
                    uy = dy / distance

                    forces[i, 0] += force_mag * ux
                    forces[i, 1] += force_mag * uy
                    forces[j, 0] -= force_mag * ux
                    forces[j, 1] -= force_mag * uy

                    total_energy += 0.5 * self.timing_constant * 2.0 * distance**2

        return ForceResult(
            forces=forces * self.weight,
            energy=total_energy * self.weight,
            max_force=max_force * self.weight,
            force_details={
                'critical_path_delay': system.critical_path_delay(),
                'target_clock_ps': self.target_clock_ps
            }
        )

    def calculate_energy(self, system: FluxionParticleSystem) -> float:
        """Calculate total timing energy."""
        total_energy = 0.0
        particles = list(system.particles.values())
        max_level = max((p.level for p in particles), default=1)

        for particle in particles:
            target_x = (particle.level / max_level) * system.die_width if max_level > 0 else 0
            dx = target_x - particle.x

            path_delay = particle.delay_ps
            slack = self.target_clock_ps - path_delay
            timing_weight = 1.0 + max(0, -slack / self.target_clock_ps)

            total_energy += 0.5 * self.timing_constant * timing_weight * (dx**2)

        return total_energy * self.weight


class TopoLossForce(ForceField):
    """
    TopoLoss Force - Shape preservation force (circuit topology integrity).

    This force maintains the circuit's logical topology during placement.
    It uses persistent homology concepts to ensure that the placement
    doesn't break the circuit's functional structure.

    The key insight is that circuits have a natural "shape" defined by
    their connectivity patterns. TopoLoss preserves this shape during
    physical placement, preventing the optimizer from finding physically
    optimal but logically invalid solutions.

    Physics analogy: Shape-preserving elastic potential
    E = k * sum((d_ij - d_ij_expected)^2) for topologically important pairs
    """

    def __init__(self, weight: float = 1.0, topology_constant: float = 0.3,
                 preserve_hierarchy: bool = True):
        """
        Initialize TopoLoss force.

        Args:
            weight: Overall weight factor
            topology_constant: Topology preservation strength
            preserve_hierarchy: Whether to preserve hierarchical structure
        """
        super().__init__("TopoLoss", weight)
        self.topology_constant = topology_constant
        self.preserve_hierarchy = preserve_hierarchy

    def calculate(self, system: FluxionParticleSystem) -> ForceResult:
        """Calculate TopoLoss forces on all particles."""
        n = len(system.particles)
        forces = np.zeros((n, 2))
        total_energy = 0.0
        max_force = 0.0

        particles = list(system.particles.values())
        id_to_idx = {p.id: idx for idx, p in enumerate(particles)}

        # Compute expected distances based on topology
        # Gates that are logically connected should be physically close
        expected_distances = {}

        for conn in system.connections:
            if conn.source_id in id_to_idx and conn.dest_id in id_to_idx:
                key = (min(conn.source_id, conn.dest_id),
                       max(conn.source_id, conn.dest_id))
                # Expected distance based on connection type
                expected_distances[key] = 20.0  # Base expected distance
                if conn.is_critical_path:
                    expected_distances[key] = 10.0  # Critical paths should be closer

        # Calculate forces to maintain topology
        for conn in system.connections:
            if conn.source_id not in id_to_idx or conn.dest_id not in id_to_idx:
                continue

            i = id_to_idx[conn.source_id]
            j = id_to_idx[conn.dest_id]

            p_i = particles[i]
            p_j = particles[j]

            dx = p_j.x - p_i.x
            dy = p_j.y - p_i.y
            distance = np.sqrt(dx**2 + dy**2)

            if distance < 1e-10:
                continue

            key = (min(conn.source_id, conn.dest_id), max(conn.source_id, conn.dest_id))
            expected = expected_distances.get(key, 20.0)

            # Force to restore expected distance
            # F = k * (distance - expected) * direction
            delta = distance - expected
            force_mag = self.topology_constant * delta

            ux = dx / distance
            uy = dy / distance

            # Pull together if too far, push apart if too close
            if delta > 0:  # Too far apart
                forces[i, 0] += force_mag * ux
                forces[i, 1] += force_mag * uy
                forces[j, 0] -= force_mag * ux
                forces[j, 1] -= force_mag * uy
            else:  # Too close together
                forces[i, 0] += force_mag * ux  # Negative force pushes apart
                forces[i, 1] += force_mag * uy
                forces[j, 0] -= force_mag * ux
                forces[j, 1] -= force_mag * uy

            # Energy
            total_energy += 0.5 * self.topology_constant * (delta**2)
            max_force = max(max_force, abs(force_mag))

        # Hierarchical structure preservation
        if self.preserve_hierarchy:
            # Gates from same module should cluster together
            # This is identified by name prefix
            module_clusters = {}
            for particle in particles:
                # Extract module name from hierarchy
                parts = particle.name.split('.')
                if len(parts) > 1:
                    module = '.'.join(parts[:-1])
                else:
                    module = 'top'

                if module not in module_clusters:
                    module_clusters[module] = []
                module_clusters[module].append(particle)

            # Calculate cluster centroids and pull towards them
            for module, cluster_particles in module_clusters.items():
                if len(cluster_particles) < 2:
                    continue

                # Calculate centroid
                cx = np.mean([p.x for p in cluster_particles])
                cy = np.mean([p.y for p in cluster_particles])

                # Weak force towards centroid
                for particle in cluster_particles:
                    idx = id_to_idx[particle.id]

                    dx = cx - particle.x
                    dy = cy - particle.y

                    # Small force to keep hierarchy together
                    force_x = 0.1 * self.topology_constant * dx
                    force_y = 0.1 * self.topology_constant * dy

                    forces[idx, 0] += force_x
                    forces[idx, 1] += force_y

                    total_energy += 0.05 * self.topology_constant * (dx**2 + dy**2)

        return ForceResult(
            forces=forces * self.weight,
            energy=total_energy * self.weight,
            max_force=max_force * self.weight,
            force_details={
                'topology_violations': 0,  # Can be computed with more analysis
                'hierarchical_depth': max((p.name.count('.') for p in particles), default=0)
            }
        )

    def calculate_energy(self, system: FluxionParticleSystem) -> float:
        """Calculate total TopoLoss energy."""
        # Simplified - uses distance variance within connected components
        total_energy = 0.0
        particles = system.particles

        for conn in system.connections:
            if conn.source_id in particles and conn.dest_id in particles:
                p_src = particles[conn.source_id]
                p_dst = particles[conn.dest_id]

                distance = np.sqrt((p_src.x - p_dst.x)**2 + (p_src.y - p_dst.y)**2)
                expected = 10.0 if conn.is_critical_path else 20.0

                delta = distance - expected
                total_energy += 0.5 * self.topology_constant * (delta**2)

        return total_energy * self.weight


class CompositeForceField(ForceField):
    """
    Composite force field combining multiple force fields.

    This combines all four FLUXION forces with configurable weights:
    - Wire Tension
    - Thermal Repulsion
    - Timing Gravity
    - TopoLoss

    The total force is the weighted sum of all component forces.
    """

    def __init__(self, weight: float = 1.0):
        """Initialize composite force field with all FLUXION forces."""
        super().__init__("Composite", weight)

        self.wire_tension = WireTensionForce(weight=1.0)
        self.thermal_repulsion = ThermalRepulsionForce(weight=0.5)
        self.timing_gravity = TimingGravityForce(weight=0.8)
        self.topoloss = TopoLossForce(weight=0.3)
        
        # Import here to avoid circular imports
        from .force_density import DensityEqualizationForce
        from .force_electrostatic import ElectrostaticSmoothingForce
        
        self.density_equalization = DensityEqualizationForce(weight=0.0)  # Disabled by default
        self.electrostatic_smoothing = ElectrostaticSmoothingForce(weight=0.0) # Disabled by default

        self.force_fields = [
            self.wire_tension,
            self.thermal_repulsion,
            self.timing_gravity,
            self.topoloss,
            self.density_equalization,
            self.electrostatic_smoothing,
        ]

    def calculate(self, system: FluxionParticleSystem) -> ForceResult:
        """Calculate combined forces from all force fields."""
        n = len(system.particles)
        total_forces = np.zeros((n, 2))
        total_energy = 0.0
        max_force = 0.0
        all_details = {}

        for force_field in self.force_fields:
            if not force_field.enabled:
                continue

            result = force_field.calculate(system)

            total_forces += result.forces
            total_energy += result.energy
            max_force = max(max_force, result.max_force)

            if result.force_details:
                all_details[force_field.name] = result.force_details

        return ForceResult(
            forces=total_forces * self.weight,
            energy=total_energy * self.weight,
            max_force=max_force * self.weight,
            force_details=all_details
        )

    def calculate_energy(self, system: FluxionParticleSystem) -> float:
        """Calculate total energy from all force fields."""
        total_energy = 0.0
        for force_field in self.force_fields:
            if force_field.enabled:
                total_energy += force_field.calculate_energy(system)
        return total_energy * self.weight

    def set_weights(self, wire_tension: float = None, thermal_repulsion: float = None,
                    timing_gravity: float = None, topoloss: float = None,
                    density: float = None, electrostatic: float = None) -> None:
        """
        Set individual force field weights.
        """
        if wire_tension is not None:
            self.wire_tension.set_weight(wire_tension)
        if thermal_repulsion is not None:
            self.thermal_repulsion.set_weight(thermal_repulsion)
        if timing_gravity is not None:
            self.timing_gravity.set_weight(timing_gravity)
        if topoloss is not None:
            self.topoloss.set_weight(topoloss)
        if density is not None:
            self.density_equalization.set_weight(density)
        if electrostatic is not None:
            self.electrostatic_smoothing.set_weight(electrostatic)

    def get_weights(self) -> Dict[str, float]:
        """Get current weights for all force fields."""
        return {
            'wire_tension': self.wire_tension.weight,
            'thermal_repulsion': self.thermal_repulsion.weight,
            'timing_gravity': self.timing_gravity.weight,
            'topoloss': self.topoloss.weight,
            'density': self.density_equalization.weight,
            'electrostatic': self.electrostatic_smoothing.weight,
        }