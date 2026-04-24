"""
FLUXION Thermodynamic Annealing

Implements the annealing scheduler for the Thermodynamic Placement Engine.
Annealing gradually reduces the "temperature" of the system, allowing it
to settle into a global optimum rather than getting stuck in local minima.

The annealing process:
1. Start: Gates move freely and explore many arrangements
2. Cool: Movement reduces, system settles into optimum
3. Result: Best placement found without training data

Key concepts:
- Temperature: Controls random movement/exploration
- Schedule: How temperature decreases over time
- Acceptance: Metropolis-Hastings criterion for accepting moves
"""

import numpy as np
from typing import Callable, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum


class ScheduleType(Enum):
    """Types of temperature schedules."""
    LINEAR = "linear"
    EXPONENTIAL = "exponential"
    LOGARITHMIC = "logarithmic"
    ADAPTIVE = "adaptive"
    DISCOVERY = "discovery"


@dataclass
class DiscoveryBasin:
    """A distinct placement basin discovered during exploration."""
    positions: np.ndarray
    energy: float
    cycle: int


@dataclass
class TemperatureSchedule:
    """
    Temperature schedule for annealing.

    Defines how temperature decreases from initial to final value.
    """

    initial_temp: float = 1000.0
    final_temp: float = 0.1
    schedule_type: ScheduleType = ScheduleType.EXPONENTIAL
    cooling_rate: float = 0.95  # For exponential schedule
    steps_per_temp: int = 100  # Steps at each temperature

    def temperature(self, step: int, total_steps: int) -> float:
        """
        Get temperature at given step.

        Args:
            step: Current step (0 to total_steps-1)
            total_steps: Total number of steps

        Returns:
            Temperature at this step
        """
        progress = step / max(total_steps - 1, 1)

        if self.schedule_type == ScheduleType.LINEAR:
            # Linear: T = T0 * (1 - progress) + Tf * progress
            return self.initial_temp * (1 - progress) + self.final_temp * progress

        elif self.schedule_type == ScheduleType.EXPONENTIAL:
            # Exponential: T = T0 * rate^step
            # Compute rate so that T actually reaches final_temp at the last step
            if self.final_temp > 0 and self.initial_temp > self.final_temp:
                rate = (self.final_temp / self.initial_temp) ** (1.0 / max(total_steps - 1, 1))
            else:
                rate = self.cooling_rate ** (1.0 / self.steps_per_temp)
            return self.initial_temp * (rate ** step)

        elif self.schedule_type == ScheduleType.LOGARITHMIC:
            # Logarithmic: T = T0 / log(step + 2)
            # Good for theoretical convergence guarantees
            return self.initial_temp / np.log(step + 2)

        elif self.schedule_type == ScheduleType.ADAPTIVE:
            # Adaptive: adjust based on acceptance rate
            # This is a simplified version; real adaptive uses acceptance history
            return self.initial_temp * np.exp(-5 * progress)

        else:
            return self.initial_temp

    def get_schedule(self, total_steps: int) -> np.ndarray:
        """
        Get full temperature schedule as array.

        Args:
            total_steps: Total number of steps

        Returns:
            Array of temperatures
        """
        return np.array([self.temperature(i, total_steps) for i in range(total_steps)])


@dataclass
class AnnealingState:
    """
    State of the annealing process.

    Tracks current position, energy, and statistics.
    """
    step: int = 0
    temperature: float = 1000.0
    current_energy: float = float('inf')
    best_energy: float = float('inf')
    current_positions: Optional[np.ndarray] = None
    best_positions: Optional[np.ndarray] = None
    accepted_moves: int = 0
    rejected_moves: int = 0
    energy_history: List[float] = field(default_factory=list)
    temperature_history: List[float] = field(default_factory=list)

    @property
    def acceptance_rate(self) -> float:
        """Calculate move acceptance rate."""
        total = self.accepted_moves + self.rejected_moves
        return self.accepted_moves / max(total, 1)

    def record_step(self, energy: float, temperature: float) -> None:
        """Record state for this step."""
        self.current_energy = energy
        self.energy_history.append(energy)
        self.temperature_history.append(temperature)

        if energy < self.best_energy:
            self.best_energy = energy


class ThermodynamicAnnealing:
    """
    Thermodynamic annealing optimizer for placement.

    Uses simulated annealing with configurable temperature schedules
    to find globally optimal placements.
    """

    def __init__(self,
                 schedule: TemperatureSchedule = None,
                 seed: int = None):
        """
        Initialize annealing optimizer.

        Args:
            schedule: Temperature schedule (default: exponential)
            seed: Random seed for reproducibility
        """
        self.schedule = schedule or TemperatureSchedule()
        self.rng = np.random.default_rng(seed)
        self.state = AnnealingState()

    def metropolis_hastings_accept(self,
                                    current_energy: float,
                                    proposed_energy: float,
                                    temperature: float) -> bool:
        """
        Metropolis-Hastings acceptance criterion.

        Always accept if energy improves.
        Accept worse states with probability exp(-delta_E / T).

        Args:
            current_energy: Current system energy
            proposed_energy: Proposed system energy
            temperature: Current temperature

        Returns:
            True if move should be accepted
        """
        delta_energy = proposed_energy - current_energy

        # Always accept improvements
        if delta_energy <= 0:
            return True

        # Accept worse states with probability exp(-delta / T)
        if temperature <= 0:
            return False

        probability = np.exp(-delta_energy / temperature)
        return self.rng.random() < probability

    def propose_move(self,
                     positions: np.ndarray,
                     temperature: float,
                     bounds: Tuple[float, float, float, float] = None) -> Tuple[np.ndarray, int]:
        """
        Propose a random move for a particle.

        Args:
            positions: Current positions (Nx2 array)
            temperature: Current temperature
            bounds: (xmin, ymin, xmax, ymax) boundary

        Returns:
            (new_positions, moved_particle_index)
        """
        n = positions.shape[0]

        # Select random particle
        idx = self.rng.integers(0, n)

        # Move magnitude scales with temperature
        move_scale = np.sqrt(temperature) * 0.1

        # Propose new position
        new_positions = positions.copy()
        new_positions[idx, 0] += self.rng.normal(0, move_scale)
        new_positions[idx, 1] += self.rng.normal(0, move_scale)

        # Apply boundary constraints
        if bounds:
            xmin, ymin, xmax, ymax = bounds
            new_positions[idx, 0] = np.clip(new_positions[idx, 0], xmin, xmax)
            new_positions[idx, 1] = np.clip(new_positions[idx, 1], ymin, ymax)

        return new_positions, idx

    def propose_batch_move(self,
                           positions: np.ndarray,
                           temperature: float,
                           batch_size: int = None,
                           bounds: Tuple[float, float, float, float] = None) -> np.ndarray:
        """
        Propose moves for multiple particles simultaneously.

        Args:
            positions: Current positions (Nx2 array)
            temperature: Current temperature
            batch_size: Number of particles to move (default: sqrt(N))
            bounds: (xmin, ymin, xmax, ymax) boundary

        Returns:
            New positions array
        """
        n = positions.shape[0]
        if batch_size is None:
            batch_size = max(1, int(np.sqrt(n)))

        new_positions = positions.copy()

        # Move magnitude scales with temperature
        move_scale = np.sqrt(temperature) * 0.05

        # Select random particles
        indices = self.rng.choice(n, size=min(batch_size, n), replace=False)

        for idx in indices:
            new_positions[idx, 0] += self.rng.normal(0, move_scale)
            new_positions[idx, 1] += self.rng.normal(0, move_scale)

        # Apply boundary constraints
        if bounds:
            xmin, ymin, xmax, ymax = bounds
            new_positions[:, 0] = np.clip(new_positions[:, 0], xmin, xmax)
            new_positions[:, 1] = np.clip(new_positions[:, 1], ymin, ymax)

        return new_positions

    def step(self,
             positions: np.ndarray,
             energy_function: Callable[[np.ndarray], float],
             temperature: float,
             bounds: Tuple[float, float, float, float] = None) -> Tuple[np.ndarray, float, bool]:
        """
        Perform one annealing step.

        Args:
            positions: Current positions
            energy_function: Function that computes energy for positions
            temperature: Current temperature
            bounds: Position boundaries

        Returns:
            (new_positions, new_energy, accepted)
        """
        current_energy = energy_function(positions)

        # Propose move
        new_positions, _ = self.propose_move(positions, temperature, bounds)
        new_energy = energy_function(new_positions)

        # Accept or reject
        accepted = self.metropolis_hastings_accept(current_energy, new_energy, temperature)

        if accepted:
            self.state.accepted_moves += 1
            return new_positions, new_energy, True
        else:
            self.state.rejected_moves += 1
            return positions, current_energy, False

    def anneal(self,
               initial_positions: np.ndarray,
               energy_function: Callable[[np.ndarray], float],
               total_steps: int = 10000,
               bounds: Tuple[float, float, float, float] = None,
               callback: Callable[[int, float, float, np.ndarray], None] = None,
               verbose: bool = True) -> AnnealingState:
        """
        Run full annealing optimization.

        Args:
            initial_positions: Starting positions (Nx2 array)
            energy_function: Function that computes energy for positions
            total_steps: Total number of annealing steps
            bounds: (xmin, ymin, xmax, ymax) boundary
            callback: Optional callback function(step, temp, energy, positions)
            verbose: Print progress

        Returns:
            Final annealing state
        """
        # Initialize state
        self.state = AnnealingState()
        self.state.current_positions = initial_positions.copy()
        self.state.best_positions = initial_positions.copy()
        self.state.current_energy = energy_function(initial_positions)
        self.state.best_energy = self.state.current_energy

        positions = initial_positions.copy()
        best_positions = initial_positions.copy()

        if verbose:
            print(f"Starting annealing with {total_steps} steps")
            print(f"Initial energy: {self.state.current_energy:.2f}")

        for step in range(total_steps):
            # Get temperature
            temperature = self.schedule.temperature(step, total_steps)
            self.state.temperature = temperature
            self.state.step = step

            # Perform step
            positions, energy, accepted = self.step(
                positions, energy_function, temperature, bounds
            )

            # Update state
            self.state.current_energy = energy
            self.state.current_positions = positions.copy()
            self.state.record_step(energy, temperature)

            # Track best
            if energy < self.state.best_energy:
                self.state.best_energy = energy
                best_positions = positions.copy()
                self.state.best_positions = best_positions.copy()

            # Callback
            if callback:
                callback(step, temperature, energy, positions)

            # Progress report
            if verbose and (step + 1) % max(1, total_steps // 10) == 0:
                print(f"Step {step+1}/{total_steps}: T={temperature:.4f}, "
                      f"E={energy:.2f}, Best={self.state.best_energy:.2f}, "
                      f"Acc={self.state.acceptance_rate:.2%}")

        if verbose:
            print(f"Annealing complete. Best energy: {self.state.best_energy:.2f}")

        return self.state

    def fast_anneal(self,
                    initial_positions: np.ndarray,
                    energy_function: Callable[[np.ndarray], float],
                    total_steps: int = 10000,
                    bounds: Tuple[float, float, float, float] = None,
                    verbose: bool = True) -> AnnealingState:
        """
        Fast annealing with vectorized operations.

        Uses batch moves and vectorized energy calculation for speed.

        Args:
            initial_positions: Starting positions
            energy_function: Energy function
            total_steps: Total steps
            bounds: Position boundaries
            verbose: Print progress

        Returns:
            Final annealing state
        """
        n = initial_positions.shape[0]

        # Initialize
        self.state = AnnealingState()
        positions = initial_positions.copy()
        best_positions = initial_positions.copy()

        current_energy = energy_function(positions)
        self.state.current_energy = current_energy
        self.state.best_energy = current_energy

        if verbose:
            print(f"Fast annealing: {n} particles, {total_steps} steps")

        for step in range(total_steps):
            temperature = self.schedule.temperature(step, total_steps)

            # Batch move
            new_positions = self.propose_batch_move(positions, temperature, bounds=bounds)
            new_energy = energy_function(new_positions)

            # Accept/reject
            if self.metropolis_hastings_accept(current_energy, new_energy, temperature):
                positions = new_positions
                current_energy = new_energy
                self.state.accepted_moves += 1
            else:
                self.state.rejected_moves += 1

            # Track best
            if current_energy < self.state.best_energy:
                self.state.best_energy = current_energy
                best_positions = positions.copy()

            # Record
            self.state.record_step(current_energy, temperature)

            # Progress
            if verbose and (step + 1) % max(1, total_steps // 10) == 0:
                print(f"Step {step+1}/{total_steps}: T={temperature:.4f}, "
                      f"E={current_energy:.2f}, Best={self.state.best_energy:.2f}")

        self.state.best_positions = best_positions.copy()
        return self.state

    @staticmethod
    def default_schedule() -> TemperatureSchedule:
        """Get default temperature schedule."""
        return TemperatureSchedule(
            initial_temp=1000.0,
            final_temp=0.01,
            schedule_type=ScheduleType.EXPONENTIAL,
            cooling_rate=0.95,
            steps_per_temp=100
        )

    @staticmethod
    def fast_schedule() -> TemperatureSchedule:
        """Get fast temperature schedule for quick iterations."""
        return TemperatureSchedule(
            initial_temp=100.0,
            final_temp=0.1,
            schedule_type=ScheduleType.EXPONENTIAL,
            cooling_rate=0.90,
            steps_per_temp=50
        )

    @staticmethod
    def thorough_schedule() -> TemperatureSchedule:
        """Get thorough schedule for high-quality results."""
        return TemperatureSchedule(
            initial_temp=1000.0,
            final_temp=0.001,
            schedule_type=ScheduleType.LOGARITHMIC,
            cooling_rate=0.98,
            steps_per_temp=200
        )

    # ------------------------------------------------------------------ #
    #  DISCOVERY MODE — Stochastic Exploration for Novel Placements       #
    # ------------------------------------------------------------------ #

    def propose_levy_move(
        self,
        positions: np.ndarray,
        temperature: float,
        alpha: float = 1.5,
        bounds: Tuple[float, float, float, float] = None,
    ) -> np.ndarray:
        """
        Propose a Lévy flight move — heavy-tailed jump for exploration.

        Unlike Gaussian moves that make small local steps, Lévy flights
        occasionally make very large jumps, enabling the optimizer to
        teleport across the die and discover radically different basins.

        Args:
            positions: Current positions (Nx2)
            temperature: Current temperature (scales magnitude)
            alpha: Lévy exponent (1.0–2.0; lower = heavier tail)
            bounds: (xmin, ymin, xmax, ymax)

        Returns:
            New positions array
        """
        n = positions.shape[0]
        new_positions = positions.copy()

        # Pick a random subset to move (~sqrt(N) particles)
        k = max(1, int(np.sqrt(n)))
        indices = self.rng.choice(n, size=min(k, n), replace=False)

        # Lévy stable distribution: u / |v|^(1/alpha)
        # This gives heavy-tailed step sizes
        scale = np.sqrt(temperature) * 0.2
        u = self.rng.normal(0, scale, size=(len(indices), 2))
        v = self.rng.normal(0, 1.0, size=(len(indices), 2))
        v = np.maximum(np.abs(v), 1e-8)
        steps = u / (v ** (1.0 / alpha))

        # Clip extreme jumps to 30% of die dimension to stay reasonable
        if bounds:
            max_jump_x = (bounds[2] - bounds[0]) * 0.3
            max_jump_y = (bounds[3] - bounds[1]) * 0.3
            steps[:, 0] = np.clip(steps[:, 0], -max_jump_x, max_jump_x)
            steps[:, 1] = np.clip(steps[:, 1], -max_jump_y, max_jump_y)

        new_positions[indices] += steps

        if bounds:
            xmin, ymin, xmax, ymax = bounds
            new_positions[:, 0] = np.clip(new_positions[:, 0], xmin, xmax)
            new_positions[:, 1] = np.clip(new_positions[:, 1], ymin, ymax)

        return new_positions

    def discovery_anneal(
        self,
        initial_positions: np.ndarray,
        energy_function: Callable[[np.ndarray], float],
        total_steps: int = 10000,
        num_cycles: int = 3,
        reheat_fraction: float = 0.6,
        levy_probability: float = 0.15,
        num_basins: int = 5,
        bounds: Tuple[float, float, float, float] = None,
        callback: Callable[[int, float, float, np.ndarray], None] = None,
        verbose: bool = True,
    ) -> Tuple[AnnealingState, List[DiscoveryBasin]]:
        """
        Discovery annealing — explore multiple basins of the energy landscape.

        Instead of converging to a single minimum, this mode:
        1. Cools normally, then REHEATS to explore a new region
        2. Uses Lévy flights for occasional large jumps
        3. Tracks the top-K distinct placements discovered

        This finds novel, non-obvious gate arrangements that a greedy
        optimizer would never discover.

        Args:
            initial_positions: Starting positions (Nx2)
            energy_function: Energy function
            total_steps: Total steps across ALL cycles
            num_cycles: Number of reheat cycles
            reheat_fraction: How much of initial temp to reheat to (0-1)
            levy_probability: Probability of using Lévy flight per step
            num_basins: Track top-K distinct basins
            bounds: Position boundaries
            callback: Optional callback
            verbose: Print progress

        Returns:
            (final_state, list_of_discovered_basins)
        """
        steps_per_cycle = total_steps // max(num_cycles, 1)
        basins: List[DiscoveryBasin] = []

        self.state = AnnealingState()
        positions = initial_positions.copy()
        best_positions = initial_positions.copy()

        current_energy = energy_function(positions)
        self.state.current_energy = current_energy
        self.state.best_energy = current_energy

        if verbose:
            print(f"Discovery mode: {num_cycles} cycles × {steps_per_cycle} steps")
            print(f"  Lévy flight probability: {levy_probability:.0%}")
            print(f"  Reheat fraction: {reheat_fraction:.0%}")

        global_step = 0

        for cycle in range(num_cycles):
            # Compute cycle-specific temperatures
            if cycle == 0:
                cycle_init_temp = self.schedule.initial_temp
            else:
                cycle_init_temp = self.schedule.initial_temp * reheat_fraction

            cycle_final_temp = self.schedule.final_temp

            if verbose:
                print(f"\n--- Cycle {cycle+1}/{num_cycles} "
                      f"(T: {cycle_init_temp:.1f} → {cycle_final_temp:.4f}) ---")

            for step in range(steps_per_cycle):
                # Temperature for this step within the cycle
                progress = step / max(steps_per_cycle - 1, 1)
                if cycle_init_temp > cycle_final_temp and cycle_final_temp > 0:
                    rate = (cycle_final_temp / cycle_init_temp) ** (1.0 / max(steps_per_cycle - 1, 1))
                    temperature = cycle_init_temp * (rate ** step)
                else:
                    temperature = cycle_init_temp * (1 - progress) + cycle_final_temp * progress

                self.state.temperature = temperature
                self.state.step = global_step

                # Decide: Lévy flight or normal Gaussian move
                use_levy = self.rng.random() < levy_probability
                if use_levy:
                    new_positions = self.propose_levy_move(
                        positions, temperature, bounds=bounds
                    )
                else:
                    new_positions, _ = self.propose_move(
                        positions, temperature, bounds
                    )

                new_energy = energy_function(new_positions)

                # Accept/reject
                if self.metropolis_hastings_accept(current_energy, new_energy, temperature):
                    positions = new_positions
                    current_energy = new_energy
                    self.state.accepted_moves += 1
                else:
                    self.state.rejected_moves += 1

                # Track best
                if current_energy < self.state.best_energy:
                    self.state.best_energy = current_energy
                    best_positions = positions.copy()

                self.state.record_step(current_energy, temperature)

                if callback:
                    callback(global_step, temperature, current_energy, positions)

                global_step += 1

            # End of cycle — record this basin
            basin = DiscoveryBasin(
                positions=positions.copy(),
                energy=current_energy,
                cycle=cycle,
            )
            basins.append(basin)

            if verbose:
                print(f"  Basin {cycle+1}: energy={current_energy:.2f}")

        # Keep only top-K basins by energy
        basins.sort(key=lambda b: b.energy)
        basins = basins[:num_basins]

        # Set final state to the global best
        self.state.best_positions = best_positions.copy()
        self.state.current_positions = positions.copy()

        if verbose:
            print(f"\nDiscovery complete. Found {len(basins)} basins.")
            print(f"  Best energy: {self.state.best_energy:.2f}")
            for i, b in enumerate(basins):
                print(f"  Basin {i+1}: energy={b.energy:.2f} (cycle {b.cycle+1})")

        return self.state, basins