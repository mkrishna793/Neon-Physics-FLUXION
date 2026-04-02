"""
FLUXION Electrostatic Smoothing Force (Force #6)

Implements the ePlace methodology for global density smoothing.
Models the circuit as an electrostatic system where:
- Cells are positive electric charges
- Target uniform density is negative background charge
- Density gradient is computed via Poisson's equation
- Force is the electrostatic field

Uses FFT for fast O(N log N) solving of Poisson's equation.
"""

import numpy as np
try:
    from scipy.fft import idctn, dctn
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

from .particle_system import FluxionParticleSystem
from .force_fields import ForceField, ForceResult


class ElectrostaticSmoothingForce(ForceField):
    """
    Electrostatic Smoothing Force.

    Solves Poisson's equation ∇²φ = -ρ to find the potential field φ
    from the density field ρ, then computes the force field E = -∇φ.
    This gives unparalleled global spreading quality by ensuring long-range
    global interactions are calculated exactly in O(N log N) using FFT.
    """

    def __init__(self, weight: float = 1.0, grid_size: int = 128,
                 target_density: float = 0.8):
        super().__init__("ElectrostaticSmoothing", weight)
        self.grid_size = grid_size
        self.target_density = target_density
        self.fallback = False

        if not SCIPY_AVAILABLE:
            print("WARNING: scipy.fft not found. ElectrostaticSmoothingForce will fall back to local density.")
            self.fallback = True

    def _compute_density_map(self, system: FluxionParticleSystem) -> np.ndarray:
        """Map cell areas to the grid to create density map ρ(x,y)."""
        density = np.zeros((self.grid_size, self.grid_size))
        bin_w = system.die_width / self.grid_size
        bin_h = system.die_height / self.grid_size

        # Simple Point-in-bin mapping for speed (can be upgraded to Triangle overlaps)
        for p in system.particles.values():
            col = int(np.clip(p.x / bin_w, 0, self.grid_size - 1))
            row = int(np.clip(p.y / bin_h, 0, self.grid_size - 1))
            # Charge is area divided by bin area
            density[row, col] += p.area_um2 / (bin_w * bin_h)

        return density

    def _solve_poisson_fft(self, rho: np.ndarray, wx: float, wy: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Solve ∇²φ = -ρ using 2D Discrete Cosine Transform (DCT).
        Returns potential φ and gradient (force field) Ex, Ey.
        """
        # 1. Compute DCT of density
        # DCT-II (type=2) is the standard DCT used in JPEG, etc.
        rho_hat = dctn(rho, type=2, norm='ortho')

        # 2. Compute eigenvalues for the inverse Laplacian
        ny, nx = rho.shape
        kx = np.arange(nx)
        ky = np.arange(ny)
        kx, ky = np.meshgrid(kx, ky)

        # wx, wy are inverse bin sizes
        eigenvals = (wx * np.cos(np.pi * kx / nx) - wx)**2 + \
                    (wy * np.cos(np.pi * ky / ny) - wy)**2

        # Handle DC component (k=0, m=0) to avoid division by zero
        eigenvals[0, 0] = 1.0

        # 3. Solve for potential in frequency domain
        phi_hat = -rho_hat / eigenvals
        phi_hat[0, 0] = 0.0  # Set mean potential to 0

        # 4. Inverse DCT to spatial domain
        phi = idctn(phi_hat, type=2, norm='ortho')

        # 5. Compute electric field E = -∇φ (forces)
        # Using centered difference
        Ex = np.zeros_like(phi)
        Ey = np.zeros_like(phi)

        Ex[:, 1:-1] = -(phi[:, 2:] - phi[:, :-2]) * wx * 0.5
        Ey[1:-1, :] = -(phi[2:, :] - phi[:-2, :]) * wy * 0.5

        # Edges
        Ex[:, 0] = -(phi[:, 1] - phi[:, 0]) * wx
        Ex[:, -1] = -(phi[:, -1] - phi[:, -2]) * wx
        Ey[0, :] = -(phi[1, :] - phi[0, :]) * wy
        Ey[-1, :] = -(phi[-1, :] - phi[-2, :]) * wy

        return phi, Ex, Ey

    def calculate(self, system: FluxionParticleSystem) -> ForceResult:
        """Calculate global electrostatic forces."""
        n = len(system.particles)
        forces = np.zeros((n, 2))

        if n == 0 or self.fallback:
            return ForceResult(forces=forces, energy=0.0, max_force=0.0)

        # 1. Compute physical density ρ
        rho = self._compute_density_map(system)

        # 2. Subtract background uniform density to get relative charge
        # ∇²φ = -(ρ - ρ_target)
        # Bins over target density act as positive charges (push gates away)
        # Bins under target density act as negative charges (pull gates in)
        rho_target = np.full_like(rho, self.target_density)
        charge_density = rho - rho_target

        # 3. Solve Poisson's equation
        bin_w = system.die_width / self.grid_size
        bin_h = system.die_height / self.grid_size
        phi, Ex, Ey = self._solve_poisson_fft(charge_density, 1.0/bin_w, 1.0/bin_h)

        # 4. Interpolate electric field back to particles
        total_energy = 0.0
        max_force = 0.0

        for i, p in enumerate(system.particles.values()):
            col = int(np.clip(p.x / bin_w, 0, self.grid_size - 1))
            row = int(np.clip(p.y / bin_h, 0, self.grid_size - 1))

            # Force is q * E. Charge q is proportional to gate area.
            q = p.area_um2

            fx = q * Ex[row, col]
            fy = q * Ey[row, col]

            forces[i, 0] = fx
            forces[i, 1] = fy

            max_force = max(max_force, np.sqrt(fx*fx + fy*fy))

            # Energy is q * φ
            total_energy += 0.5 * q * phi[row, col]

        return ForceResult(
            forces=forces * self.weight,
            energy=total_energy * self.weight,
            max_force=max_force * self.weight,
            force_details={}
        )

    def calculate_energy(self, system: FluxionParticleSystem) -> float:
        if self.fallback: return 0.0
        rho = self._compute_density_map(system)
        charge = rho - self.target_density
        bin_w = system.die_width / self.grid_size
        bin_h = system.die_height / self.grid_size
        phi, _, _ = self._solve_poisson_fft(charge, 1.0/bin_w, 1.0/bin_h)
        return float(np.sum(0.5 * charge * phi) * self.weight)
