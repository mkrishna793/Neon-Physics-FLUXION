"""
FLUXION GPU Accelerator

Provides GPU-accelerated force calculations using OpenCL.
This enables simulation of large circuits (10k+ gates) in seconds
rather than minutes.

The GPU acceleration uses OpenCL for hardware-agnostic acceleration,
supporting AMD, NVIDIA, and Intel GPUs.
"""

import numpy as np
from typing import Optional, Tuple, List
from dataclasses import dataclass
import warnings

# OpenCL kernel for force calculations
OPENCL_KERNEL_SOURCE = """
// Polyfill for sliding atomic add on floating point pointers
#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable

inline void atomicAddFloat(volatile __global float *addr, float val)
{
    union {
        unsigned int u32;
        float f32;
    } current, expected, next;

    current.f32 = *addr;
    do {
        expected.f32 = current.f32;
        next.f32 = expected.f32 + val;
        current.u32 = atomic_cmpxchg((volatile __global unsigned int *)addr,
                                     expected.u32, next.u32);
    } while (current.u32 != expected.u32);
}

// Wire tension force kernel
__kernel void wire_tension_force(
    __global const float* positions,    // Nx2 array
    __global const int* connections,    // Mx2 array (source, dest pairs)
    __global const float* weights,      // M weights (critical path multiplier)
    const int num_connections,
    const float spring_constant,
    __global float* forces              // Nx2 output forces
) {
    int idx = get_global_id(0);
    if (idx >= num_connections) return;

    int src = connections[idx * 2];
    int dst = connections[idx * 2 + 1];
    float weight = weights[idx];

    float sx = positions[src * 2];
    float sy = positions[src * 2 + 1];
    float dx_pos = positions[dst * 2];
    float dy_pos = positions[dst * 2 + 1];

    float diff_x = dx_pos - sx;
    float diff_y = dy_pos - sy;
    float dist = sqrt(diff_x * diff_x + diff_y * diff_y);

    if (dist < 1e-6f) return;

    float force_mag = spring_constant * weight * dist;
    float fx = force_mag * diff_x / dist;
    float fy = force_mag * diff_y / dist;

    // Atomic add for forces (each connection contributes)
    atomicAddFloat(&forces[src * 2], fx);
    atomicAddFloat(&forces[src * 2 + 1], fy);
    atomicAddFloat(&forces[dst * 2], -fx);
    atomicAddFloat(&forces[dst * 2 + 1], -fy);
}

// Thermal repulsion force kernel
__kernel void thermal_repulsion_force(
    __global const float* positions,    // Nx2 array
    __global const float* powers,      // N power values
    const int num_particles,
    const float thermal_constant,
    const float min_distance,
    __global float* forces              // Nx2 output forces
) {
    int i = get_global_id(0);
    if (i >= num_particles) return;

    float px_i = positions[i * 2];
    float py_i = positions[i * 2 + 1];
    float pwr_i = sqrt(powers[i] + 1.0f);

    float fx = 0.0f;
    float fy = 0.0f;

    // O(n^2) pairwise - can be optimized with spatial hashing
    for (int j = 0; j < num_particles; j++) {
        if (i == j) continue;

        float px_j = positions[j * 2];
        float py_j = positions[j * 2 + 1];
        float pwr_j = sqrt(powers[j] + 1.0f);

        float diff_x = px_j - px_i;
        float diff_y = py_j - py_i;
        float dist = max(sqrt(diff_x * diff_x + diff_y * diff_y), min_distance);

        float force_mag = thermal_constant * pwr_i * pwr_j / (dist * dist);

        // Repulsion: push away from j
        fx -= force_mag * diff_x / dist;
        fy -= force_mag * diff_y / dist;
    }

    forces[i * 2] += fx;
    forces[i * 2 + 1] += fy;
}

// Energy calculation kernel
__kernel void calculate_energy(
    __global const float* positions,
    __global const int* connections,
    __global const float* weights,
    const int num_connections,
    const float spring_constant,
    __global float* partial_energies
) {
    int idx = get_global_id(0);
    if (idx >= num_connections) return;

    int src = connections[idx * 2];
    int dst = connections[idx * 2 + 1];
    float weight = weights[idx];

    float sx = positions[src * 2];
    float sy = positions[src * 2 + 1];
    float dx_pos = positions[dst * 2];
    float dy_pos = positions[dst * 2 + 1];

    float diff_x = dx_pos - sx;
    float diff_y = dy_pos - sy;
    float dist_sq = diff_x * diff_x + diff_y * diff_y;

    partial_energies[idx] = 0.5f * spring_constant * weight * dist_sq;
}

// Reduce energy kernel
__kernel void reduce_energy(
    __global const float* partial_energies,
    const int num_elements,
    __global float* total_energy
) {
    int idx = get_global_id(0);
    if (idx >= num_elements) return;

    // Simple reduction - can be optimized with parallel reduction
    atomic_add(total_energy, partial_energies[idx]);
}

// Position update kernel (for annealing)
__kernel void update_positions(
    __global float* positions,
    __global const float* velocities,
    __global const float* forces,
    const int num_particles,
    const float dt,
    const float xmin,
    const float ymin,
    const float xmax,
    const float ymax
) {
    int idx = get_global_id(0);
    if (idx >= num_particles) return;

    // Update velocity (simple Euler integration)
    float vx = velocities[idx * 2] + forces[idx * 2] * dt;
    float vy = velocities[idx * 2 + 1] + forces[idx * 2 + 1] * dt;

    // Damping
    vx *= 0.99f;
    vy *= 0.99f;

    // Update position
    float x = positions[idx * 2] + vx * dt;
    float y = positions[idx * 2 + 1] + vy * dt;

    // Boundary constraints
    x = clamp(x, xmin, xmax);
    y = clamp(y, ymin, ymax);

    positions[idx * 2] = x;
    positions[idx * 2 + 1] = y;
    velocities[idx * 2] = vx;
    velocities[idx * 2 + 1] = vy;
}
"""


@dataclass
class GPUDevice:
    """Information about a GPU device."""
    name: str
    vendor: str
    device_type: str
    global_memory: int
    compute_units: int
    max_work_group_size: int


class GPUAccelerator:
    """
    Base class for GPU acceleration.

    Provides interface for GPU-accelerated force calculations.
    Falls back to CPU if OpenCL is not available.
    """

    def __init__(self):
        """Initialize GPU accelerator."""
        self.available = False
        self.context = None
        self.queue = None
        self.program = None
        self.devices: List[GPUDevice] = []

    def is_available(self) -> bool:
        """Check if GPU acceleration is available."""
        return self.available

    def get_devices(self) -> List[GPUDevice]:
        """Get list of available GPU devices."""
        return self.devices

    def calculate_forces(self,
                         positions: np.ndarray,
                         connections: np.ndarray,
                         weights: np.ndarray,
                         powers: np.ndarray,
                         spring_constant: float = 1.0,
                         thermal_constant: float = 100.0) -> np.ndarray:
        """
        Calculate forces using GPU acceleration.

        Args:
            positions: Nx2 array of particle positions
            connections: Mx2 array of (source, dest) pairs
            weights: M array of connection weights
            powers: N array of particle powers
            spring_constant: Wire tension spring constant
            thermal_constant: Thermal repulsion constant

        Returns:
            Nx2 array of forces
        """
        # Base implementation - CPU fallback
        return self._cpu_calculate_forces(
            positions, connections, weights, powers,
            spring_constant, thermal_constant
        )

    def _cpu_calculate_forces(self,
                              positions: np.ndarray,
                              connections: np.ndarray,
                              weights: np.ndarray,
                              powers: np.ndarray,
                              spring_constant: float,
                              thermal_constant: float) -> np.ndarray:
        """CPU fallback for force calculation."""
        n = positions.shape[0]
        forces = np.zeros((n, 2), dtype=np.float32)

        # Wire tension forces
        for i in range(len(connections)):
            src, dst = connections[i]
            weight = weights[i]

            diff = positions[dst] - positions[src]
            dist = np.linalg.norm(diff)

            if dist < 1e-6:
                continue

            force_mag = spring_constant * weight * dist
            force = force_mag * diff / dist

            forces[src] += force
            forces[dst] -= force

        # Thermal repulsion forces (simplified, O(n^2))
        for i in range(n):
            for j in range(i + 1, n):
                diff = positions[j] - positions[i]
                dist = max(np.linalg.norm(diff), 5.0)

                q_i = np.sqrt(powers[i] + 1)
                q_j = np.sqrt(powers[j] + 1)

                force_mag = thermal_constant * q_i * q_j / (dist * dist)
                force = force_mag * diff / dist

                forces[i] -= force
                forces[j] += force

        return forces

    def calculate_energy(self,
                         positions: np.ndarray,
                         connections: np.ndarray,
                         weights: np.ndarray,
                         spring_constant: float = 1.0) -> float:
        """
        Calculate total energy using GPU acceleration.

        Args:
            positions: Nx2 array of particle positions
            connections: Mx2 array of (source, dest) pairs
            weights: M array of connection weights
            spring_constant: Spring constant

        Returns:
            Total energy
        """
        # Base implementation - CPU fallback
        total_energy = 0.0

        for i in range(len(connections)):
            src, dst = connections[i]
            weight = weights[i]

            diff = positions[dst] - positions[src]
            dist_sq = np.dot(diff, diff)

            total_energy += 0.5 * spring_constant * weight * dist_sq

        return total_energy


class OpenCLAccelerator(GPUAccelerator):
    """
    OpenCL-based GPU acceleration.

    Uses pyopencl to accelerate force calculations.
    Falls back to CPU if OpenCL is not available.
    """

    def __init__(self, device_index: int = 0):
        """
        Initialize OpenCL accelerator.

        Args:
            device_index: Index of GPU device to use
        """
        super().__init__()
        self.device_index = device_index

        self._try_init_opencl()

    def _try_init_opencl(self) -> None:
        """Try to initialize OpenCL."""
        try:
            import pyopencl as cl

            # Get platforms
            platforms = cl.get_platforms()
            if not platforms:
                warnings.warn("No OpenCL platforms found, using CPU fallback")
                return

            # Get devices
            all_devices = []
            for platform in platforms:
                devices = platform.get_devices()
                for device in devices:
                    device_type = cl.device_type.to_string(device.type)
                    gpu_device = GPUDevice(
                        name=device.name,
                        vendor=device.vendor,
                        device_type=device_type,
                        global_memory=device.global_mem_size,
                        compute_units=device.max_compute_units,
                        max_work_group_size=device.max_work_group_size
                    )
                    all_devices.append((device, gpu_device))

            if not all_devices:
                warnings.warn("No OpenCL devices found, using CPU fallback")
                return

            # Select device
            if self.device_index >= len(all_devices):
                self.device_index = 0

            selected_device, device_info = all_devices[self.device_index]
            self.devices = [device_info]

            # Create context and queue
            self.context = cl.Context([selected_device])
            self.queue = cl.CommandQueue(self.context)

            # Build program
            self.program = cl.Program(self.context, OPENCL_KERNEL_SOURCE).build()

            self.available = True

        except ImportError:
            warnings.warn("pyopencl not installed, using CPU fallback")
        except Exception as e:
            warnings.warn(f"OpenCL initialization failed: {e}, using CPU fallback")

    def calculate_forces(self,
                         positions: np.ndarray,
                         connections: np.ndarray,
                         weights: np.ndarray,
                         powers: np.ndarray,
                         spring_constant: float = 1.0,
                         thermal_constant: float = 100.0) -> np.ndarray:
        """Calculate forces using OpenCL."""
        if not self.available:
            return self._cpu_calculate_forces(
                positions, connections, weights, powers,
                spring_constant, thermal_constant
            )

        try:
            import pyopencl as cl

            n = positions.shape[0]
            m = connections.shape[0]

            # Ensure correct dtypes
            positions = np.ascontiguousarray(positions, dtype=np.float32)
            connections = np.ascontiguousarray(connections, dtype=np.int32)
            weights = np.ascontiguousarray(weights, dtype=np.float32)
            powers = np.ascontiguousarray(powers, dtype=np.float32)

            forces = np.zeros((n, 2), dtype=np.float32)

            # Create buffers
            mf = cl.mem_flags
            positions_buf = cl.Buffer(self.context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=positions)
            connections_buf = cl.Buffer(self.context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=connections)
            weights_buf = cl.Buffer(self.context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=weights)
            powers_buf = cl.Buffer(self.context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=powers)
            forces_buf = cl.Buffer(self.context, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=forces)

            # Wire tension kernel
            self.program.wire_tension_force(
                self.queue, (m,), None,
                positions_buf, connections_buf, weights_buf,
                np.int32(m), np.float32(spring_constant), forces_buf
            )

            # Thermal repulsion kernel
            self.program.thermal_repulsion_force(
                self.queue, (n,), None,
                positions_buf, powers_buf,
                np.int32(n), np.float32(thermal_constant), np.float32(5.0), forces_buf
            )

            # Read back forces
            cl.enqueue_copy(self.queue, forces, forces_buf)

            return forces

        except Exception as e:
            warnings.warn(f"OpenCL force calculation failed: {e}, using CPU fallback")
            return self._cpu_calculate_forces(
                positions, connections, weights, powers,
                spring_constant, thermal_constant
            )

    def calculate_energy(self,
                         positions: np.ndarray,
                         connections: np.ndarray,
                         weights: np.ndarray,
                         spring_constant: float = 1.0) -> float:
        """Calculate energy using OpenCL."""
        if not self.available:
            return super().calculate_energy(positions, connections, weights, spring_constant)

        try:
            import pyopencl as cl

            m = connections.shape[0]

            # Ensure correct dtypes
            positions = np.ascontiguousarray(positions, dtype=np.float32)
            connections = np.ascontiguousarray(connections, dtype=np.int32)
            weights = np.ascontiguousarray(weights, dtype=np.float32)

            partial_energies = np.zeros(m, dtype=np.float32)
            total_energy = np.zeros(1, dtype=np.float32)

            # Create buffers
            mf = cl.mem_flags
            positions_buf = cl.Buffer(self.context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=positions)
            connections_buf = cl.Buffer(self.context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=connections)
            weights_buf = cl.Buffer(self.context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=weights)
            partial_buf = cl.Buffer(self.context, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=partial_energies)
            total_buf = cl.Buffer(self.context, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=total_energy)

            # Energy kernel
            self.program.calculate_energy(
                self.queue, (m,), None,
                positions_buf, connections_buf, weights_buf,
                np.int32(m), np.float32(spring_constant), partial_buf
            )

            # Reduce
            self.program.reduce_energy(
                self.queue, (m,), None,
                partial_buf, np.int32(m), total_buf
            )

            # Read result
            cl.enqueue_copy(self.queue, total_energy, total_buf)

            return float(total_energy[0])

        except Exception as e:
            warnings.warn(f"OpenCL energy calculation failed: {e}, using CPU fallback")
            return super().calculate_energy(positions, connections, weights, spring_constant)


def create_accelerator(device_index: int = 0, force_cpu: bool = False) -> GPUAccelerator:
    """
    Create a GPU accelerator instance.

    Args:
        device_index: Index of GPU device to use
        force_cpu: Force CPU fallback

    Returns:
        GPUAccelerator instance
    """
    if force_cpu:
        return GPUAccelerator()

    accelerator = OpenCLAccelerator(device_index)

    if not accelerator.is_available():
        warnings.warn("GPU acceleration not available, using CPU")
        return GPUAccelerator()

    return accelerator