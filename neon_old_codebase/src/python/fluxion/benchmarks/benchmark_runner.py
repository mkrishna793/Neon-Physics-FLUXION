"""
FLUXION Benchmark Runner

Unified runner that auto-detects benchmark format, runs FLUXION
optimization, and reports standard placement metrics:
- HPWL (Half-Perimeter Wire Length)
- Runtime
- Density overflow
- Critical path delay estimate
"""

import time
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

from ..particle_system import CircuitParticles
from ..tpe import ThermodynamicPlacementEngine, PlacementConfig, PlacementResult


@dataclass
class BenchmarkMetrics:
    """Standard placement quality metrics."""
    design_name: str
    num_cells: int
    num_nets: int
    die_width: float
    die_height: float

    # Quality
    hpwl: float = 0.0                # Half-Perimeter Wire Length
    total_wirelength: float = 0.0     # Manhattan wirelength
    density_overflow: float = 0.0     # Fraction of bins over capacity
    max_density: float = 0.0          # Worst-case bin utilization

    # Performance
    runtime_s: float = 0.0
    annealing_steps: int = 0
    ms_per_step: float = 0.0

    # Energy
    total_energy: float = 0.0
    wire_energy: float = 0.0
    thermal_energy: float = 0.0

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}

    def summary_line(self) -> str:
        return (
            f"{self.design_name:<20} | "
            f"{self.num_cells:>8,} cells | "
            f"HPWL={self.hpwl:>12,.0f} | "
            f"Time={self.runtime_s:>6.1f}s | "
            f"Density={self.max_density:>5.1%}"
        )


class BenchmarkRunner:
    """
    Runs FLUXION on industry benchmarks and reports metrics.

    Usage:
        runner = BenchmarkRunner(steps=5000)
        metrics = runner.run("path/to/design.aux")
        print(metrics.summary_line())
    """

    def __init__(self, steps: int = 5000, verbose: bool = True,
                 discovery_mode: bool = False,
                 adaptive_weights: bool = True,
                 seed: int = 42):
        self.steps = steps
        self.verbose = verbose
        self.discovery_mode = discovery_mode
        self.adaptive_weights = adaptive_weights
        self.seed = seed

    def run(self, benchmark_path: str, format: str = None) -> BenchmarkMetrics:
        """
        Run a benchmark.

        Args:
            benchmark_path: Path to benchmark file
            format: Force format ("bookshelf", "lefdef", "blif").
                    Auto-detected from extension if None.

        Returns:
            BenchmarkMetrics with results
        """
        path = Path(benchmark_path)
        fmt = format or self._detect_format(path)

        if self.verbose:
            print(f"\n{'='*60}")
            print(f"FLUXION Benchmark: {path.stem}")
            print(f"Format: {fmt} | Steps: {self.steps}")
            print(f"{'='*60}")

        # 1. Parse
        t0 = time.time()
        circuit = self._parse(path, fmt)
        parse_time = time.time() - t0

        if self.verbose:
            print(f"Parsed in {parse_time:.2f}s: "
                  f"{len(circuit.particles)} cells, "
                  f"{len(circuit.connections)} connections")

        # 2. Run FLUXION
        config = PlacementConfig(
            die_width=circuit.die_width,
            die_height=circuit.die_height,
            annealing_steps=self.steps,
            verbose=self.verbose,
            random_seed=self.seed,
            discovery_mode=self.discovery_mode,
            adaptive_weights=self.adaptive_weights,
        )

        engine = ThermodynamicPlacementEngine(config)
        engine.set_circuit(circuit)

        start = time.time()
        result = engine.optimize()
        runtime = time.time() - start

        # 3. Compute metrics
        metrics = self._compute_metrics(circuit, result, runtime, path.stem)
        metrics.annealing_steps = self.steps
        metrics.ms_per_step = (runtime / self.steps) * 1000 if self.steps else 0

        if self.verbose:
            print(f"\n{metrics.summary_line()}")

        return metrics

    def run_suite(self, benchmark_paths: List[str],
                  format: str = None) -> List[BenchmarkMetrics]:
        """Run multiple benchmarks and return all metrics."""
        results = []
        for path in benchmark_paths:
            try:
                m = self.run(path, format)
                results.append(m)
            except Exception as e:
                print(f"FAILED: {path} — {e}")
        return results

    def _detect_format(self, path: Path) -> str:
        """Auto-detect benchmark format from file extension."""
        ext = path.suffix.lower()
        if ext == '.aux':
            return 'bookshelf'
        elif ext == '.def':
            return 'lefdef'
        elif ext in ('.blif', '.v'):
            return 'blif'
        elif ext == '.nodes':
            return 'bookshelf'
        elif ext == '.lef':
            return 'lefdef'
        else:
            # Try to detect from contents
            return 'bookshelf'

    def _parse(self, path: Path, fmt: str) -> CircuitParticles:
        """Parse benchmark file into CircuitParticles."""
        if fmt == 'bookshelf':
            from .bookshelf_parser import BookshelfParser
            parser = BookshelfParser()
            if path.suffix == '.aux':
                return parser.parse(str(path))
            else:
                # Try to find related files
                base = path.parent
                stem = path.stem
                nodes = base / f"{stem}.nodes"
                nets = base / f"{stem}.nets"
                pl = base / f"{stem}.pl"
                scl = base / f"{stem}.scl"
                return parser.parse_from_files(
                    str(nodes) if nodes.exists() else str(path),
                    str(nets) if nets.exists() else str(path),
                    str(pl) if pl.exists() else None,
                    str(scl) if scl.exists() else None,
                    design_name=stem,
                )

        elif fmt == 'lefdef':
            from .lefdef_parser import LEFDEFParser
            parser = LEFDEFParser()
            if path.suffix == '.def':
                # Look for .lef in same directory
                lef_files = list(path.parent.glob("*.lef"))
                lef_path = str(lef_files[0]) if lef_files else None
                return parser.parse(lef_path=lef_path, def_path=str(path))
            else:
                return parser.parse(lef_path=str(path))

        elif fmt == 'blif':
            from .blif_parser import BLIFParser
            parser = BLIFParser()
            return parser.parse(str(path))

        raise ValueError(f"Unknown benchmark format: {fmt}")

    def _compute_metrics(self, circuit: CircuitParticles,
                         result: PlacementResult,
                         runtime: float,
                         design_name: str) -> BenchmarkMetrics:
        """Compute standard placement quality metrics."""
        # HPWL (Half-Perimeter Wire Length)
        hpwl = self._compute_hpwl(circuit)

        # Density overflow
        max_density, overflow = self._compute_density(circuit)

        return BenchmarkMetrics(
            design_name=design_name,
            num_cells=len(circuit.particles),
            num_nets=len(circuit.connections),
            die_width=circuit.die_width,
            die_height=circuit.die_height,
            hpwl=hpwl,
            total_wirelength=result.total_wirelength,
            density_overflow=overflow,
            max_density=max_density,
            runtime_s=runtime,
            total_energy=result.total_energy,
            wire_energy=result.wire_energy,
            thermal_energy=result.thermal_energy,
        )

    def _compute_hpwl(self, circuit: CircuitParticles) -> float:
        """Compute Half-Perimeter Wire Length (standard metric)."""
        # Group connections by net name to form hyperedge bounding boxes
        nets: Dict[str, List[int]] = {}
        for conn in circuit.connections:
            net_name = conn.name or f"auto_{conn.source_id}"
            if net_name not in nets:
                nets[net_name] = set()
            nets[net_name].add(conn.source_id)
            nets[net_name].add(conn.dest_id)

        total_hpwl = 0.0
        for net_name, cell_ids in nets.items():
            if len(cell_ids) < 2:
                continue
            xs = []
            ys = []
            for cid in cell_ids:
                if cid in circuit.particles:
                    p = circuit.particles[cid]
                    xs.append(p.x)
                    ys.append(p.y)
            if xs:
                total_hpwl += (max(xs) - min(xs)) + (max(ys) - min(ys))

        return total_hpwl

    def _compute_density(self, circuit: CircuitParticles,
                         grid_size: int = 50) -> Tuple[float, float]:
        """Compute density overflow using grid-based binning."""
        if not circuit.particles:
            return 0.0, 0.0

        bin_w = circuit.die_width / grid_size
        bin_h = circuit.die_height / grid_size
        bin_area = bin_w * bin_h
        target = 0.8  # 80% utilization target

        bins = np.zeros((grid_size, grid_size))
        for p in circuit.particles.values():
            col = int(np.clip(p.x / bin_w, 0, grid_size - 1))
            row = int(np.clip(p.y / bin_h, 0, grid_size - 1))
            bins[row, col] += p.area_um2

        utilization = bins / bin_area
        max_density = float(np.max(utilization))
        overflow = float(np.sum(np.maximum(utilization - target, 0))) / (grid_size * grid_size)

        return max_density, overflow

    @staticmethod
    def print_results_table(metrics_list: List[BenchmarkMetrics]) -> None:
        """Print results as a formatted table."""
        print(f"\n{'='*80}")
        print(f"{'Design':<20} | {'Cells':>8} | {'HPWL':>12} | {'Time(s)':>7} | {'Density':>7}")
        print(f"{'-'*80}")
        for m in metrics_list:
            print(f"{m.design_name:<20} | {m.num_cells:>8,} | "
                  f"{m.hpwl:>12,.0f} | {m.runtime_s:>7.1f} | {m.max_density:>7.1%}")
        print(f"{'='*80}")

    @staticmethod
    def save_results(metrics_list: List[BenchmarkMetrics], filepath: str) -> None:
        """Save results to JSON."""
        data = [m.to_dict() for m in metrics_list]
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
