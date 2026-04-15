"""
FLUXION Bookshelf Parser — ISPD 2005/2006 Benchmark Support

Parses GSRC Bookshelf format (.aux, .nodes, .nets, .pl, .scl)
into FLUXION's CircuitParticles representation.

File format reference:
- .aux:   Master file listing all other design files
- .nodes: Cell definitions with dimensions (terminal/non-terminal)
- .nets:  Hypergraph netlist (one net = 1 driver + N sinks)
- .pl:    Cell placements (x, y) and orientation
- .scl:   Row definitions (site width, row height, positions)
"""

import os
import re
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional

from ..particle_system import CircuitParticles, FluxionParticle, FluxionConnection


class BookshelfParser:
    """
    Parser for GSRC Bookshelf format (ISPD 2005/2006 benchmarks).

    Usage:
        parser = BookshelfParser()
        circuit = parser.parse("path/to/design.aux")
    """

    def __init__(self):
        self.nodes: Dict[str, dict] = {}
        self.nets: List[dict] = []
        self.placements: Dict[str, dict] = {}
        self.rows: List[dict] = []
        self.die_width = 0.0
        self.die_height = 0.0

    def parse(self, aux_path: str) -> CircuitParticles:
        """
        Parse a Bookshelf benchmark from its .aux file.

        Args:
            aux_path: Path to the .aux master file

        Returns:
            CircuitParticles ready for FLUXION optimization
        """
        aux_path = Path(aux_path)
        base_dir = aux_path.parent

        # 1. Parse .aux to find all related files
        files = self._parse_aux(aux_path)

        # 2. Parse components
        if 'nodes' in files:
            self._parse_nodes(base_dir / files['nodes'])
        if 'nets' in files:
            self._parse_nets(base_dir / files['nets'])
        if 'pl' in files:
            self._parse_pl(base_dir / files['pl'])
        if 'scl' in files:
            self._parse_scl(base_dir / files['scl'])

        # 3. Convert to CircuitParticles
        return self._to_circuit_particles(aux_path.stem)

    def parse_from_files(self, nodes_path: str, nets_path: str,
                         pl_path: str = None, scl_path: str = None,
                         design_name: str = "benchmark") -> CircuitParticles:
        """Parse directly from individual files without .aux."""
        self._parse_nodes(Path(nodes_path))
        self._parse_nets(Path(nets_path))
        if pl_path:
            self._parse_pl(Path(pl_path))
        if scl_path:
            self._parse_scl(Path(scl_path))
        return self._to_circuit_particles(design_name)

    def _parse_aux(self, aux_path: Path) -> Dict[str, str]:
        """Parse .aux file to extract filenames."""
        files = {}
        with open(aux_path, 'r') as f:
            for line in f:
                line = line.strip()
                if ':' in line:
                    # Format: "RowBasedPlacement : file1.nodes file2.nets ..."
                    _, file_list = line.split(':', 1)
                    for fname in file_list.strip().split():
                        ext = Path(fname).suffix.lstrip('.')
                        files[ext] = fname
        return files

    def _parse_nodes(self, path: Path) -> None:
        """Parse .nodes file — cell names and dimensions."""
        with open(path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#') or line.startswith('UCLA'):
                    continue
                if line.startswith('NumNodes') or line.startswith('NumTerminals'):
                    continue

                parts = line.split()
                if len(parts) >= 3:
                    name = parts[0]
                    try:
                        width = float(parts[1])
                        height = float(parts[2])
                    except ValueError:
                        continue

                    is_fixed = 'terminal' in line.lower()

                    self.nodes[name] = {
                        'name': name,
                        'width': width,
                        'height': height,
                        'area': width * height,
                        'fixed': is_fixed,
                    }

    def _parse_nets(self, path: Path) -> None:
        """Parse .nets file — hypergraph connectivity."""
        current_net = None
        with open(path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#') or line.startswith('UCLA'):
                    continue
                if line.startswith('NumNets') or line.startswith('NumPins'):
                    continue

                if line.startswith('NetDegree'):
                    # "NetDegree : N  netName"
                    parts = line.split()
                    degree = int(parts[2])
                    net_name = parts[3] if len(parts) > 3 else f"net_{len(self.nets)}"
                    current_net = {
                        'name': net_name,
                        'degree': degree,
                        'pins': [],
                    }
                    self.nets.append(current_net)
                elif current_net is not None:
                    # "cellName  direction  : offsetX offsetY"
                    parts = line.split()
                    if parts:
                        cell_name = parts[0]
                        direction = parts[1] if len(parts) > 1 else 'I'
                        current_net['pins'].append({
                            'cell': cell_name,
                            'direction': direction,
                        })

    def _parse_pl(self, path: Path) -> None:
        """Parse .pl file — cell placements."""
        with open(path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#') or line.startswith('UCLA'):
                    continue

                parts = line.split()
                if len(parts) >= 3:
                    name = parts[0]
                    try:
                        x = float(parts[1])
                        y = float(parts[2])
                    except ValueError:
                        continue

                    orient = parts[4] if len(parts) > 4 else 'N'
                    fixed = '/FIXED' in line

                    self.placements[name] = {
                        'x': x, 'y': y,
                        'orient': orient,
                        'fixed': fixed,
                    }

    def _parse_scl(self, path: Path) -> None:
        """Parse .scl file — row definitions."""
        current_row = None
        with open(path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#') or line.startswith('UCLA'):
                    continue
                if line.startswith('NumRows'):
                    continue

                if line.startswith('CoreRow'):
                    current_row = {}
                elif line == 'End':
                    if current_row:
                        self.rows.append(current_row)
                    current_row = None
                elif current_row is not None:
                    parts = line.split()
                    if len(parts) >= 3 and parts[1] == ':':
                        key = parts[0]
                        val = parts[2]
                        try:
                            current_row[key] = float(val)
                        except ValueError:
                            current_row[key] = val
                    elif len(parts) >= 2:
                        key = parts[0].rstrip(':')
                        val = parts[1]
                        try:
                            current_row[key] = float(val)
                        except ValueError:
                            current_row[key] = val

        # Compute die dimensions from rows
        if self.rows:
            max_y = max(float(r.get('Coordinate', 0)) + float(r.get('Height', 0)) for r in self.rows)
            max_x = max(
                float(r.get('SubrowOrigin', 0)) + float(r.get('NumSites', 0)) * float(r.get('Sitewidth', 1))
                for r in self.rows
            )
            self.die_height = max_y
            self.die_width = max_x

    def _to_circuit_particles(self, design_name: str) -> CircuitParticles:
        """Convert parsed data into a CircuitParticles object."""
        # Auto-detect die size from node positions if not set by .scl
        if self.die_width == 0 and self.placements:
            xs = [p['x'] for p in self.placements.values()]
            ys = [p['y'] for p in self.placements.values()]
            self.die_width = max(xs) * 1.2 if xs else 1000.0
            self.die_height = max(ys) * 1.2 if ys else 1000.0
        elif self.die_width == 0:
            # Estimate from total area
            total_area = sum(n['area'] for n in self.nodes.values())
            side = np.sqrt(total_area) * 2.0
            self.die_width = side
            self.die_height = side

        # Scale: Bookshelf uses DBU, convert to microns (assume 1 DBU = 1 unit)
        # For ISPD benchmarks, dimensions are already in site units
        circuit = CircuitParticles(
            module_name=design_name,
            die_width=self.die_width,
            die_height=self.die_height,
        )

        # Create name → ID mapping
        name_to_id = {}
        for idx, (name, node) in enumerate(self.nodes.items()):
            name_to_id[name] = idx

            # Get placement if available
            pl = self.placements.get(name, {})
            x = pl.get('x', np.random.uniform(0, self.die_width))
            y = pl.get('y', np.random.uniform(0, self.die_height))

            # Estimate gate type from size
            gate_type = "NAND"
            if node['area'] > 500:
                gate_type = "DFF"
            elif node['area'] > 100:
                gate_type = "MUX"

            particle = FluxionParticle(
                id=idx,
                name=name,
                type=gate_type,
                x=x,
                y=y,
                power_pw=np.sqrt(node['area']) * 2.0,  # Estimate power from area
                area_um2=node['area'],
                delay_ps=np.sqrt(node['area']) * 1.5,   # Estimate delay
            )
            particle.fixed = node.get('fixed', False) or pl.get('fixed', False)
            circuit.add_particle(particle)

        # Create connections from nets
        conn_id = 0
        for net in self.nets:
            pins = net['pins']
            if len(pins) < 2:
                continue

            # First output pin is the driver; rest are sinks
            driver = None
            sinks = []
            for pin in pins:
                cell = pin['cell']
                if cell not in name_to_id:
                    continue
                if pin['direction'] == 'O' and driver is None:
                    driver = name_to_id[cell]
                else:
                    sinks.append(name_to_id[cell])

            # If no explicit driver, use first pin
            if driver is None and sinks:
                driver = sinks.pop(0)

            if driver is not None:
                for sink in sinks:
                    if driver != sink:
                        conn = FluxionConnection(
                            source_id=driver,
                            dest_id=sink,
                            name=net['name'],
                        )
                        circuit.add_connection(conn)
                        conn_id += 1

        return circuit
