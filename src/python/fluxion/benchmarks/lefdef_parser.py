"""
FLUXION LEF/DEF Parser — ICCAD 2014/2015 Benchmark Support

Lightweight parser for LEF (Library Exchange Format) and DEF
(Design Exchange Format) files used in ICCAD placement contests.

This is an academic parser sufficient to extract:
- Cell macros and dimensions from LEF
- Component instances and netlist from DEF
- Die area and row definitions

NOT a full production LEF/DEF parser — it handles the subset
needed for placement benchmarks.
"""

import re
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from ..particle_system import CircuitParticles, FluxionParticle, FluxionConnection


class LEFDEFParser:
    """
    Parser for LEF/DEF format (ICCAD 2014/2015 benchmarks).

    Usage:
        parser = LEFDEFParser()
        circuit = parser.parse(lef_path="tech.lef", def_path="design.def")
    """

    def __init__(self):
        self.macros: Dict[str, dict] = {}
        self.components: Dict[str, dict] = {}
        self.nets: Dict[str, dict] = {}
        self.die_area: Tuple[float, float, float, float] = (0, 0, 0, 0)
        self.dbu_per_micron: int = 1000
        self.rows: List[dict] = []

    def parse(self, lef_path: str = None, def_path: str = None,
              design_name: str = None) -> CircuitParticles:
        """
        Parse LEF and/or DEF files.

        Args:
            lef_path: Path to .lef file (cell definitions)
            def_path: Path to .def file (design netlist + placement)
            design_name: Override design name

        Returns:
            CircuitParticles ready for FLUXION optimization
        """
        if lef_path:
            self._parse_lef(Path(lef_path))
        if def_path:
            self._parse_def(Path(def_path))

        name = design_name or "lef_def_design"
        return self._to_circuit_particles(name)

    def _parse_lef(self, path: Path) -> None:
        """Parse LEF file for macro definitions."""
        with open(path, 'r') as f:
            content = f.read()

        # Extract MACRO blocks
        macro_pattern = re.compile(
            r'MACRO\s+(\S+)(.*?)END\s+\1',
            re.DOTALL
        )

        for match in macro_pattern.finditer(content):
            name = match.group(1)
            body = match.group(2)

            macro = {'name': name, 'width': 0.0, 'height': 0.0, 'pins': []}

            # Extract SIZE
            size_match = re.search(r'SIZE\s+([\d.]+)\s+BY\s+([\d.]+)', body)
            if size_match:
                macro['width'] = float(size_match.group(1))
                macro['height'] = float(size_match.group(2))

            # Extract PIN names
            pin_matches = re.findall(r'PIN\s+(\S+)', body)
            macro['pins'] = pin_matches

            self.macros[name] = macro

    def _parse_def(self, path: Path) -> None:
        """Parse DEF file for components, nets, and die area."""
        with open(path, 'r') as f:
            lines = f.readlines()

        section = None  # 'COMPONENTS', 'NETS', etc.
        current_net = None

        for line in lines:
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            # Units
            units_match = re.match(r'UNITS\s+DISTANCE\s+MICRONS\s+(\d+)', line)
            if units_match:
                self.dbu_per_micron = int(units_match.group(1))
                continue

            # Die area
            die_match = re.match(
                r'DIEAREA\s+\(\s*([\d.-]+)\s+([\d.-]+)\s*\)\s*\(\s*([\d.-]+)\s+([\d.-]+)\s*\)',
                line
            )
            if die_match:
                self.die_area = (
                    float(die_match.group(1)) / self.dbu_per_micron,
                    float(die_match.group(2)) / self.dbu_per_micron,
                    float(die_match.group(3)) / self.dbu_per_micron,
                    float(die_match.group(4)) / self.dbu_per_micron,
                )
                continue

            # Section headers
            if line.startswith('COMPONENTS'):
                section = 'COMPONENTS'
                continue
            elif line.startswith('END COMPONENTS'):
                section = None
                continue
            elif line.startswith('NETS'):
                section = 'NETS'
                continue
            elif line.startswith('END NETS'):
                section = None
                current_net = None
                continue
            elif line.startswith('ROW'):
                parts = line.split()
                if len(parts) >= 6:
                    row = {
                        'name': parts[1],
                        'site': parts[2],
                        'x': float(parts[3]) / self.dbu_per_micron,
                        'y': float(parts[4]) / self.dbu_per_micron,
                    }
                    self.rows.append(row)
                continue

            # Parse sections
            if section == 'COMPONENTS':
                self._parse_component_line(line)
            elif section == 'NETS':
                self._parse_net_line(line)

    def _parse_component_line(self, line: str) -> None:
        """Parse a single COMPONENTS line."""
        # Format: "- instName macroName + PLACED ( x y ) orient ;"
        comp_match = re.match(
            r'-\s+(\S+)\s+(\S+).*?\(\s*([\d.-]+)\s+([\d.-]+)\s*\)',
            line
        )
        if comp_match:
            inst_name = comp_match.group(1)
            macro_name = comp_match.group(2)
            x = float(comp_match.group(3)) / self.dbu_per_micron
            y = float(comp_match.group(4)) / self.dbu_per_micron
            fixed = 'FIXED' in line

            self.components[inst_name] = {
                'name': inst_name,
                'macro': macro_name,
                'x': x,
                'y': y,
                'fixed': fixed,
            }

    def _parse_net_line(self, line: str) -> None:
        """Parse net definition lines."""
        if line.startswith('-'):
            # Start new net: "- netName"
            parts = line.split()
            net_name = parts[1] if len(parts) > 1 else f"net_{len(self.nets)}"
            self.nets[net_name] = {'name': net_name, 'pins': []}
        elif line.startswith('('):
            # Pin: "( instName pinName )"
            pin_match = re.match(r'\(\s*(\S+)\s+(\S+)\s*\)', line)
            if pin_match and self.nets:
                last_net = list(self.nets.values())[-1]
                last_net['pins'].append({
                    'instance': pin_match.group(1),
                    'pin': pin_match.group(2),
                })

    def _to_circuit_particles(self, design_name: str) -> CircuitParticles:
        """Convert parsed LEF/DEF data to CircuitParticles."""
        die_w = self.die_area[2] - self.die_area[0]
        die_h = self.die_area[3] - self.die_area[1]

        if die_w <= 0 or die_h <= 0:
            # Estimate from components
            if self.components:
                xs = [c['x'] for c in self.components.values()]
                ys = [c['y'] for c in self.components.values()]
                die_w = max(xs) * 1.3 if xs else 1000.0
                die_h = max(ys) * 1.3 if ys else 1000.0
            else:
                die_w, die_h = 1000.0, 1000.0

        circuit = CircuitParticles(
            module_name=design_name,
            die_width=die_w,
            die_height=die_h,
        )

        # Map instance names to IDs
        name_to_id = {}
        for idx, (inst_name, comp) in enumerate(self.components.items()):
            name_to_id[inst_name] = idx

            # Get dimensions from LEF macro
            macro = self.macros.get(comp['macro'], {})
            width = macro.get('width', 1.0)
            height = macro.get('height', 1.0)
            area = width * height

            # Map macro type to gate type
            macro_name = comp['macro'].upper()
            gate_type = "NAND"
            for t in ["DFF", "MUX", "XOR", "AND", "OR", "NOR", "INV", "BUF"]:
                if t in macro_name:
                    gate_type = t
                    break

            particle = FluxionParticle(
                id=idx,
                name=inst_name,
                type=gate_type,
                x=comp['x'],
                y=comp['y'],
                power_pw=np.sqrt(area) * 2.0,
                area_um2=area,
                delay_ps=np.sqrt(area) * 1.5,
            )
            particle.fixed = comp.get('fixed', False)
            circuit.add_particle(particle)

        # Create connections from nets
        conn_id = 0
        for net_name, net in self.nets.items():
            pins = net['pins']
            if len(pins) < 2:
                continue

            # First pin is driver
            driver_inst = pins[0]['instance']
            if driver_inst not in name_to_id:
                continue
            driver_id = name_to_id[driver_inst]

            for pin in pins[1:]:
                sink_inst = pin['instance']
                if sink_inst not in name_to_id:
                    continue
                sink_id = name_to_id[sink_inst]
                if driver_id != sink_id:
                    conn = FluxionConnection(
                        source_id=driver_id,
                        dest_id=sink_id,
                        name=net_name,
                    )
                    circuit.add_connection(conn)
                    conn_id += 1

        return circuit
