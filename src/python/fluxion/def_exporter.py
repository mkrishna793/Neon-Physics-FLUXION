"""
FLUXION DEF Exporter

Converts internal particle representations (with continuous/float coordinates)
into the industry-standard ASCII DEF (Design Exchange Format).

Foundry tools require discrete integer coordinates based on Database Units (DBU).
The DEF exporter handles formatting, scaling, alignment, and grid adherence.
"""

import os
from typing import TextIO, Dict
import numpy as np

from .particle_system import CircuitParticles
from .lef_library import LEFLibrary


class DEFExporter:
    """
    Exports a FLUXION CircuitParticles object to DEF format.
    """
    
    def __init__(self, dbu_per_micron: int = 1000):
        """
        Initialize DEF exporter.
        
        Args:
            dbu_per_micron: Database units per micrometer (standard is 1000 or 2000)
        """
        self.dbu = dbu_per_micron
        self.lef_lib = LEFLibrary(dbu_per_micron=self.dbu)
        
    def _to_dbu(self, microns: float) -> int:
        """Convert float microns to integer DBU."""
        return int(round(microns * self.dbu))

    def export(self, circuit: CircuitParticles, filepath: str) -> None:
        """
        Export circuit to DEF file.
        
        Args:
            circuit: The placed circuit
            filepath: Destination .def path
        """
        with open(filepath, 'w') as f:
            self._write_header(f, circuit)
            self._write_die_area(f, circuit)
            self._write_components(f, circuit)
            self._write_nets(f, circuit)
            f.write("END DESIGN\n")

    def _write_header(self, f: TextIO, circuit: CircuitParticles) -> None:
        f.write(f"VERSION 5.8 ;\n")
        f.write(f"DIVIDERCHAR \"/\" ;\n")
        f.write(f"BUSBITCHARS \"[]\" ;\n")
        f.write(f"DESIGN {circuit.module_name} ;\n")
        f.write(f"UNITS DISTANCE MICRONS {self.dbu} ;\n\n")

    def _write_die_area(self, f: TextIO, circuit: CircuitParticles) -> None:
        width_dbu = self._to_dbu(circuit.die_width)
        height_dbu = self._to_dbu(circuit.die_height)
        f.write(f"DIEAREA ( 0 0 ) ( {width_dbu} {height_dbu} ) ;\n\n")

    def _write_components(self, f: TextIO, circuit: CircuitParticles) -> None:
        f.write(f"COMPONENTS {len(circuit.particles)} ;\n")
        
        for p in circuit.particles.values():
            # Translate raw type to LEF MACRO type (NAND -> NAND2_X1, etc.)
            macro_type = self.lef_lib.get_macro_name(p.type)
            
            x_dbu = self._to_dbu(p.x)
            y_dbu = self._to_dbu(p.y)
            
            # Using PLACED instead of FIXED since routing still needs to happen
            f.write(f"  - {p.name} {macro_type} + PLACED ( {x_dbu} {y_dbu} ) N ;\n")
            
        f.write("END COMPONENTS\n\n")

    def _write_nets(self, f: TextIO, circuit: CircuitParticles) -> None:
        # Group connections by source (driver) to form logical nets
        # In a real netlist, a net has 1 driver and N sinks
        nets: Dict[int, list] = {}
        
        for conn in circuit.connections:
            if conn.source_id not in nets:
                nets[conn.source_id] = []
            nets[conn.source_id].append(conn.dest_id)
            
        f.write(f"NETS {len(nets)} ;\n")
        
        for i, (src_id, dest_ids) in enumerate(nets.items()):
            src = circuit.particles[src_id]
            
            # Use provided name or auto-generate
            # Try to grab a connection name from the first dest if available
            conns_for_src = [c for c in circuit.connections if c.source_id == src_id]
            net_name = conns_for_src[0].name if conns_for_src and conns_for_src[0].name else f"net_{src.name}_OUT"
            if not net_name:
                net_name = f"net_auto_{i}"
                
            f.write(f"  - {net_name}\n")
            
            # Driver pin (assumed generic 'Y' for output)
            f.write(f"      ( {src.name} Y )")
            
            # Sink pins (assumed generic 'A', 'B' etc. We'll just use 'A' for simplicity in demo)
            for dest_id in dest_ids:
                dest = circuit.particles[dest_id]
                f.write(f"\n      ( {dest.name} A )")
                
            f.write(" ;\n")
            
        f.write("END NETS\n\n")


def export_def(circuit: CircuitParticles, filepath: str, dbu: int = 1000) -> None:
    """Convenience function to export circuit to DEF."""
    exporter = DEFExporter(dbu_per_micron=dbu)
    exporter.export(circuit, filepath)
