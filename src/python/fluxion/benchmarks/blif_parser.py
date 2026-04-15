"""
FLUXION BLIF Parser — IWLS Benchmark Support

Parses Berkeley Logic Interchange Format (BLIF) files used in
IWLS (International Workshop on Logic & Synthesis) benchmarks.

BLIF represents combinational and sequential logic as:
- .model: Top-level module name
- .inputs/.outputs: Primary I/O names
- .names: Combinational logic blocks (truth-table)
- .latch: Sequential elements (flip-flops)
- .subckt: Hierarchical sub-circuit instances
"""

import re
import numpy as np
from pathlib import Path
from typing import Dict, List, Set, Optional

from ..particle_system import CircuitParticles, FluxionParticle, FluxionConnection


class BLIFParser:
    """
    Parser for BLIF format (IWLS benchmarks).

    Usage:
        parser = BLIFParser()
        circuit = parser.parse("path/to/design.blif")
    """

    def __init__(self):
        self.model_name: str = "blif_design"
        self.inputs: List[str] = []
        self.outputs: List[str] = []
        self.gates: Dict[str, dict] = {}
        self.latches: Dict[str, dict] = []
        self.signal_drivers: Dict[str, str] = {}  # signal -> gate_name
        self.signal_sinks: Dict[str, List[str]] = {}  # signal -> [gate_names]

    def parse(self, blif_path: str, die_scale: float = 10.0) -> CircuitParticles:
        """
        Parse a BLIF file into CircuitParticles.

        Args:
            blif_path: Path to .blif file
            die_scale: Scale factor for die dimensions (sqrt(gates) * die_scale)

        Returns:
            CircuitParticles ready for FLUXION optimization
        """
        self._parse_blif(Path(blif_path))
        return self._to_circuit_particles(die_scale)

    def _parse_blif(self, path: Path) -> None:
        """Parse the BLIF file."""
        with open(path, 'r') as f:
            lines = f.readlines()

        # Join continuation lines (ending with \)
        joined_lines = []
        buf = ""
        for line in lines:
            line = line.rstrip('\n').rstrip('\r')
            if line.endswith('\\'):
                buf += line[:-1] + " "
            else:
                buf += line
                joined_lines.append(buf.strip())
                buf = ""

        gate_id = 0
        i = 0
        while i < len(joined_lines):
            line = joined_lines[i]

            if not line or line.startswith('#'):
                i += 1
                continue

            if line.startswith('.model'):
                parts = line.split()
                if len(parts) > 1:
                    self.model_name = parts[1]

            elif line.startswith('.inputs'):
                self.inputs = line.split()[1:]

            elif line.startswith('.outputs'):
                self.outputs = line.split()[1:]

            elif line.startswith('.names'):
                # Combinational logic: ".names in1 in2 ... out"
                signals = line.split()[1:]
                if signals:
                    output_signal = signals[-1]
                    input_signals = signals[:-1]

                    gate_name = f"g_{gate_id}"
                    gate_id += 1

                    # Count truth table rows (until next directive)
                    num_rows = 0
                    j = i + 1
                    while j < len(joined_lines):
                        tline = joined_lines[j]
                        if tline.startswith('.') or not tline:
                            break
                        num_rows += 1
                        j += 1

                    # Classify gate type by input count
                    n_inputs = len(input_signals)
                    if n_inputs == 0:
                        gate_type = "BUF"
                    elif n_inputs == 1:
                        gate_type = "INV" if num_rows == 1 else "BUF"
                    elif n_inputs == 2:
                        gate_type = "NAND"
                    elif n_inputs == 3:
                        gate_type = "AND"
                    else:
                        gate_type = "MUX"

                    self.gates[gate_name] = {
                        'name': gate_name,
                        'type': gate_type,
                        'inputs': input_signals,
                        'output': output_signal,
                        'n_inputs': n_inputs,
                    }

                    # Track signal connectivity
                    self.signal_drivers[output_signal] = gate_name
                    for sig in input_signals:
                        if sig not in self.signal_sinks:
                            self.signal_sinks[sig] = []
                        self.signal_sinks[sig].append(gate_name)

            elif line.startswith('.latch'):
                # Sequential: ".latch input output [type] [clk] [init]"
                parts = line.split()
                if len(parts) >= 3:
                    latch_input = parts[1]
                    latch_output = parts[2]

                    gate_name = f"latch_{gate_id}"
                    gate_id += 1

                    self.gates[gate_name] = {
                        'name': gate_name,
                        'type': 'DFF',
                        'inputs': [latch_input],
                        'output': latch_output,
                        'n_inputs': 1,
                    }

                    self.signal_drivers[latch_output] = gate_name
                    if latch_input not in self.signal_sinks:
                        self.signal_sinks[latch_input] = []
                    self.signal_sinks[latch_input].append(gate_name)

            elif line.startswith('.subckt'):
                # Hierarchical sub-circuit
                parts = line.split()
                if len(parts) >= 2:
                    subckt_type = parts[1]
                    gate_name = f"subckt_{gate_id}"
                    gate_id += 1

                    # Parse pin=signal pairs
                    input_sigs = []
                    output_sig = None
                    for pair in parts[2:]:
                        if '=' in pair:
                            pin, sig = pair.split('=', 1)
                            if pin.upper().startswith('O') or pin.upper() == 'Y':
                                output_sig = sig
                            else:
                                input_sigs.append(sig)

                    if output_sig is None:
                        output_sig = f"_subckt_out_{gate_id}"

                    self.gates[gate_name] = {
                        'name': gate_name,
                        'type': subckt_type.upper()[:3] if len(subckt_type) >= 3 else "AND",
                        'inputs': input_sigs,
                        'output': output_sig,
                        'n_inputs': len(input_sigs),
                    }

                    self.signal_drivers[output_sig] = gate_name
                    for sig in input_sigs:
                        if sig not in self.signal_sinks:
                            self.signal_sinks[sig] = []
                        self.signal_sinks[sig].append(gate_name)

            i += 1

    def _to_circuit_particles(self, die_scale: float) -> CircuitParticles:
        """Convert parsed BLIF to CircuitParticles."""
        n_gates = len(self.gates)
        die_size = max(np.sqrt(n_gates) * die_scale, 100.0)

        circuit = CircuitParticles(
            module_name=self.model_name,
            die_width=die_size,
            die_height=die_size,
        )

        # Gate type → area/delay estimates
        type_params = {
            'INV': (1.5, 3.0, 2.0),   # (area, power, delay)
            'BUF': (2.0, 4.0, 3.0),
            'NAND': (3.0, 6.0, 5.0),
            'NOR': (3.0, 6.0, 5.0),
            'AND': (4.0, 7.0, 7.0),
            'OR': (4.0, 7.0, 7.0),
            'XOR': (5.0, 9.0, 8.0),
            'MUX': (6.0, 10.0, 10.0),
            'DFF': (10.0, 30.0, 40.0),
        }

        # Create particles
        name_to_id = {}
        for idx, (gate_name, gate) in enumerate(self.gates.items()):
            name_to_id[gate_name] = idx
            gtype = gate['type']
            area, power, delay = type_params.get(gtype, (4.0, 7.0, 7.0))

            # Scale by input count for complex gates
            if gate['n_inputs'] > 3:
                scale = 1.0 + 0.2 * (gate['n_inputs'] - 3)
                area *= scale
                delay *= scale

            particle = FluxionParticle(
                id=idx,
                name=gate_name,
                type=gtype,
                x=np.random.uniform(0, die_size),
                y=np.random.uniform(0, die_size),
                power_pw=power,
                area_um2=area,
                delay_ps=delay,
            )
            circuit.add_particle(particle)

        # Create connections: for each signal, connect driver → sinks
        conn_id = 0
        for signal, driver_name in self.signal_drivers.items():
            if driver_name not in name_to_id:
                continue
            driver_id = name_to_id[driver_name]

            sinks = self.signal_sinks.get(signal, [])
            for sink_name in sinks:
                if sink_name not in name_to_id:
                    continue
                sink_id = name_to_id[sink_name]
                if driver_id != sink_id:
                    conn = FluxionConnection(
                        source_id=driver_id,
                        dest_id=sink_id,
                        name=f"sig_{signal}",
                    )
                    circuit.add_connection(conn)
                    conn_id += 1

        return circuit
