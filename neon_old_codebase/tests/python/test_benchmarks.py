"""
Tests for FLUXION benchmark parsers.

Tests each parser with small inline benchmark data.
"""

import sys
import os
import tempfile
import pytest
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "python"))

from fluxion.benchmarks.bookshelf_parser import BookshelfParser
from fluxion.benchmarks.blif_parser import BLIFParser
from fluxion.benchmarks.lefdef_parser import LEFDEFParser
from fluxion.benchmarks.benchmark_runner import BenchmarkRunner, BenchmarkMetrics


class TestBookshelfParser:
    """Tests for ISPD 2005/2006 Bookshelf parser."""

    def _create_sample(self, tmpdir):
        """Create minimal Bookshelf files in tmpdir."""
        # .aux
        aux = tmpdir / "test.aux"
        aux.write_text("RowBasedPlacement : test.nodes test.nets test.pl test.scl\n")

        # .nodes
        nodes = tmpdir / "test.nodes"
        nodes.write_text(
            "UCLA nodes 1.0\n"
            "NumNodes : 4\n"
            "NumTerminals : 0\n"
            "cell_0\t4\t4\n"
            "cell_1\t6\t4\n"
            "cell_2\t4\t4\n"
            "cell_3\t8\t4\n"
        )

        # .nets
        nets = tmpdir / "test.nets"
        nets.write_text(
            "UCLA nets 1.0\n"
            "NumNets : 3\n"
            "NumPins : 6\n"
            "NetDegree : 2 n0\n"
            "cell_0 O\n"
            "cell_1 I\n"
            "NetDegree : 2 n1\n"
            "cell_1 O\n"
            "cell_2 I\n"
            "NetDegree : 2 n2\n"
            "cell_2 O\n"
            "cell_3 I\n"
        )

        # .pl
        pl = tmpdir / "test.pl"
        pl.write_text(
            "UCLA pl 1.0\n"
            "cell_0\t10\t20\t: N\n"
            "cell_1\t30\t20\t: N\n"
            "cell_2\t50\t20\t: N\n"
            "cell_3\t70\t20\t: N\n"
        )

        # .scl
        scl = tmpdir / "test.scl"
        scl.write_text(
            "UCLA scl 1.0\n"
            "NumRows : 2\n"
            "CoreRow Horizontal\n"
            "  Coordinate : 0\n"
            "  Height : 4\n"
            "  Sitewidth : 1\n"
            "  NumSites : 100\n"
            "  SubrowOrigin : 0\n"
            "End\n"
            "CoreRow Horizontal\n"
            "  Coordinate : 4\n"
            "  Height : 4\n"
            "  Sitewidth : 1\n"
            "  NumSites : 100\n"
            "  SubrowOrigin : 0\n"
            "End\n"
        )
        return str(aux)

    def test_parse_bookshelf(self, tmp_path):
        """Parse a minimal Bookshelf benchmark."""
        aux_path = self._create_sample(tmp_path)
        parser = BookshelfParser()
        circuit = parser.parse(aux_path)

        assert len(circuit.particles) == 4
        assert len(circuit.connections) == 3
        assert circuit.die_width > 0
        assert circuit.die_height > 0

    def test_parse_positions(self, tmp_path):
        """Check that parsed positions match .pl file."""
        aux_path = self._create_sample(tmp_path)
        parser = BookshelfParser()
        circuit = parser.parse(aux_path)

        # cell_0 should be at (10, 20)
        p0 = circuit.particles[0]
        assert abs(p0.x - 10.0) < 0.01
        assert abs(p0.y - 20.0) < 0.01

    def test_parse_connectivity(self, tmp_path):
        """Verify net connectivity is correct."""
        aux_path = self._create_sample(tmp_path)
        parser = BookshelfParser()
        circuit = parser.parse(aux_path)

        # Check that connections form a chain: 0→1→2→3
        sources = {c.source_id for c in circuit.connections}
        dests = {c.dest_id for c in circuit.connections}
        assert 0 in sources  # cell_0 drives
        assert 3 in dests    # cell_3 is a sink


class TestBLIFParser:
    """Tests for IWLS BLIF parser."""

    def _create_sample(self, tmpdir):
        blif = tmpdir / "test.blif"
        blif.write_text(
            ".model test_circuit\n"
            ".inputs a b c\n"
            ".outputs y\n"
            "\n"
            ".names a b w0\n"
            "11 1\n"
            "\n"
            ".names b c w1\n"
            "11 1\n"
            "\n"
            ".names w0 w1 y\n"
            "1- 1\n"
            "-1 1\n"
            "\n"
            ".end\n"
        )
        return str(blif)

    def test_parse_blif(self, tmp_path):
        """Parse a minimal BLIF file."""
        blif_path = self._create_sample(tmp_path)
        parser = BLIFParser()
        circuit = parser.parse(blif_path)

        assert len(circuit.particles) == 3  # 3 .names blocks
        assert len(circuit.connections) > 0
        assert circuit.module_name == "test_circuit"

    def test_gate_types(self, tmp_path):
        """Verify gate type classification."""
        blif_path = self._create_sample(tmp_path)
        parser = BLIFParser()
        circuit = parser.parse(blif_path)

        # 2-input gates should be NAND type
        types = {p.type for p in circuit.particles.values()}
        assert "NAND" in types

    def test_latch_parsing(self, tmp_path):
        """Test sequential element parsing."""
        blif = tmp_path / "seq.blif"
        blif.write_text(
            ".model seq_test\n"
            ".inputs d clk\n"
            ".outputs q\n"
            "\n"
            ".names d w0\n"
            "1 1\n"
            "\n"
            ".latch w0 q re clk 0\n"
            "\n"
            ".end\n"
        )

        parser = BLIFParser()
        circuit = parser.parse(str(blif))

        types = {p.type for p in circuit.particles.values()}
        assert "DFF" in types


class TestLEFDEFParser:
    """Tests for ICCAD LEF/DEF parser."""

    def _create_sample_def(self, tmpdir):
        def_file = tmpdir / "test.def"
        def_file.write_text(
            "VERSION 5.8 ;\n"
            "UNITS DISTANCE MICRONS 1000 ;\n"
            "DESIGN test_design ;\n"
            "DIEAREA ( 0 0 ) ( 100000 100000 ) ;\n"
            "\n"
            "COMPONENTS 3 ;\n"
            "  - inst0 NAND2_X1 + PLACED ( 10000 20000 ) N ;\n"
            "  - inst1 NOR2_X1 + PLACED ( 30000 20000 ) N ;\n"
            "  - inst2 DFF_X1 + PLACED ( 50000 40000 ) N ;\n"
            "END COMPONENTS\n"
            "\n"
            "NETS 2 ;\n"
            "  - net_0\n"
            "    ( inst0 Y )\n"
            "    ( inst1 A ) ;\n"
            "  - net_1\n"
            "    ( inst1 Y )\n"
            "    ( inst2 D ) ;\n"
            "END NETS\n"
            "\n"
            "END DESIGN\n"
        )
        return str(def_file)

    def test_parse_def(self, tmp_path):
        """Parse a minimal DEF file."""
        def_path = self._create_sample_def(tmp_path)
        parser = LEFDEFParser()
        circuit = parser.parse(def_path=def_path)

        assert len(circuit.particles) == 3
        assert len(circuit.connections) == 2
        assert circuit.die_width == 100.0
        assert circuit.die_height == 100.0

    def test_parse_positions_def(self, tmp_path):
        """Verify positions are converted from DBU to microns."""
        def_path = self._create_sample_def(tmp_path)
        parser = LEFDEFParser()
        circuit = parser.parse(def_path=def_path)

        p0 = circuit.particles[0]
        assert abs(p0.x - 10.0) < 0.01
        assert abs(p0.y - 20.0) < 0.01


class TestBenchmarkMetrics:
    """Tests for metric computation."""

    def test_hpwl_calculation(self):
        """Test HPWL calculation on simple circuit."""
        from fluxion.particle_system import CircuitParticles, FluxionParticle, FluxionConnection

        circuit = CircuitParticles(
            module_name="test", die_width=100, die_height=100,
        )
        circuit.add_particle(FluxionParticle(id=0, name="a", type="NAND",
                                              x=0, y=0, power_pw=5, area_um2=2))
        circuit.add_particle(FluxionParticle(id=1, name="b", type="NAND",
                                              x=10, y=20, power_pw=5, area_um2=2))
        circuit.add_connection(FluxionConnection(source_id=0, dest_id=1, name="n0"))

        runner = BenchmarkRunner(steps=10, verbose=False)
        hpwl = runner._compute_hpwl(circuit)
        # HPWL = |10-0| + |20-0| = 30
        assert abs(hpwl - 30.0) < 0.01

    def test_metrics_summary(self):
        """Test metrics formatting."""
        m = BenchmarkMetrics(
            design_name="test_design",
            num_cells=100,
            num_nets=200,
            die_width=1000,
            die_height=1000,
            hpwl=50000,
            runtime_s=1.5,
            max_density=0.85,
        )
        line = m.summary_line()
        assert "test_design" in line
        assert "100" in line
