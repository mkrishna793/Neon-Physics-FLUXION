import pytest
import numpy as np
from fluxion.particle_system import CircuitParticles, FluxionParticle, FluxionConnection
from fluxion.force_congestion import CongestionAwareForce

def test_congestion_force_calculation():
    # Create a simple circuit
    circuit = CircuitParticles(die_width=100.0, die_height=100.0)

    # Add two particles
    p1 = FluxionParticle(id=0, name="g1", type="NAND", x=10, y=10, area_um2=10)
    p2 = FluxionParticle(id=1, name="g2", type="NAND", x=90, y=90, area_um2=10)
    circuit.add_particle(p1)
    circuit.add_particle(p2)

    # Add a connection between them
    conn = FluxionConnection(source_id=0, dest_id=1)
    circuit.add_connection(conn)

    # Create force with low track limit to trigger it
    force = CongestionAwareForce(weight=1.0, grid_size=10, max_tracks_per_bin=0.01)

    result = force.calculate(circuit)

    assert result.energy >= 0
    assert result.forces.shape == (2, 2)
    assert 'max_congestion' in result.force_details

if __name__ == "__main__":
    test_congestion_force_calculation()
