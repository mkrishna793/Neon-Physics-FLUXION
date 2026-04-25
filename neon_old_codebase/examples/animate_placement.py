#!/usr/bin/env python3
"""
FLUXION Placement Animation Generator

Creates an animated GIF showing gates moving from random positions
to optimized positions through thermodynamic annealing.

Author: N. Mohana Krishna
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import FancyBboxPatch
from matplotlib.colors import LinearSegmentedColormap
import os

# Try to import from parent
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.python.fluxion import (
    ThermodynamicPlacementEngine,
    PlacementConfig,
    CircuitParticles,
    FluxionParticle,
    FluxionConnection,
)


def create_demo_circuit(num_gates=50):
    """Create a demo circuit for animation."""
    circuit = CircuitParticles(
        module_name="demo_animation",
        die_width=100.0,
        die_height=100.0,
    )

    np.random.seed(42)

    # Create gates with realistic properties
    gate_types = ["NAND", "NOR", "AND", "OR", "XOR", "DFF", "MUX"]
    type_weights = [30, 20, 10, 10, 10, 15, 5]

    for i in range(num_gates):
        gtype = np.random.choice(gate_types, p=[w/100 for w in type_weights])

        if gtype == "DFF":
            delay = np.random.uniform(40, 80)
            area = np.random.uniform(8, 15)
            power = np.random.uniform(80, 200)
        elif gtype == "MUX":
            delay = np.random.uniform(15, 30)
            area = np.random.uniform(5, 12)
            power = np.random.uniform(15, 40)
        else:
            delay = np.random.uniform(5, 15)
            area = np.random.uniform(1.5, 4)
            power = np.random.uniform(3, 10)

        particle = FluxionParticle(
            id=i,
            name=f"{gtype.lower()}_{i}",
            type=gtype,
            power_pw=power,
            area_um2=area,
            delay_ps=delay,
        )
        circuit.add_particle(particle)

    # Create connections
    gate_ids = list(circuit.particles.keys())
    for i, src_id in enumerate(gate_ids):
        num_outputs = np.random.randint(1, 4)
        for _ in range(num_outputs):
            dst_idx = min(i + np.random.randint(1, 10), len(gate_ids) - 1)
            dst_id = gate_ids[dst_idx]
            if src_id != dst_id:
                conn = FluxionConnection(
                    source_id=src_id,
                    dest_id=dst_id,
                    name=f"net_{len(circuit.connections)}",
                    is_critical_path=(np.random.random() < 0.1),
                )
                circuit.add_connection(conn)

    return circuit


def run_annealing_with_history(circuit, num_steps=200):
    """Run annealing and return position history."""
    config = PlacementConfig(
        die_width=100.0,
        die_height=100.0,
        annealing_steps=num_steps,
        initial_temperature=100.0,
        final_temperature=0.01,
        verbose=False,
        use_gpu=False,
    )

    engine = ThermodynamicPlacementEngine(config)
    engine.set_circuit(circuit)

    # Track positions during annealing
    position_history = []
    energy_history = []
    temp_history = []

    def callback(step, temp, energy, positions):
        if step % 2 == 0:  # Record every 2 steps
            position_history.append(positions.copy())
            energy_history.append(energy)
            temp_history.append(temp)

    # Run optimization with callback
    result = engine.optimize(callback=callback)

    return position_history, energy_history, temp_history, result


def create_animation(position_history, energy_history, temp_history, circuit, output_path):
    """Create animated GIF of placement optimization."""

    # Setup figure with dark theme
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(14, 8))

    # Create grid for subplots
    gs = fig.add_gridspec(2, 3, width_ratios=[2, 1, 1], height_ratios=[1, 1])

    # Main placement view (left, spans both rows)
    ax_main = fig.add_subplot(gs[:, 0])
    ax_main.set_xlim(-5, 105)
    ax_main.set_ylim(-5, 105)
    ax_main.set_aspect('equal')
    ax_main.set_xlabel('X Position (μm)', fontsize=10, color='white')
    ax_main.set_ylabel('Y Position (μm)', fontsize=10, color='white')
    ax_main.set_title('Gate Placement', fontsize=12, color='white', fontweight='bold')

    # Draw die boundary
    die_rect = plt.Rectangle((0, 0), 100, 100, fill=False, edgecolor='#00ff88', linewidth=2, linestyle='--')
    ax_main.add_patch(die_rect)

    # Energy plot (top right)
    ax_energy = fig.add_subplot(gs[0, 1:])
    ax_energy.set_xlabel('Step', fontsize=9, color='white')
    ax_energy.set_ylabel('Energy', fontsize=9, color='white')
    ax_energy.set_title('Energy Convergence', fontsize=10, color='white', fontweight='bold')

    # Temperature plot (bottom right)
    ax_temp = fig.add_subplot(gs[1, 1:])
    ax_temp.set_xlabel('Step', fontsize=9, color='white')
    ax_temp.set_ylabel('Temperature', fontsize=9, color='white')
    ax_temp.set_title('Annealing Temperature', fontsize=10, color='white', fontweight='bold')

    # Gate type colors
    type_colors = {
        'NAND': '#ff6b6b',
        'NOR': '#4ecdc4',
        'AND': '#45b7d1',
        'OR': '#96ceb4',
        'XOR': '#ffeaa7',
        'DFF': '#dfe6e9',
        'MUX': '#a29bfe',
    }

    # Get gate info
    gate_types = [circuit.particles[i].type for i in range(len(circuit.particles))]
    colors = [type_colors.get(gt, '#ffffff') for gt in gate_types]

    # Get initial positions for scatter initialization
    initial_positions = position_history[0] if position_history else np.zeros((len(colors), 2))

    # Initialize scatter plot with initial positions
    scatter = ax_main.scatter(initial_positions[:, 0], initial_positions[:, 1],
                               c=colors, s=80, alpha=0.8, edgecolors='white', linewidths=0.5)

    # Energy line
    energy_line, = ax_energy.plot([], [], color='#00ff88', linewidth=2)
    ax_energy.set_xlim(0, len(energy_history))
    ax_energy.set_ylim(0, max(energy_history) * 1.1)

    # Temperature line
    temp_line, = ax_temp.plot([], [], color='#ff6b6b', linewidth=2)
    ax_temp.set_xlim(0, len(temp_history))
    ax_temp.set_ylim(0, max(temp_history) * 1.1)

    # Title with frame counter
    title_text = ax_main.text(0.5, 1.02, '', transform=ax_main.transAxes,
                               ha='center', fontsize=11, color='white', fontweight='bold')

    # Info text
    info_text = fig.text(0.02, 0.02, '', fontsize=9, color='white',
                         family='monospace')

    # Legend for gate types
    legend_elements = [plt.scatter([], [], c=color, s=80, label=gt, alpha=0.8)
                       for gt, color in type_colors.items()]
    ax_main.legend(handles=legend_elements[-4:], loc='upper right', fontsize=8,
                   framealpha=0.3, labelcolor='white')

    def init():
        scatter.set_offsets(initial_positions)
        energy_line.set_data([], [])
        temp_line.set_data([], [])
        return scatter, energy_line, temp_line, title_text, info_text

    def animate(frame):
        if frame >= len(position_history):
            frame = len(position_history) - 1

        positions = position_history[frame]

        # Update scatter positions
        scatter.set_offsets(positions)

        # Update energy line
        energy_line.set_data(range(frame + 1), energy_history[:frame + 1])

        # Update temperature line
        temp_line.set_data(range(frame + 1), temp_history[:frame + 1])

        # Update title
        progress = int((frame / len(position_history)) * 100)
        title_text.set_text(f'Step {frame * 2} / {len(position_history) * 2} ({progress}%)')

        # Update info
        current_energy = energy_history[frame] if frame < len(energy_history) else energy_history[-1]
        current_temp = temp_history[frame] if frame < len(temp_history) else temp_history[-1]
        info_text.set_text(f'Energy: {current_energy:.1f}  |  Temperature: {current_temp:.2f}  |  Gates: {len(positions)}')

        return scatter, energy_line, temp_line, title_text, info_text

    # Create animation
    anim = animation.FuncAnimation(
        fig, animate, init_func=init,
        frames=len(position_history),
        interval=50, blit=True, repeat=True
    )

    plt.tight_layout()

    # Save as GIF
    print(f"Saving animation to {output_path}...")
    anim.save(output_path, writer='pillow', fps=20, dpi=100)
    print(f"Animation saved!")

    plt.close()


def create_static_result(position_history, circuit, output_path):
    """Create a static image of final placement."""
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(10, 10))

    # Get final positions
    final_positions = position_history[-1]

    # Gate type colors
    type_colors = {
        'NAND': '#ff6b6b',
        'NOR': '#4ecdc4',
        'AND': '#45b7d1',
        'OR': '#96ceb4',
        'XOR': '#ffeaa7',
        'DFF': '#dfe6e9',
        'MUX': '#a29bfe',
    }

    # Draw connections
    ax.set_xlim(-5, 105)
    ax.set_ylim(-5, 105)

    # Draw die boundary
    die_rect = plt.Rectangle((0, 0), 100, 100, fill=False, edgecolor='#00ff88', linewidth=2)
    ax.add_patch(die_rect)

    # Draw connections (faded)
    for conn in circuit.connections:
        src_pos = final_positions[conn.source_id]
        dst_pos = final_positions[conn.dest_id]
        ax.plot([src_pos[0], dst_pos[0]], [src_pos[1], dst_pos[1]],
                'w-', alpha=0.1, linewidth=0.5)

    # Draw gates
    for i, particle in enumerate(circuit.particles.values()):
        color = type_colors.get(particle.type, '#ffffff')
        ax.scatter(final_positions[i, 0], final_positions[i, 1],
                  c=color, s=100, alpha=0.9, edgecolors='white', linewidths=0.5)

    ax.set_xlabel('X Position (μm)', color='white', fontsize=12)
    ax.set_ylabel('Y Position (μm)', color='white', fontsize=12)
    ax.set_title('FLUXION Optimized Placement', color='white', fontsize=14, fontweight='bold')
    ax.set_aspect('equal')

    # Legend
    legend_elements = [plt.scatter([], [], c=color, s=100, label=gt, alpha=0.9)
                       for gt, color in list(type_colors.items())[:5]]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10,
              framealpha=0.3, labelcolor='white')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='#1a1a2e')
    plt.close()
    print(f"Static image saved to {output_path}")


def main():
    print("=" * 60)
    print("FLUXION Placement Animation Generator")
    print("Author: N. Mohana Krishna")
    print("=" * 60)

    # Create demo circuit
    print("\n[1/4] Creating demo circuit...")
    circuit = create_demo_circuit(num_gates=40)

    # Run annealing
    print("[2/4] Running thermodynamic annealing...")
    position_history, energy_history, temp_history, result = run_annealing_with_history(
        circuit, num_steps=300
    )
    print(f"      Final energy: {result.total_energy:.2f}")
    print(f"      Wirelength: {result.total_wirelength:.2f} um")

    # Create output directory
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'docs')
    os.makedirs(output_dir, exist_ok=True)

    # Create animation
    print("[3/4] Creating animation...")
    gif_path = os.path.join(output_dir, 'placement_animation.gif')
    create_animation(position_history, energy_history, temp_history, circuit, gif_path)

    # Create static result
    print("[4/4] Creating static result image...")
    static_path = os.path.join(output_dir, 'placement_result.png')
    create_static_result(position_history, circuit, static_path)

    print("\n" + "=" * 60)
    print("Animation generation complete!")
    print(f"GIF: {gif_path}")
    print(f"PNG: {static_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()