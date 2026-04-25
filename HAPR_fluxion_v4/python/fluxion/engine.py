"""
Python API for FLUXION v4.
Calls the Rust core CLI binary for now, can be extended to use PyO3 bindings.
"""
import os
import subprocess
import yaml
from pathlib import Path

class FluxionEngine:
    """
    Usage:
        engine = FluxionEngine("config/default.yaml")
        engine.place("circuit.blif")
    """

    def __init__(self, config_path="default.yaml"):
        self.config_path = Path(config_path)
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config not found: {self.config_path}")
            
        with open(self.config_path, 'r') as f:
            self.config = yaml.safe_load(f)

    def place(self, input_file: str, output_file: str = "placed.def"):
        """Run full HAPR pipeline via Rust CLI."""
        input_path = Path(input_file)
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")
            
        # Assuming we are running from project root where cargo is available
        cmd = [
            "cargo", "run", "--release", "--",
            "--config", str(self.config_path),
            "--input", str(input_path),
            "--output", str(output_file)
        ]
        
        print(f"Running FLUXION core: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print("FLUXION Core Error:")
            print(result.stderr)
            raise RuntimeError("Placement failed")
            
        print(result.stdout)
        print(f"Placement saved to {output_file}")
        
        return output_file
