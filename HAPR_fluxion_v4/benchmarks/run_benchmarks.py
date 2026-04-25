import argparse
import subprocess
from pathlib import Path
import time
import sys

def run_benchmark(input_file, output_file, config_file="config/default.yaml"):
    print(f"=== Running Benchmark on {input_file} ===")
    
    cmd = [
        "cargo", "run", "--release", "--",
        "--config", config_file,
        "--input", str(input_file),
        "--output", str(output_file)
    ]
    
    start_time = time.time()
    
    try:
        # We stream output to see progress
        process = subprocess.Popen(
            cmd, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.STDOUT,
            text=True
        )
        
        for line in process.stdout:
            print(line, end="")
            
        process.wait()
        
        if process.returncode != 0:
            print("Benchmark failed!")
            sys.exit(1)
            
    except FileNotFoundError:
        print("Error: cargo not found. Please ensure Rust is installed and in your PATH.")
        sys.exit(1)
        
    duration = time.time() - start_time
    print(f"\n✅ Benchmark completed in {duration:.2f} seconds.")
    print(f"📁 Output saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run FLUXION v4 benchmarks")
    parser.add_argument("--circuit", required=True, help="Path to input circuit file (.blif or .def)")
    parser.add_argument("--out", default="benchmark_placed.def", help="Path to output def file")
    
    args = parser.parse_args()
    
    circuit_path = Path(args.circuit)
    if not circuit_path.exists():
        print(f"Error: Circuit file {circuit_path} not found.")
        sys.exit(1)
        
    run_benchmark(circuit_path, args.out)
