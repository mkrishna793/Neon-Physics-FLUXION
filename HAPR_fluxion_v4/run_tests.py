#!/usr/bin/env python3
"""
FLUXION v4 Test Script

This script automatically verifies that the HAPR engine compiles and
runs correctly on a small test circuit. It does not require any specific
cloud environment (like Kaggle) — it just needs Python and Rust installed.

Usage:
    python run_tests.py
"""

import subprocess
import sys
from pathlib import Path

def print_header(msg):
    print(f"\n{'='*50}\n{msg}\n{'='*50}")

def run_cmd(cmd, desc):
    print(f"🔄 {desc}...")
    try:
        result = subprocess.run(
            cmd, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.STDOUT, 
            text=True
        )
        if result.returncode != 0:
            print(f"❌ {desc} FAILED!\n")
            print(result.stdout)
            return False
        
        print(f"✅ {desc} passed.")
        return True
    except FileNotFoundError:
        print(f"❌ Error: Command '{cmd[0]}' not found. Is it installed?")
        return False

def main():
    print_header("FLUXION v4 HAPR Engine — Verification Test")
    
    # 1. Check if Rust is installed
    if not run_cmd(["cargo", "--version"], "Checking Rust installation"):
        sys.exit(1)

    # 2. Compile the Rust Engine
    if not run_cmd(["cargo", "build", "--release"], "Compiling HAPR Engine (Release mode)"):
        sys.exit(1)

    # 3. Run the Unit Tests
    if not run_cmd(["cargo", "test"], "Running Rust Unit Tests"):
        sys.exit(1)

    # 4. Run End-to-End Placement Pipeline
    fixture_path = Path("tests/fixtures/small.blif")
    output_path = Path("test_placed.def")
    
    if not fixture_path.exists():
        print(f"❌ Error: Test fixture not found at {fixture_path}")
        sys.exit(1)
        
    cmd = [
        "cargo", "run", "--release", "--",
        "--input", str(fixture_path),
        "--output", str(output_path)
    ]
    
    if run_cmd(cmd, "Executing End-to-End Placement (small.blif)"):
        if output_path.exists():
            print(f"✅ Output successfully generated at: {output_path}")
            # Clean up
            output_path.unlink()
        else:
            print("❌ Placement succeeded, but DEF output file is missing.")
            sys.exit(1)
    else:
        sys.exit(1)

    print_header("🎉 ALL TESTS PASSED SUCCESSFULLY 🎉")

if __name__ == "__main__":
    main()
