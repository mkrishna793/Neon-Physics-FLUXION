import pytest
import subprocess
from pathlib import Path

def test_rust_binary_compiles():
    """Ensure the rust binary builds successfully."""
    result = subprocess.run(["cargo", "build", "--release"], capture_output=True)
    assert result.returncode == 0, f"Cargo build failed: {result.stderr.decode()}"

def test_placement_pipeline():
    """Run the placement pipeline on the small fixture."""
    fixture = Path("tests/fixtures/small.blif")
    assert fixture.exists(), "Test fixture not found!"
    
    output_file = Path("test_output.def")
    
    # Run the placement
    result = subprocess.run([
        "cargo", "run", "--release", "--",
        "--input", str(fixture),
        "--output", str(output_file)
    ], capture_output=True, text=True)
    
    # It should succeed
    assert result.returncode == 0, f"Placement failed: {result.stderr}"
    
    # It should have produced a DEF file
    assert output_file.exists(), "DEF output was not created"
    
    # The output should contain components
    with open(output_file, 'r') as f:
        content = f.read()
        assert "COMPONENTS" in content
        assert "END COMPONENTS" in content
        assert "+ PLACED" in content

    # Cleanup
    output_file.unlink()
