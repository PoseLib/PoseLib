#!/usr/bin/env python3
"""
Cross-platform stub generation script for pybind11 modules.
Adapted from generate_stubs.sh to work on Windows, macOS, and Linux.
"""

import sys
import os
import subprocess
import re
import shutil
from pathlib import Path


def main():
    if len(sys.argv) != 3:
        print("Usage: python generate_stubs.py <python_executable> <output_dir>")
        sys.exit(1)
    
    python_exec = sys.argv[1]
    output_dir = Path(sys.argv[2])
    package_name = "_core"
    
    print(f"Building stubs with {python_exec} to {output_dir}")
    
    # Run pybind11_stubgen
    cmd = [
        python_exec, "-m", "pybind11_stubgen", package_name,
        "-o", str(output_dir),
        "--numpy-array-use-type-var",
        f"--enum-class-locations=.+:{package_name}",
        "--ignore-invalid-expressions", "poselib::*",
        "--print-invalid-expressions-as-is",
        "--print-safe-value-reprs", "[a-zA-Z]+Options\\(\\)"
    ]
    
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running pybind11_stubgen: {e}")
        sys.exit(1)
    
    # Find the generated stub files
    stub_file = output_dir / f"{package_name}.pyi"
    stub_dir = output_dir / package_name
    
    files_to_process = []
    
    if stub_file.exists():
        files_to_process = [stub_file]
    elif stub_dir.exists():
        files_to_process = list(stub_dir.glob("**/*.pyi"))
    else:
        print(f"Error: Neither stub file {stub_file} nor directory {stub_dir} exists")
        print(f"Available files in {output_dir}:")
        for item in output_dir.iterdir():
            print(f"  {item}")
        sys.exit(1)
    
    # Process each stub file
    for file_path in files_to_process:
        process_stub_file(file_path)
    
    # Format with ruff if available
    if shutil.which("ruff"):
        print("Formatting stubs with ruff...")
        try:
            subprocess.run(["ruff", "format"] + [str(f) for f in files_to_process], check=False)
        except Exception as e:
            print(f"Warning: ruff formatting failed: {e}")
    else:
        print("ruff not found, skipping formatting")


def process_stub_file(file_path):
    """Apply regex replacements to clean up the stub file."""
    print(f"Processing {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    content = re.sub(r'\b_core.\b', '', content)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)


if __name__ == "__main__":
    main() 