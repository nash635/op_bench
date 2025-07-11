#!/usr/bin/env python3
"""
Main entry point for the Universal Operator Benchmarking Framework
This script provides a convenient way to run the comparator tool from the project root
"""

import sys
import os
import tempfile

# Fix matplotlib cache directory issue before importing anything else
try:
    # Try to create matplotlib cache directory
    cache_dir = os.path.expanduser('~/.cache/matplotlib')
    if not os.path.exists(cache_dir):
        try:
            os.makedirs(cache_dir, exist_ok=True)
        except (OSError, PermissionError):
            # If we can't create the default cache directory, use a temp directory
            temp_dir = tempfile.mkdtemp(prefix='matplotlib_cache_')
            os.environ['MPLCONFIGDIR'] = temp_dir
            print(f"[INFO] Using temporary matplotlib cache directory: {temp_dir}")
except Exception as e:
    print(f"[WARN] Matplotlib cache setup failed: {e}")

# Add both src and project root directory to Python path
project_root = os.path.dirname(__file__)
sys.path.insert(0, project_root)  # For tools module
sys.path.insert(0, os.path.join(project_root, 'src'))  # For framework and operators modules

# Import and run the comparator tool
from tools.operator_comparator_tool import main

if __name__ == "__main__":
    main()
