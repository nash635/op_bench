#!/usr/bin/env python3
"""
Main entry point for the Universal Operator Benchmarking Framework
This script provides a convenient way to run the comparator tool from the project root
"""

import sys
import os

# Add both src and project root directory to Python path
project_root = os.path.dirname(__file__)
sys.path.insert(0, project_root)  # For tools module
sys.path.insert(0, os.path.join(project_root, 'src'))  # For framework and operators modules

# Import and run the comparator tool
from tools.operator_comparator_tool import main

if __name__ == "__main__":
    main()
