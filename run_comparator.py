#!/usr/bin/env python3
"""
Main entry point for the Universal Operator Benchmarking Framework
This script provides a convenient way to run the comparator tool from the project root
"""

import sys
import os

# Add src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Import and run the comparator tool
from tools.operator_comparator_tool import main

if __name__ == "__main__":
    main()
