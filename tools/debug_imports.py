#!/usr/bin/env python3
"""
Comprehensive diagnostic script for cross-server deployment issues
"""

import sys
import os

def debug_import_issue():
    print("=== Python Import Debugging ===")
    
    # Get the project root
    project_root = os.path.dirname(os.path.abspath(__file__))
    print(f"Project root: {project_root}")
    print(f"Current working directory: {os.getcwd()}")
    print(f"Script location: {os.path.abspath(__file__)}")
    
    # Check directory structure
    print("\n=== Directory Structure Check ===")
    required_paths = [
        "tools",
        "tools/operator_comparator_tool.py",
        "src",
        "src/framework",
        "src/operators",
        "run_comparator.py"
    ]
    
    for path in required_paths:
        full_path = os.path.join(project_root, path)
        exists = os.path.exists(full_path)
        print(f"{'✓' if exists else '✗'} {path}: {full_path}")
    
    # Check Python path
    print("\n=== Python Path Before Modification ===")
    for i, path in enumerate(sys.path):
        print(f"  {i}: {path}")
    
    # Add paths like run_comparator.py does
    sys.path.insert(0, project_root)  # For tools module
    sys.path.insert(0, os.path.join(project_root, 'src'))  # For framework and operators modules
    
    print("\n=== Python Path After Modification ===")
    for i, path in enumerate(sys.path):
        print(f"  {i}: {path}")
    
    # Test imports step by step
    print("\n=== Step-by-Step Import Testing ===")
    
    # Test 1: Can we import tools as a module?
    try:
        import tools
        print("✓ Successfully imported 'tools' module")
        print(f"  tools.__file__: {getattr(tools, '__file__', 'N/A')}")
        print(f"  tools.__path__: {getattr(tools, '__path__', 'N/A')}")
    except ImportError as e:
        print(f"✗ Failed to import 'tools' module: {e}")
    
    # Test 2: Can we list tools directory contents?
    tools_dir = os.path.join(project_root, 'tools')
    if os.path.exists(tools_dir):
        print(f"\n=== Contents of {tools_dir} ===")
        for item in os.listdir(tools_dir):
            item_path = os.path.join(tools_dir, item)
            print(f"  {'[DIR]' if os.path.isdir(item_path) else '[FILE]'} {item}")
    
    # Test 3: Can we import operator_comparator_tool directly?
    try:
        from tools.operator_comparator_tool import main
        print("✓ Successfully imported 'tools.operator_comparator_tool.main'")
    except ImportError as e:
        print(f"✗ Failed to import 'tools.operator_comparator_tool.main': {e}")
    
    # Test 4: Try alternative import methods
    print("\n=== Alternative Import Methods ===")
    
    # Method 1: Add tools to sys.path directly
    tools_path = os.path.join(project_root, 'tools')
    if tools_path not in sys.path:
        sys.path.insert(0, tools_path)
    
    try:
        import operator_comparator_tool
        print("✓ Successfully imported 'operator_comparator_tool' directly")
    except ImportError as e:
        print(f"✗ Failed to import 'operator_comparator_tool' directly: {e}")
    
    # Method 2: Use importlib
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "operator_comparator_tool", 
            os.path.join(project_root, 'tools', 'operator_comparator_tool.py')
        )
        if spec:
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            print("✓ Successfully imported using importlib")
        else:
            print("✗ Failed to create spec using importlib")
    except Exception as e:
        print(f"✗ Failed to import using importlib: {e}")
    
    # Test 5: Check __init__.py files
    print("\n=== __init__.py Files Check ===")
    init_files = [
        "tools/__init__.py",
        "src/__init__.py",
        "src/framework/__init__.py",
        "src/operators/__init__.py"
    ]
    
    for init_file in init_files:
        full_path = os.path.join(project_root, init_file)
        exists = os.path.exists(full_path)
        print(f"{'✓' if exists else '✗'} {init_file}")
        if not exists and 'tools' in init_file:
            # Create missing tools/__init__.py
            try:
                os.makedirs(os.path.dirname(full_path), exist_ok=True)
                with open(full_path, 'w') as f:
                    f.write('"""Tools package for Universal Operator Benchmarking Framework"""\n')
                print(f"  → Created {init_file}")
            except Exception as e:
                print(f"  → Failed to create {init_file}: {e}")

if __name__ == "__main__":
    debug_import_issue()
