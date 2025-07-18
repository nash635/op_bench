#!/usr/bin/env python3
"""
Test script for RDMA Stress Operator
Note: This test has known issues with the implementation dict structure.
It's kept for demonstration but may need fixes.
"""

import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

def test_rdma_operator():
    """Test the RDMA stress operator"""
    try:
        from operators.rdma_stress_operator import RDMAStressOperator
        print("[INFO] Successfully imported RDMAStressOperator")
        
        # Create operator instance
        rdma_op = RDMAStressOperator()
        print(f"[INFO] Created RDMA operator instance")
        
        # Get operator info
        info = rdma_op.get_operator_info()
        print(f"[INFO] Operator info: {info}")
        
        # List test cases
        test_cases = rdma_op.get_test_cases()
        print(f"[INFO] Available test cases:")
        for tc in test_cases:
            print(f"  - {tc.name}: {tc.description}")
        
        # List available implementations
        implementations = rdma_op.get_available_implementations()
        print(f"[INFO] Available implementations: {implementations}")
        
        # Run a simple test
        print("\n[INFO] Running simple bandwidth test...")
        import torch
        
        # Create test inputs
        test_input = torch.randn(1024, 1024)
        
        # Test memory bandwidth implementation
        if 'memory_bandwidth' in implementations:
            result = rdma_op.implementations['memory_bandwidth'](
                [test_input], 
                {'data_size_mb': 1, 'duration_sec': 2}
            )
            print(f"[INFO] Memory bandwidth test result: {result}")
        
        # Test GPU memory if available
        if 'pytorch_gpu_memory' in implementations:
            result = rdma_op.implementations['pytorch_gpu_memory'](
                [test_input], 
                {'data_size_mb': 1, 'duration_sec': 2}
            )
            print(f"[INFO] GPU memory test result: {result}")
        
        # Test TCP bandwidth
        if 'tcp_bandwidth' in implementations:
            result = rdma_op.implementations['tcp_bandwidth'](
                [test_input], 
                {'data_size_mb': 1, 'duration_sec': 2}
            )
            print(f"[INFO] TCP bandwidth test result: {result}")
        
        print("\n[SUCCESS] RDMA operator test completed successfully!")
        return True
        
    except Exception as e:
        print(f"[ERROR] RDMA operator test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_rdma_operator()
    sys.exit(0 if success else 1)
