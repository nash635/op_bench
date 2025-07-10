#!/usr/bin/env python3
"""
Network Stress Test Operator
Unified network stress testing operator that combines TCP, RDMA, and PCIe tests
"""

import torch
import numpy as np
import time
import sys
import os
from typing import List, Dict, Any, Optional, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from framework.operator_framework import BaseOperator, OperatorType, OperatorTestCase

# Import individual operators
from .tcp_bandwidth_operator import TCPBandwidthOperator
from .rdma_bandwidth_operator import RDMABandwidthOperator
from .pcie_bandwidth_operator import PCIeBandwidthOperator

class NetworkStressOperator(BaseOperator):
    """Unified network stress testing operator"""
    
    def __init__(self):
        super().__init__(OperatorType.RDMA_STRESS)  # Use existing enum
        self.tcp_operator = TCPBandwidthOperator()
        self.rdma_operator = RDMABandwidthOperator()
        self.pcie_operator = PCIeBandwidthOperator()
        self._setup_implementations()
        
    def _setup_implementations(self):
        """Setup network stress test implementations"""
        self.register_implementation(
            'tcp_stress',
            self._tcp_stress_test,
            'TCP Stress Test',
            'Comprehensive TCP network stress testing'
        )
        
        self.register_implementation(
            'rdma_stress',
            self._rdma_stress_test,
            'RDMA Stress Test',
            'Comprehensive RDMA network stress testing'
        )
        
        self.register_implementation(
            'pcie_stress',
            self._pcie_stress_test,
            'PCIe Stress Test',
            'Comprehensive PCIe bandwidth stress testing'
        )
        
        self.register_implementation(
            'full_network_stress',
            self._full_network_stress_test,
            'Full Network Stress Test',
            'Combined TCP, RDMA, and PCIe stress testing'
        )
        
        self.set_reference_implementation('tcp_stress')
    
    def get_test_cases(self) -> List[OperatorTestCase]:
        """Return network stress test cases"""
        test_cases = []
        
        # TCP stress test cases
        test_cases.append(OperatorTestCase(
            name="tcp_stress_comprehensive",
            input_shapes=[(1024*1024,)],
            input_dtypes=[torch.uint8],
            additional_params={
                'test_type': 'tcp',
                'duration': 30,
                'concurrent_connections': 10,
                'buffer_sizes': [64*1024, 256*1024, 1024*1024],
                'server_host': 'localhost',
                'server_port': 12345,
            },
            description="Comprehensive TCP network stress test"
        ))
        
        # RDMA stress test cases
        test_cases.append(OperatorTestCase(
            name="rdma_stress_comprehensive",
            input_shapes=[(1024*1024,)],
            input_dtypes=[torch.uint8],
            additional_params={
                'test_type': 'rdma',
                'duration': 30,
                'concurrent_qps': 8,
                'buffer_sizes': [4*1024, 64*1024, 1024*1024],
                'rdma_operations': ['write', 'read', 'send'],
                'server_host': 'localhost',
                'server_port': 18515,
            },
            description="Comprehensive RDMA network stress test"
        ))
        
        # PCIe stress test cases
        test_cases.append(OperatorTestCase(
            name="pcie_stress_comprehensive",
            input_shapes=[(256*1024*1024//4,)],
            input_dtypes=[torch.float32],
            additional_params={
                'test_type': 'pcie',
                'duration': 30,
                'gpu_ids': [0] if torch.cuda.is_available() else [],
                'transfer_types': ['h2d', 'd2h', 'd2d'],
                'buffer_sizes': [16*1024*1024, 256*1024*1024, 1024*1024*1024],
            },
            description="Comprehensive PCIe bandwidth stress test"
        ))
        
        # Full network stress test
        test_cases.append(OperatorTestCase(
            name="full_network_stress",
            input_shapes=[(1024*1024,)],
            input_dtypes=[torch.uint8],
            additional_params={
                'test_type': 'full',
                'duration': 60,
                'enable_tcp': True,
                'enable_rdma': True,
                'enable_pcie': True,
                'concurrent_tests': True,
            },
            description="Full network stress test combining all protocols"
        ))
        
        return test_cases
    
    def generate_inputs(self, test_case: OperatorTestCase) -> List[torch.Tensor]:
        """Generate test data for network stress test"""
        shape = test_case.input_shapes[0]
        dtype = test_case.input_dtypes[0]
        
        if dtype == torch.uint8:
            data = torch.randint(0, 256, shape, dtype=dtype)
        else:
            data = torch.randn(shape, dtype=dtype)
        
        return [data]
    
    def calculate_flops(self, test_case: OperatorTestCase) -> int:
        """Calculate operations for network stress test"""
        # Network stress tests measure throughput, not FLOPs
        params = test_case.additional_params
        duration = params.get('duration', 30)
        buffer_size = test_case.input_shapes[0][0]
        
        # Estimate operations based on test type
        if params['test_type'] == 'tcp':
            concurrent = params.get('concurrent_connections', 1)
            return buffer_size * concurrent * duration * 100  # Assume 100 ops/sec
        elif params['test_type'] == 'rdma':
            concurrent_qps = params.get('concurrent_qps', 1)
            return buffer_size * concurrent_qps * duration * 1000  # Higher throughput
        elif params['test_type'] == 'pcie':
            gpu_count = len(params.get('gpu_ids', [1]))
            return buffer_size * gpu_count * duration * 10000  # Very high throughput
        else:  # full
            return buffer_size * duration * 10000  # Combined throughput
    
    def get_reference_result(self, inputs: List[torch.Tensor], 
                           test_case: OperatorTestCase) -> torch.Tensor:
        """Get reference result for network stress test"""
        test_type = test_case.additional_params['test_type']
        
        if test_type == 'tcp':
            return self._tcp_stress_test(inputs, test_case.additional_params)
        elif test_type == 'rdma':
            return self._rdma_stress_test(inputs, test_case.additional_params)
        elif test_type == 'pcie':
            return self._pcie_stress_test(inputs, test_case.additional_params)
        else:  # full
            return self._full_network_stress_test(inputs, test_case.additional_params)
    
    def _tcp_stress_test(self, inputs: List[torch.Tensor], 
                        params: Dict[str, Any]) -> torch.Tensor:
        """TCP stress test implementation"""
        data = inputs[0]
        duration = params['duration']
        concurrent_connections = params.get('concurrent_connections', 1)
        buffer_sizes = params.get('buffer_sizes', [1024*1024])
        
        total_bandwidth = 0.0
        test_count = 0
        
        try:
            # Test different buffer sizes
            for buffer_size in buffer_sizes:
                # Create test case for TCP operator
                tcp_test_case = OperatorTestCase(
                    name=f"tcp_stress_{buffer_size}",
                    input_shapes=[(buffer_size,)],
                    input_dtypes=[torch.uint8],
                    additional_params={
                        'buffer_size': buffer_size,
                        'duration': duration // len(buffer_sizes),
                        'server_host': params['server_host'],
                        'server_port': params['server_port'],
                    }
                )
                
                # Generate appropriate input
                tcp_data = torch.randint(0, 256, (buffer_size,), dtype=torch.uint8)
                
                # Run multiple concurrent tests
                for _ in range(concurrent_connections):
                    result = self.tcp_operator._tcp_client_server_test(
                        [tcp_data], tcp_test_case.additional_params
                    )
                    total_bandwidth += result.item()
                    test_count += 1
            
            average_bandwidth = total_bandwidth / max(test_count, 1)
            return torch.tensor([average_bandwidth], dtype=torch.float32)
            
        except Exception as e:
            print(f"TCP stress test failed: {e}")
            return torch.tensor([0.0], dtype=torch.float32)
    
    def _rdma_stress_test(self, inputs: List[torch.Tensor], 
                         params: Dict[str, Any]) -> torch.Tensor:
        """RDMA stress test implementation"""
        if not self.rdma_operator.rdma_devices:
            print("No RDMA devices available")
            return torch.tensor([0.0], dtype=torch.float32)
        
        data = inputs[0]
        duration = params['duration']
        concurrent_qps = params.get('concurrent_qps', 1)
        buffer_sizes = params.get('buffer_sizes', [1024*1024])
        rdma_operations = params.get('rdma_operations', ['write'])
        
        total_bandwidth = 0.0
        test_count = 0
        
        try:
            # Test different operations and buffer sizes
            for operation in rdma_operations:
                for buffer_size in buffer_sizes:
                    # Create test case for RDMA operator
                    rdma_test_case = OperatorTestCase(
                        name=f"rdma_stress_{operation}_{buffer_size}",
                        input_shapes=[(buffer_size,)],
                        input_dtypes=[torch.uint8],
                        additional_params={
                            'buffer_size': buffer_size,
                            'duration': duration // (len(rdma_operations) * len(buffer_sizes)),
                            'server_host': params['server_host'],
                            'server_port': params['server_port'],
                            'rdma_device': self.rdma_operator.rdma_devices[0]['device'],
                            'test_type': operation,
                        }
                    )
                    
                    # Generate appropriate input
                    rdma_data = torch.randint(0, 256, (buffer_size,), dtype=torch.uint8)
                    
                    # Run multiple concurrent tests
                    for _ in range(concurrent_qps):
                        result = self.rdma_operator._rdma_perftest_test(
                            [rdma_data], rdma_test_case.additional_params
                        )
                        total_bandwidth += result.item()
                        test_count += 1
            
            average_bandwidth = total_bandwidth / max(test_count, 1)
            return torch.tensor([average_bandwidth], dtype=torch.float32)
            
        except Exception as e:
            print(f"RDMA stress test failed: {e}")
            return torch.tensor([0.0], dtype=torch.float32)
    
    def _pcie_stress_test(self, inputs: List[torch.Tensor], 
                         params: Dict[str, Any]) -> torch.Tensor:
        """PCIe stress test implementation"""
        if not self.pcie_operator.gpu_devices:
            print("No GPU devices available")
            return torch.tensor([0.0], dtype=torch.float32)
        
        data = inputs[0]
        duration = params['duration']
        gpu_ids = params.get('gpu_ids', [0])
        transfer_types = params.get('transfer_types', ['h2d'])
        buffer_sizes = params.get('buffer_sizes', [256*1024*1024])
        
        total_bandwidth = 0.0
        test_count = 0
        
        try:
            # Test different GPUs and transfer types
            for gpu_id in gpu_ids:
                for transfer_type in transfer_types:
                    for buffer_size in buffer_sizes:
                        # Create test case for PCIe operator
                        pcie_test_case = OperatorTestCase(
                            name=f"pcie_stress_{gpu_id}_{transfer_type}_{buffer_size}",
                            input_shapes=[(buffer_size//4,)],
                            input_dtypes=[torch.float32],
                            additional_params={
                                'buffer_size': buffer_size,
                                'gpu_id': gpu_id,
                                'transfer_type': transfer_type,
                                'iterations': 50,
                            }
                        )
                        
                        # Generate appropriate input
                        pcie_data = torch.randn(buffer_size//4, dtype=torch.float32)
                        
                        result = self.pcie_operator._pcie_gpu_memcopy_test(
                            [pcie_data], pcie_test_case.additional_params
                        )
                        total_bandwidth += result.item()
                        test_count += 1
            
            average_bandwidth = total_bandwidth / max(test_count, 1)
            return torch.tensor([average_bandwidth], dtype=torch.float32)
            
        except Exception as e:
            print(f"PCIe stress test failed: {e}")
            return torch.tensor([0.0], dtype=torch.float32)
    
    def _full_network_stress_test(self, inputs: List[torch.Tensor], 
                                 params: Dict[str, Any]) -> torch.Tensor:
        """Full network stress test combining all protocols"""
        data = inputs[0]
        duration = params['duration']
        enable_tcp = params.get('enable_tcp', True)
        enable_rdma = params.get('enable_rdma', True)
        enable_pcie = params.get('enable_pcie', True)
        concurrent_tests = params.get('concurrent_tests', True)
        
        results = []
        
        try:
            # Run TCP tests
            if enable_tcp:
                tcp_params = {
                    'duration': duration // 3,
                    'concurrent_connections': 5,
                    'buffer_sizes': [256*1024, 1024*1024],
                    'server_host': 'localhost',
                    'server_port': 12345,
                }
                tcp_result = self._tcp_stress_test(inputs, tcp_params)
                results.append(tcp_result.item())
                print(f"TCP stress test: {tcp_result.item():.2f} Gbps")
            
            # Run RDMA tests
            if enable_rdma and self.rdma_operator.rdma_devices:
                rdma_params = {
                    'duration': duration // 3,
                    'concurrent_qps': 4,
                    'buffer_sizes': [64*1024, 1024*1024],
                    'rdma_operations': ['write', 'read'],
                    'server_host': 'localhost',
                    'server_port': 18515,
                }
                rdma_result = self._rdma_stress_test(inputs, rdma_params)
                results.append(rdma_result.item())
                print(f"RDMA stress test: {rdma_result.item():.2f} Gbps")
            
            # Run PCIe tests
            if enable_pcie and self.pcie_operator.gpu_devices:
                pcie_params = {
                    'duration': duration // 3,
                    'gpu_ids': [0],
                    'transfer_types': ['h2d', 'd2h'],
                    'buffer_sizes': [256*1024*1024],
                }
                pcie_result = self._pcie_stress_test(inputs, pcie_params)
                results.append(pcie_result.item())
                print(f"PCIe stress test: {pcie_result.item():.2f} Gbps")
            
            # Calculate overall score
            if results:
                overall_score = sum(results) / len(results)
                return torch.tensor([overall_score], dtype=torch.float32)
            else:
                return torch.tensor([0.0], dtype=torch.float32)
                
        except Exception as e:
            print(f"Full network stress test failed: {e}")
            return torch.tensor([0.0], dtype=torch.float32)
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get comprehensive system information"""
        return {
            'tcp_info': {
                'available': True,
                'default_port': self.tcp_operator.default_port,
                'buffer_sizes': self.tcp_operator.buffer_sizes,
            },
            'rdma_info': {
                'available': len(self.rdma_operator.rdma_devices) > 0,
                'devices': self.rdma_operator.rdma_devices,
                'device_info': self.rdma_operator.get_rdma_device_info(),
            },
            'pcie_info': {
                'available': len(self.pcie_operator.gpu_devices) > 0,
                'gpu_devices': self.pcie_operator.gpu_devices,
                'topology': self.pcie_operator.get_pcie_topology(),
            }
        }
    
    def run_quick_benchmark(self) -> Dict[str, float]:
        """Run a quick benchmark of all available protocols"""
        results = {}
        
        # Quick TCP test
        try:
            tcp_data = torch.randint(0, 256, (1024*1024,), dtype=torch.uint8)
            tcp_params = {
                'buffer_size': 1024*1024,
                'duration': 5,
                'server_host': 'localhost',
                'server_port': 12345,
            }
            tcp_result = self.tcp_operator._tcp_client_server_test([tcp_data], tcp_params)
            results['tcp_gbps'] = tcp_result.item()
        except Exception as e:
            results['tcp_gbps'] = 0.0
            print(f"TCP quick test failed: {e}")
        
        # Quick RDMA test
        if self.rdma_operator.rdma_devices:
            try:
                rdma_data = torch.randint(0, 256, (1024*1024,), dtype=torch.uint8)
                rdma_params = {
                    'buffer_size': 1024*1024,
                    'duration': 5,
                    'server_host': 'localhost',
                    'server_port': 18515,
                    'rdma_device': self.rdma_operator.rdma_devices[0]['device'],
                    'test_type': 'write',
                }
                rdma_result = self.rdma_operator._rdma_perftest_test([rdma_data], rdma_params)
                results['rdma_gbps'] = rdma_result.item()
            except Exception as e:
                results['rdma_gbps'] = 0.0
                print(f"RDMA quick test failed: {e}")
        else:
            results['rdma_gbps'] = 0.0
        
        # Quick PCIe test
        if self.pcie_operator.gpu_devices:
            try:
                pcie_data = torch.randn(256*1024*1024//4, dtype=torch.float32)
                pcie_params = {
                    'buffer_size': 256*1024*1024,
                    'gpu_id': 0,
                    'transfer_type': 'h2d',
                    'iterations': 10,
                }
                pcie_result = self.pcie_operator._pcie_gpu_memcopy_test([pcie_data], pcie_params)
                results['pcie_gbps'] = pcie_result.item()
            except Exception as e:
                results['pcie_gbps'] = 0.0
                print(f"PCIe quick test failed: {e}")
        else:
            results['pcie_gbps'] = 0.0
        
        return results
