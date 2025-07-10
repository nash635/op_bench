#!/usr/bin/env python3
"""
PCIe Bandwidth Test Operator
Standard implementation of PCIe bandwidth testing using the operator framework
"""

import torch
import numpy as np
import time
import threading
import subprocess
import os
from typing import List, Dict, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from framework.operator_framework import BaseOperator, OperatorType, OperatorTestCase

class PCIeBandwidthOperator(BaseOperator):
    """PCIe bandwidth testing operator"""
    
    def __init__(self):
        super().__init__(OperatorType.PCIE_BANDWIDTH)
        self.gpu_devices = self._discover_gpu_devices()
        self.pcie_devices = self._discover_pcie_devices()
        self.buffer_sizes = [1024*1024, 16*1024*1024, 256*1024*1024, 1024*1024*1024]  # 1MB to 1GB
        self._setup_implementations()
        
    def _discover_gpu_devices(self) -> List[Dict[str, Any]]:
        """Discover available GPU devices"""
        devices = []
        try:
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    props = torch.cuda.get_device_properties(i)
                    devices.append({
                        'id': i,
                        'name': props.name,
                        'memory': props.total_memory,
                        'compute_capability': f"{props.major}.{props.minor}"
                    })
        except Exception as e:
            print(f"Warning: Failed to discover GPU devices: {e}")
        
        return devices
    
    def _discover_pcie_devices(self) -> List[Dict[str, Any]]:
        """Discover PCIe devices using lspci"""
        devices = []
        try:
            result = subprocess.run(['lspci', '-v'], capture_output=True, text=True)
            if result.returncode == 0:
                current_device = None
                for line in result.stdout.split('\n'):
                    if line and not line.startswith('\t'):
                        # New device
                        parts = line.split()
                        if len(parts) >= 2:
                            current_device = {
                                'address': parts[0],
                                'description': ' '.join(parts[1:]),
                                'capabilities': []
                            }
                            devices.append(current_device)
                    elif line.startswith('\t') and current_device:
                        # Device property
                        line = line.strip()
                        if 'LnkSta:' in line:
                            current_device['link_status'] = line
                        elif 'LnkCap:' in line:
                            current_device['link_capability'] = line
        except Exception as e:
            print(f"Warning: Failed to discover PCIe devices: {e}")
        
        return devices
    
    def _setup_implementations(self):
        """Setup PCIe bandwidth test implementations"""
        self.register_implementation(
            'pcie_gpu_memcopy',
            self._pcie_gpu_memcopy_test,
            'GPU Memory Copy Test',
            'PCIe bandwidth test using GPU memory copy operations'
        )
        
        self.register_implementation(
            'pcie_cuda_bandwidth',
            self._pcie_cuda_bandwidth_test,
            'CUDA Bandwidth Test',
            'PCIe bandwidth test using CUDA memory transfer'
        )
        
        self.register_implementation(
            'pcie_p2p_test',
            self._pcie_p2p_test,
            'GPU P2P Test',
            'PCIe peer-to-peer bandwidth test between GPUs'
        )
        
        self.register_implementation(
            'pcie_nvlink_test',
            self._pcie_nvlink_test,
            'NVLink Test',
            'High-speed GPU interconnect bandwidth test'
        )
        
        # Set reference implementation
        if self.gpu_devices:
            self.set_reference_implementation('pcie_gpu_memcopy')
    
    def get_test_cases(self) -> List[OperatorTestCase]:
        """Return PCIe bandwidth test cases"""
        test_cases = []
        
        if not self.gpu_devices:
            print("Warning: No GPU devices found")
            return test_cases
        
        # Different buffer sizes
        for buffer_size in self.buffer_sizes:
            test_cases.append(OperatorTestCase(
                name=f"pcie_bandwidth_{buffer_size//1024//1024}MB",
                input_shapes=[(buffer_size//4,)],  # Divide by 4 for float32
                input_dtypes=[torch.float32],
                additional_params={
                    'buffer_size': buffer_size,
                    'gpu_id': 0,
                    'transfer_type': 'h2d',  # host to device
                    'iterations': 100,
                },
                description=f"PCIe bandwidth test with {buffer_size//1024//1024}MB buffer"
            ))
        
        # Different transfer types
        for transfer_type in ['h2d', 'd2h', 'd2d']:
            test_cases.append(OperatorTestCase(
                name=f"pcie_{transfer_type}_256MB",
                input_shapes=[(256*1024*1024//4,)],
                input_dtypes=[torch.float32],
                additional_params={
                    'buffer_size': 256*1024*1024,
                    'gpu_id': 0,
                    'transfer_type': transfer_type,
                    'iterations': 100,
                },
                description=f"PCIe {transfer_type} bandwidth test with 256MB buffer"
            ))
        
        # Multi-GPU P2P test
        if len(self.gpu_devices) > 1:
            test_cases.append(OperatorTestCase(
                name="pcie_p2p_multi_gpu",
                input_shapes=[(256*1024*1024//4,)],
                input_dtypes=[torch.float32],
                additional_params={
                    'buffer_size': 256*1024*1024,
                    'gpu_id': 0,
                    'target_gpu_id': 1,
                    'transfer_type': 'p2p',
                    'iterations': 100,
                },
                description="PCIe P2P bandwidth test between GPUs"
            ))
        
        return test_cases
    
    def generate_inputs(self, test_case: OperatorTestCase) -> List[torch.Tensor]:
        """Generate test data for PCIe bandwidth test"""
        shape = test_case.input_shapes[0]
        dtype = test_case.input_dtypes[0]
        # Generate random data for transfer
        data = torch.randn(shape, dtype=dtype)
        return [data]
    
    def calculate_flops(self, test_case: OperatorTestCase) -> int:
        """Calculate operations for PCIe bandwidth test"""
        # PCIe tests measure memory transfer operations, not FLOPs
        buffer_size = test_case.additional_params['buffer_size']
        iterations = test_case.additional_params['iterations']
        return buffer_size * iterations
    
    def get_reference_result(self, inputs: List[torch.Tensor], 
                           test_case: OperatorTestCase) -> torch.Tensor:
        """Get reference result for PCIe bandwidth test"""
        # For network tests, we don't have a fixed reference result
        # Return a placeholder that indicates network test
        return torch.tensor([0.0], dtype=torch.float32)
    
    def verify_correctness(self, result: torch.Tensor, reference: torch.Tensor, 
                         tolerance: float = 1e-4) -> bool:
        """Verify correctness for PCIe bandwidth tests"""
        # For PCIe bandwidth tests, we consider a test correct if:
        # 1. The result is not None
        # 2. The bandwidth is > 0 (indicating successful data transfer)
        if result is None:
            return False
        try:
            bandwidth = float(result.item())
            return bandwidth > 0.0  # Any positive bandwidth indicates success
        except Exception:
            return False
    
    def _pcie_gpu_memcopy_test(self, inputs: List[torch.Tensor], 
                              params: Dict[str, Any]) -> torch.Tensor:
        """PCIe bandwidth test using GPU memory copy operations"""
        if not self.gpu_devices:
            return torch.tensor([0.0], dtype=torch.float32)
        
        data = inputs[0]
        gpu_id = params['gpu_id']
        transfer_type = params['transfer_type']
        iterations = params['iterations']
        
        try:
            device = torch.device(f'cuda:{gpu_id}')
            
            # Warm up
            for _ in range(5):
                if transfer_type == 'h2d':
                    _ = data.to(device)
                elif transfer_type == 'd2h':
                    gpu_data = data.to(device)
                    _ = gpu_data.cpu()
                elif transfer_type == 'd2d':
                    gpu_data = data.to(device)
                    _ = gpu_data.clone()
            
            # Synchronize
            torch.cuda.synchronize()
            
            # Measure bandwidth
            start_time = time.time()
            
            for _ in range(iterations):
                if transfer_type == 'h2d':
                    gpu_data = data.to(device)
                elif transfer_type == 'd2h':
                    gpu_data = data.to(device)
                    cpu_data = gpu_data.cpu()
                elif transfer_type == 'd2d':
                    gpu_data = data.to(device)
                    gpu_data2 = gpu_data.clone()
            
            torch.cuda.synchronize()
            end_time = time.time()
            
            # Calculate bandwidth
            total_time = end_time - start_time
            data_size = data.numel() * data.element_size()  # bytes
            total_bytes = data_size * iterations
            bandwidth_gbps = (total_bytes * 8) / (total_time * 1e9)  # Convert to Gbps
            
            return torch.tensor([bandwidth_gbps], dtype=torch.float32)
            
        except Exception as e:
            print(f"PCIe GPU memcopy test failed: {e}")
            return torch.tensor([0.0], dtype=torch.float32)
    
    def _pcie_cuda_bandwidth_test(self, inputs: List[torch.Tensor], 
                                 params: Dict[str, Any]) -> torch.Tensor:
        """PCIe bandwidth test using CUDA memory transfer"""
        if not self.gpu_devices:
            return torch.tensor([0.0], dtype=torch.float32)
        
        data = inputs[0]
        gpu_id = params['gpu_id']
        transfer_type = params['transfer_type']
        iterations = params['iterations']
        
        try:
            device = torch.device(f'cuda:{gpu_id}')
            
            # Allocate pinned memory for better performance
            if transfer_type in ['h2d', 'd2h']:
                pinned_data = data.pin_memory()
            else:
                pinned_data = data
            
            # Warm up
            for _ in range(5):
                if transfer_type == 'h2d':
                    _ = pinned_data.to(device, non_blocking=True)
                elif transfer_type == 'd2h':
                    gpu_data = pinned_data.to(device, non_blocking=True)
                    _ = gpu_data.cpu()
                elif transfer_type == 'd2d':
                    gpu_data = pinned_data.to(device, non_blocking=True)
                    _ = gpu_data.clone()
            
            torch.cuda.synchronize()
            
            # Measure bandwidth
            start_time = time.time()
            
            for _ in range(iterations):
                if transfer_type == 'h2d':
                    gpu_data = pinned_data.to(device, non_blocking=True)
                elif transfer_type == 'd2h':
                    gpu_data = pinned_data.to(device, non_blocking=True)
                    cpu_data = gpu_data.cpu()
                elif transfer_type == 'd2d':
                    gpu_data = pinned_data.to(device, non_blocking=True)
                    gpu_data2 = gpu_data.clone()
            
            torch.cuda.synchronize()
            end_time = time.time()
            
            # Calculate bandwidth
            total_time = end_time - start_time
            data_size = data.numel() * data.element_size()
            total_bytes = data_size * iterations
            bandwidth_gbps = (total_bytes * 8) / (total_time * 1e9)
            
            return torch.tensor([bandwidth_gbps], dtype=torch.float32)
            
        except Exception as e:
            print(f"PCIe CUDA bandwidth test failed: {e}")
            return torch.tensor([0.0], dtype=torch.float32)
    
    def _pcie_p2p_test(self, inputs: List[torch.Tensor], 
                      params: Dict[str, Any]) -> torch.Tensor:
        """PCIe peer-to-peer bandwidth test between GPUs"""
        if len(self.gpu_devices) < 2:
            return torch.tensor([0.0], dtype=torch.float32)
        
        data = inputs[0]
        gpu_id = params['gpu_id']
        target_gpu_id = params.get('target_gpu_id', 1)
        iterations = params['iterations']
        
        try:
            device1 = torch.device(f'cuda:{gpu_id}')
            device2 = torch.device(f'cuda:{target_gpu_id}')
            
            # Check if P2P is enabled
            if torch.cuda.can_device_access_peer(gpu_id, target_gpu_id):
                print(f"P2P enabled between GPU {gpu_id} and GPU {target_gpu_id}")
            else:
                print(f"P2P not available between GPU {gpu_id} and GPU {target_gpu_id}")
            
            # Allocate data on first GPU
            gpu_data1 = data.to(device1)
            
            # Warm up
            for _ in range(5):
                gpu_data2 = gpu_data1.to(device2)
            
            torch.cuda.synchronize()
            
            # Measure bandwidth
            start_time = time.time()
            
            for _ in range(iterations):
                gpu_data2 = gpu_data1.to(device2)
            
            torch.cuda.synchronize()
            end_time = time.time()
            
            # Calculate bandwidth
            total_time = end_time - start_time
            data_size = data.numel() * data.element_size()
            total_bytes = data_size * iterations
            bandwidth_gbps = (total_bytes * 8) / (total_time * 1e9)
            
            return torch.tensor([bandwidth_gbps], dtype=torch.float32)
            
        except Exception as e:
            print(f"PCIe P2P test failed: {e}")
            return torch.tensor([0.0], dtype=torch.float32)
    
    def _pcie_nvlink_test(self, inputs: List[torch.Tensor], 
                         params: Dict[str, Any]) -> torch.Tensor:
        """NVLink bandwidth test (high-speed GPU interconnect)"""
        if len(self.gpu_devices) < 2:
            return torch.tensor([0.0], dtype=torch.float32)
        
        # Check if NVLink is available
        try:
            result = subprocess.run(['nvidia-smi', 'nvlink', '--status'], 
                                  capture_output=True, text=True)
            if result.returncode != 0:
                print("NVLink not available or nvidia-smi not found")
                return torch.tensor([0.0], dtype=torch.float32)
        except Exception:
            print("Cannot check NVLink status")
            return torch.tensor([0.0], dtype=torch.float32)
        
        # Run P2P test (NVLink is transparent to CUDA)
        return self._pcie_p2p_test(inputs, params)
    
    def get_pcie_topology(self) -> Dict[str, Any]:
        """Get PCIe topology information"""
        topology = {
            'gpu_devices': self.gpu_devices,
            'pcie_devices': [],
            'nvlink_status': None
        }
        
        # Get detailed PCIe info for GPU devices
        for gpu in self.gpu_devices:
            try:
                # Get PCIe info using nvidia-smi
                result = subprocess.run([
                    'nvidia-smi', '--query-gpu=pci.bus_id,pci.link.gen.current,pci.link.width.current',
                    '--format=csv,noheader,nounits', f'--id={gpu["id"]}'
                ], capture_output=True, text=True)
                
                if result.returncode == 0:
                    pci_info = result.stdout.strip().split(', ')
                    if len(pci_info) >= 3:
                        topology['pcie_devices'].append({
                            'gpu_id': gpu['id'],
                            'pci_bus_id': pci_info[0],
                            'pcie_gen': pci_info[1],
                            'pcie_width': pci_info[2]
                        })
            except Exception as e:
                print(f"Failed to get PCIe info for GPU {gpu['id']}: {e}")
        
        # Get NVLink status
        try:
            result = subprocess.run(['nvidia-smi', 'nvlink', '--status'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                topology['nvlink_status'] = result.stdout
        except Exception:
            pass
        
        return topology
    
    def benchmark_memory_bandwidth(self, gpu_id: int = 0) -> Dict[str, float]:
        """Benchmark GPU memory bandwidth"""
        if not self.gpu_devices:
            return {'bandwidth_gbps': 0.0}
        
        try:
            device = torch.device(f'cuda:{gpu_id}')
            
            # Test different sizes
            sizes = [1024*1024, 16*1024*1024, 256*1024*1024]  # 1MB, 16MB, 256MB
            results = {}
            
            for size in sizes:
                # Create test data
                data = torch.randn(size//4, device=device)  # float32
                
                # Warm up
                for _ in range(10):
                    _ = data * 2.0
                
                torch.cuda.synchronize()
                
                # Benchmark
                iterations = 100
                start_time = time.time()
                
                for _ in range(iterations):
                    result = data * 2.0  # Simple operation to test memory bandwidth
                
                torch.cuda.synchronize()
                end_time = time.time()
                
                # Calculate bandwidth
                total_time = end_time - start_time
                bytes_transferred = size * iterations * 2  # Read + Write
                bandwidth_gbps = (bytes_transferred * 8) / (total_time * 1e9)
                
                results[f'{size//1024//1024}MB'] = bandwidth_gbps
            
            return results
            
        except Exception as e:
            print(f"Memory bandwidth benchmark failed: {e}")
            return {'bandwidth_gbps': 0.0}
