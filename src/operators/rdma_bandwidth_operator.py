#!/usr/bin/env python3
"""
RDMA Bandwidth Test Operator
Standard implementation of RDMA bandwidth testing using the operator framework
"""

import torch
import numpy as np
import time
import threading
import subprocess
import os
import re
from typing import List, Dict, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from framework.operator_framework import BaseOperator, OperatorType, OperatorTestCase

class RDMABandwidthOperator(BaseOperator):
    """RDMA bandwidth testing operator"""
    
    def __init__(self):
        super().__init__(OperatorType.RDMA_BANDWIDTH)
        self.rdma_devices = self._discover_rdma_devices()
        self.default_port = 18515
        self.buffer_sizes = [4*1024, 64*1024, 1024*1024, 16*1024*1024]  # 4KB to 16MB
        self._setup_implementations()
        
    def _discover_rdma_devices(self) -> List[Dict[str, Any]]:
        """Discover available RDMA devices"""
        devices = []
        try:
            # Use ibv_devices to discover RDMA devices
            result = subprocess.run(['ibv_devices'], capture_output=True, text=True)
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                for line in lines[1:]:  # Skip header
                    if line.strip():
                        parts = line.split()
                        if len(parts) >= 2:
                            devices.append({
                                'device': parts[0],
                                'node_type': parts[1],
                                'transport': parts[2] if len(parts) > 2 else 'IB'
                            })
        except (subprocess.SubprocessError, FileNotFoundError):
            print("Warning: ibv_devices not found, RDMA tests may not work")
        
        return devices
    
    def _setup_implementations(self):
        """Setup RDMA bandwidth test implementations"""
        self.register_implementation(
            'rdma_perftest',
            self._rdma_perftest_test,
            'RDMA Perftest',
            'RDMA bandwidth test using perftest tools (ib_read_bw, ib_write_bw)'
        )
        
        self.register_implementation(
            'rdma_pingpong',
            self._rdma_pingpong_test,
            'RDMA Ping-Pong',
            'RDMA latency and bandwidth test using pingpong pattern'
        )
        
        self.register_implementation(
            'rdma_raw',
            self._rdma_raw_test,
            'RDMA Raw Test',
            'Raw RDMA bandwidth test using rdma_cm'
        )
        
        # Set reference implementation
        if self.rdma_devices:
            self.set_reference_implementation('rdma_perftest')
    
    def get_test_cases(self) -> List[OperatorTestCase]:
        """Return RDMA bandwidth test cases"""
        test_cases = []
        
        if not self.rdma_devices:
            print("Warning: No RDMA devices found")
            return test_cases
        
        # Different buffer sizes
        for buffer_size in self.buffer_sizes:
            test_cases.append(OperatorTestCase(
                name=f"rdma_bandwidth_{buffer_size//1024}KB",
                input_shapes=[(buffer_size,)],
                input_dtypes=[torch.uint8],
                additional_params={
                    'buffer_size': buffer_size,
                    'duration': 10,  # 10 seconds
                    'server_host': 'localhost',
                    'server_port': self.default_port,
                    'rdma_device': self.rdma_devices[0]['device'] if self.rdma_devices else None,
                    'test_type': 'write',  # write, read, send
                },
                description=f"RDMA bandwidth test with {buffer_size//1024}KB buffer"
            ))
        
        # Different test types
        for test_type in ['write', 'read', 'send']:
            test_cases.append(OperatorTestCase(
                name=f"rdma_{test_type}_1MB",
                input_shapes=[(1024*1024,)],
                input_dtypes=[torch.uint8],
                additional_params={
                    'buffer_size': 1024*1024,
                    'duration': 10,
                    'server_host': 'localhost',
                    'server_port': self.default_port,
                    'rdma_device': self.rdma_devices[0]['device'] if self.rdma_devices else None,
                    'test_type': test_type,
                },
                description=f"RDMA {test_type} bandwidth test with 1MB buffer"
            ))
        
        # Multi-QP test
        test_cases.append(OperatorTestCase(
            name="rdma_multi_qp",
            input_shapes=[(1024*1024,)],
            input_dtypes=[torch.uint8],
            additional_params={
                'buffer_size': 1024*1024,
                'duration': 10,
                'server_host': 'localhost',
                'server_port': self.default_port,
                'rdma_device': self.rdma_devices[0]['device'] if self.rdma_devices else None,
                'test_type': 'write',
                'num_qp': 4,  # Multiple queue pairs
            },
            description="RDMA multi-QP bandwidth test"
        ))
        
        return test_cases
    
    def generate_inputs(self, test_case: OperatorTestCase) -> List[torch.Tensor]:
        """Generate test data for RDMA bandwidth test"""
        buffer_size = test_case.additional_params['buffer_size']
        # Generate random data for RDMA transfer
        data = torch.randint(0, 256, (buffer_size,), dtype=torch.uint8)
        return [data]
    
    def calculate_flops(self, test_case: OperatorTestCase) -> int:
        """Calculate operations for RDMA bandwidth test"""
        # For RDMA tests, we measure data transfer operations
        buffer_size = test_case.additional_params['buffer_size']
        duration = test_case.additional_params['duration']
        # High-performance RDMA can achieve millions of operations per second
        estimated_operations = max(1000, 10000000 // buffer_size) * duration
        return buffer_size * estimated_operations
    
    def get_reference_result(self, inputs: List[torch.Tensor], 
                           test_case: OperatorTestCase) -> torch.Tensor:
        """Get reference result for RDMA bandwidth test"""
        # For network tests, we don't have a fixed reference result
        # Return a placeholder that indicates network test
        return torch.tensor([0.0], dtype=torch.float32)
    
    def verify_correctness(self, result: torch.Tensor, reference: torch.Tensor, 
                         tolerance: float = 1e-4) -> bool:
        """Verify correctness for RDMA bandwidth tests"""
        # For RDMA bandwidth tests, we consider a test correct if:
        # 1. The result is not None
        # 2. The bandwidth is > 0 (indicating successful data transfer)
        if result is None:
            return False
        try:
            bandwidth = float(result.item())
            return bandwidth > 0.0  # Any positive bandwidth indicates success
        except Exception:
            return False
    
    def _rdma_perftest_test(self, inputs: List[torch.Tensor], 
                           params: Dict[str, Any]) -> torch.Tensor:
        """RDMA bandwidth test using perftest tools"""
        if not self.rdma_devices:
            return torch.tensor([0.0], dtype=torch.float32)
        
        buffer_size = params['buffer_size']
        duration = params['duration']
        host = params['server_host']
        port = params['server_port']
        device = params['rdma_device']
        test_type = params['test_type']
        
        # Choose appropriate perftest tool
        if test_type == 'write':
            tool = 'ib_write_bw'
        elif test_type == 'read':
            tool = 'ib_read_bw'
        else:  # send
            tool = 'ib_send_bw'
        
        try:
            # Start server
            server_cmd = [
                tool, '-d', device, '-p', str(port), '-s', str(buffer_size),
                '-D', str(duration), '-F'
            ]
            server_proc = subprocess.Popen(
                server_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            # Wait for server to start
            time.sleep(2)
            
            # Run client
            client_cmd = [
                tool, '-d', device, '-p', str(port), '-s', str(buffer_size),
                '-D', str(duration), '-F', host
            ]
            
            result = subprocess.run(
                client_cmd,
                capture_output=True,
                text=True,
                timeout=duration + 20
            )
            
            # Parse bandwidth from output
            bandwidth_gbps = self._parse_perftest_output(result.stdout)
            
            # Clean up server
            server_proc.terminate()
            server_proc.wait(timeout=5)
            
            return torch.tensor([bandwidth_gbps], dtype=torch.float32)
            
        except Exception as e:
            print(f"RDMA perftest failed: {e}")
            return torch.tensor([0.0], dtype=torch.float32)
    
    def _parse_perftest_output(self, output: str) -> float:
        """Parse perftest output to extract bandwidth"""
        lines = output.strip().split('\n')
        for line in lines:
            # Look for bandwidth line (typically the last line with numbers)
            if 'BW' in line or 'Bandwidth' in line:
                # Extract bandwidth value
                parts = line.split()
                for part in parts:
                    try:
                        # Look for numeric value that could be bandwidth
                        if '.' in part:
                            bandwidth = float(part)
                            if bandwidth > 0:
                                return bandwidth / 1000  # Convert to Gbps if needed
                    except ValueError:
                        continue
        return 0.0
    
    def _rdma_pingpong_test(self, inputs: List[torch.Tensor], 
                           params: Dict[str, Any]) -> torch.Tensor:
        """RDMA ping-pong bandwidth test"""
        if not self.rdma_devices:
            return torch.tensor([0.0], dtype=torch.float32)
        
        buffer_size = params['buffer_size']
        duration = params['duration']
        host = params['server_host']
        port = params['server_port']
        device = params['rdma_device']
        
        try:
            # Use ib_write_lat with multiple iterations for bandwidth measurement
            server_cmd = [
                'ib_write_lat', '-d', device, '-p', str(port), 
                '-s', str(buffer_size), '-n', str(duration * 1000)
            ]
            server_proc = subprocess.Popen(
                server_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            # Wait for server to start
            time.sleep(2)
            
            # Run client
            client_cmd = [
                'ib_write_lat', '-d', device, '-p', str(port),
                '-s', str(buffer_size), '-n', str(duration * 1000), host
            ]
            
            result = subprocess.run(
                client_cmd,
                capture_output=True,
                text=True,
                timeout=duration + 20
            )
            
            # Calculate bandwidth from latency results
            bandwidth_gbps = self._calculate_bandwidth_from_latency(
                result.stdout, buffer_size, duration
            )
            
            # Clean up server
            server_proc.terminate()
            server_proc.wait(timeout=5)
            
            return torch.tensor([bandwidth_gbps], dtype=torch.float32)
            
        except Exception as e:
            print(f"RDMA ping-pong test failed: {e}")
            return torch.tensor([0.0], dtype=torch.float32)
    
    def _calculate_bandwidth_from_latency(self, output: str, buffer_size: int, duration: int) -> float:
        """Calculate bandwidth from latency test output"""
        lines = output.strip().split('\n')
        for line in lines:
            if 'usec' in line.lower() or 'latency' in line.lower():
                # Extract average latency
                parts = line.split()
                for part in parts:
                    try:
                        latency_us = float(part)
                        if latency_us > 0:
                            # Calculate bandwidth: (buffer_size * 8 bits) / (latency_us * 1e-6 seconds)
                            bandwidth_bps = (buffer_size * 8) / (latency_us * 1e-6)
                            return bandwidth_bps / 1e9  # Convert to Gbps
                    except ValueError:
                        continue
        return 0.0
    
    def _rdma_raw_test(self, inputs: List[torch.Tensor], 
                      params: Dict[str, Any]) -> torch.Tensor:
        """Raw RDMA bandwidth test using custom implementation"""
        if not self.rdma_devices:
            return torch.tensor([0.0], dtype=torch.float32)
        
        # For now, fall back to perftest since raw implementation is complex
        # In a real implementation, this would use rdma_cm library directly
        return self._rdma_perftest_test(inputs, params)
    
    def get_rdma_device_info(self) -> List[Dict[str, Any]]:
        """Get information about available RDMA devices"""
        device_info = []
        for device in self.rdma_devices:
            try:
                # Get device attributes
                result = subprocess.run(
                    ['ibv_devinfo', '-d', device['device']],
                    capture_output=True,
                    text=True
                )
                if result.returncode == 0:
                    info = {'device': device['device']}
                    for line in result.stdout.split('\n'):
                        if 'max_mr_size' in line:
                            info['max_mr_size'] = line.split()[-1]
                        elif 'max_qp' in line:
                            info['max_qp'] = line.split()[-1]
                        elif 'max_cq' in line:
                            info['max_cq'] = line.split()[-1]
                    device_info.append(info)
            except Exception as e:
                print(f"Failed to get info for device {device['device']}: {e}")
        
        return device_info
    
    def check_rdma_connectivity(self, remote_host: str) -> bool:
        """Check RDMA connectivity to remote host"""
        if not self.rdma_devices:
            return False
        
        try:
            # Try to ping remote host via RDMA
            result = subprocess.run(
                ['ibping', '-S', remote_host],
                capture_output=True,
                text=True,
                timeout=10
            )
            return result.returncode == 0
        except Exception:
            return False
