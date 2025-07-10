#!/usr/bin/env python3
"""
TCP Bandwidth Test Operator
Standard implementation of TCP bandwidth testing using the operator framework
"""

import torch
import numpy as np
import time
import threading
import subprocess
import socket
import struct
import select
import sys
import os
from typing import List, Dict, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from framework.operator_framework import BaseOperator, OperatorType, OperatorTestCase

class TCPBandwidthOperator(BaseOperator):
    """TCP bandwidth testing operator"""
    
    def __init__(self):
        super().__init__(OperatorType.TCP_BANDWIDTH)
        self.default_port = 12345
        self.port_range_start = 12345
        self.port_range_end = 12400
        self.buffer_sizes = [64*1024, 256*1024, 1024*1024, 4*1024*1024]  # 64KB to 4MB
        self._setup_implementations()
        
    def _setup_implementations(self):
        """Setup TCP bandwidth test implementations"""
        self.register_implementation(
            'tcp_client_server',
            self._tcp_client_server_test,
            'TCP Client-Server Test',
            'Standard TCP client-server bandwidth test'
        )
        
        self.register_implementation(
            'tcp_iperf3',
            self._tcp_iperf3_test,
            'TCP iperf3 Test',
            'TCP bandwidth test using iperf3'
        )
        
        self.register_implementation(
            'tcp_netcat',
            self._tcp_netcat_test,
            'TCP Netcat Test',
            'TCP bandwidth test using netcat'
        )
        
        self.set_reference_implementation('tcp_client_server')
    
    def get_test_cases(self) -> List[OperatorTestCase]:
        """Return TCP bandwidth test cases"""
        test_cases = []
        
        # Quick test cases with shorter durations for interactive use
        quick_test_cases = [
            # Small buffer sizes for quick tests (2-5 seconds)
            (64*1024, 2, "Quick test with 64KB buffer"),
            (256*1024, 3, "Quick test with 256KB buffer"),
            (1024*1024, 5, "Quick test with 1MB buffer"),
        ]
        
        for buffer_size, duration, description in quick_test_cases:
            test_cases.append(OperatorTestCase(
                name=f"tcp_bandwidth_{buffer_size//1024}KB",
                input_shapes=[(buffer_size,)],
                input_dtypes=[torch.uint8],
                additional_params={
                    'buffer_size': buffer_size,
                    'duration': duration,
                    'server_host': 'localhost',
                    'server_port': self.default_port,
                },
                description=description
            ))
        
        # Larger test case for performance measurement
        test_cases.append(OperatorTestCase(
            name=f"tcp_bandwidth_4096KB",
            input_shapes=[(4*1024*1024,)],
            input_dtypes=[torch.uint8],
            additional_params={
                'buffer_size': 4*1024*1024,
                'duration': 10,  # 10 seconds for large buffer
                'server_host': 'localhost',
                'server_port': self.default_port,
            },
            description="Performance test with 4MB buffer"
        ))
        
        # Different durations (kept shorter for interactive use)
        for duration in [5, 10]:
            test_cases.append(OperatorTestCase(
                name=f"tcp_bandwidth_{duration}s",
                input_shapes=[(1024*1024,)],  # 1MB buffer
                input_dtypes=[torch.uint8],
                additional_params={
                    'buffer_size': 1024*1024,
                    'duration': duration,
                    'server_host': 'localhost',
                    'server_port': self.default_port,
                },
                description=f"TCP bandwidth test for {duration} seconds"
            ))
        
        return test_cases
    
    def generate_inputs(self, test_case: OperatorTestCase) -> List[torch.Tensor]:
        """Generate test data for TCP bandwidth test"""
        buffer_size = test_case.additional_params['buffer_size']
        # Generate random data to send
        data = torch.randint(0, 256, (buffer_size,), dtype=torch.uint8)
        return [data]
    
    def calculate_flops(self, test_case: OperatorTestCase) -> int:
        """Calculate operations for TCP bandwidth test"""
        # For bandwidth tests, we measure data transfer rate, not FLOPS
        # Return buffer size * iterations as a proxy for operations
        buffer_size = test_case.additional_params['buffer_size']
        duration = test_case.additional_params['duration']
        # Assume ~1000 transfers per second for small buffers
        estimated_transfers = max(1, min(1000, 1000000 // buffer_size)) * duration
        return buffer_size * estimated_transfers
    
    def get_reference_result(self, inputs: List[torch.Tensor], 
                           test_case: OperatorTestCase) -> torch.Tensor:
        """Get reference result for TCP bandwidth test"""
        # For network tests, we don't have a fixed reference result
        # Return a placeholder that indicates network test
        return torch.tensor([0.0], dtype=torch.float32)
    
    def verify_correctness(self, result: torch.Tensor, reference: torch.Tensor, 
                         tolerance: float = 1e-4) -> bool:
        """Verify correctness for network tests"""
        # For network bandwidth tests, we consider a test correct if:
        # 1. The result is not None
        # 2. The bandwidth is > 0 (indicating successful data transfer)
        if result is None:
            return False
        try:
            bandwidth = float(result.item())
            return bandwidth > 0.0  # Any positive bandwidth indicates success
        except Exception:
            return False
    
    def _tcp_client_server_test(self, inputs: List[torch.Tensor], 
                               params: Dict[str, Any]) -> torch.Tensor:
        """TCP client-server bandwidth test implementation with timeout"""
        data = inputs[0]
        buffer_size = params['buffer_size']
        duration = params['duration']
        host = params['server_host']
        requested_port = params['server_port']
        
        # Add timeout protection
        max_test_time = duration + 30  # Add 30 seconds buffer
        start_time = time.time()
        
        # Find an available port
        port = self._find_available_port(requested_port)
        print(f"    Using port {port} for TCP test")
        
        # Start server in background thread
        server_thread = threading.Thread(
            target=self._tcp_server,
            args=(host, port, duration + 10)  # Server runs 10s longer than test
        )
        server_thread.daemon = True
        server_thread.start()
        
        # Wait for server to start with timeout
        server_ready = False
        for i in range(50):  # Try for 5 seconds
            if time.time() - start_time > max_test_time:
                print(f"    Test timeout exceeded ({max_test_time}s), aborting")
                return torch.tensor([0.0], dtype=torch.float32)
                
            try:
                # Test if server is ready
                test_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                test_socket.settimeout(0.1)
                result = test_socket.connect_ex((host, port))
                test_socket.close()
                if result == 0:
                    server_ready = True
                    break
            except:
                pass
            time.sleep(0.1)
        
        if not server_ready:
            print(f"    Server failed to start within timeout")
            return torch.tensor([0.0], dtype=torch.float32)
        
        print(f"    Server ready, starting client test")
        
        # Run client test with timeout
        try:
            bandwidth_gbps = self._tcp_client(data, host, port, duration, max_test_time - (time.time() - start_time))
            return torch.tensor([bandwidth_gbps], dtype=torch.float32)
        except Exception as e:
            print(f"    TCP client test failed: {e}")
            return torch.tensor([0.0], dtype=torch.float32)
    
    def _tcp_server(self, host: str, port: int, duration: int):
        """TCP server implementation with status display"""
        try:
            server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server_socket.bind((host, port))
            server_socket.listen(1)
            server_socket.settimeout(duration)
            
            print(f"    TCP server listening on {host}:{port} for {duration}s")
            
            while True:
                try:
                    conn, addr = server_socket.accept()
                    print(f"    TCP server: Client connected from {addr}")
                    conn.settimeout(1.0)
                    
                    # Receive data with monitoring
                    total_received = 0
                    start_time = time.time()
                    last_print_time = start_time
                    
                    while True:
                        try:
                            data = conn.recv(65536)
                            if not data:
                                break
                            total_received += len(data)
                            
                            # Show receive progress
                            current_time = time.time()
                            if current_time - last_print_time >= 2.0:  # Every 2 seconds
                                elapsed = current_time - start_time
                                recv_rate = (total_received * 8) / (elapsed * 1e9) if elapsed > 0 else 0
                                print(f"    TCP server: Receiving data, rate: {recv_rate:.2f} Gbps, "
                                      f"total: {total_received//1024//1024:.1f}MB")
                                last_print_time = current_time
                                
                        except socket.timeout:
                            break
                        except Exception:
                            break
                    
                    elapsed = time.time() - start_time
                    final_rate = (total_received * 8) / (elapsed * 1e9) if elapsed > 0 else 0
                    print(f"    TCP server: Session completed, received {total_received//1024//1024:.1f}MB "
                          f"at {final_rate:.2f} Gbps")
                    conn.close()
                    
                except socket.timeout:
                    print(f"    TCP server: Timeout after {duration}s, shutting down")
                    break
                except Exception as e:
                    print(f"    TCP server: Error: {e}")
                    break
            
            server_socket.close()
            
        except Exception as e:
            print(f"    TCP server error: {e}")
    
    def _tcp_client(self, data: torch.Tensor, host: str, port: int, duration: int, max_time: float = None) -> float:
        """TCP client implementation with progress display and timeout"""
        if max_time is None:
            max_time = duration + 30
            
        test_start = time.time()
        
        try:
            client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            client_socket.settimeout(5.0)  # 5 second connection timeout
            
            print(f"    Connecting to {host}:{port}...")
            client_socket.connect((host, port))
            client_socket.settimeout(1.0)
            
            # Convert tensor to bytes
            data_bytes = data.numpy().tobytes()
            data_size = len(data_bytes)
            
            # Send data for specified duration with progress display
            start_time = time.time()
            total_sent = 0
            last_print_time = start_time
            print_interval = 2.0  # Print progress every 2 seconds
            
            print(f"    Starting TCP test: target duration {duration}s, buffer size {data_size//1024}KB")
            
            while True:
                current_time = time.time()
                
                # Check various timeout conditions
                if current_time - start_time >= duration:
                    print(f"    Test duration reached ({duration}s)")
                    break
                if current_time - test_start >= max_time:
                    print(f"    Maximum test time reached ({max_time:.1f}s)")
                    break
                    
                try:
                    sent = client_socket.send(data_bytes)
                    if sent == 0:
                        print(f"    Connection closed by server")
                        break
                    total_sent += sent
                    
                    # Display progress every 2 seconds
                    if current_time - last_print_time >= print_interval:
                        elapsed = current_time - start_time
                        progress_pct = min(100, (elapsed / duration) * 100)
                        current_bandwidth = (total_sent * 8) / (elapsed * 1e9) if elapsed > 0 else 0
                        print(f"    Progress: {progress_pct:.1f}% | Elapsed: {elapsed:.1f}s | "
                              f"Current bandwidth: {current_bandwidth:.2f} Gbps | "
                              f"Data sent: {total_sent//1024//1024:.1f}MB")
                        last_print_time = current_time
                        
                except socket.timeout:
                    # Timeout is normal, continue
                    continue
                except Exception as e:
                    print(f"    Send error: {e}")
                    break
            
            elapsed_time = time.time() - start_time
            client_socket.close()
            
            # Calculate final bandwidth in Gbps
            bandwidth_gbps = (total_sent * 8) / (elapsed_time * 1e9)
            print(f"    Completed: {elapsed_time:.2f}s | Final bandwidth: {bandwidth_gbps:.3f} Gbps | "
                  f"Total data: {total_sent//1024//1024:.1f}MB")
            return bandwidth_gbps
            
        except Exception as e:
            print(f"    TCP client error: {e}")
            return 0.0
    
    def _tcp_iperf3_test(self, inputs: List[torch.Tensor], 
                        params: Dict[str, Any]) -> torch.Tensor:
        """TCP bandwidth test using iperf3"""
        # Check if iperf3 is available
        try:
            subprocess.run(['iperf3', '--version'], capture_output=True, check=True)
        except (subprocess.SubprocessError, FileNotFoundError):
            print("iperf3 not available, skipping test")
            return torch.tensor([0.0], dtype=torch.float32)
        
        duration = params['duration']
        host = params['server_host']
        requested_port = params['server_port']
        
        # Find an available port
        port = self._find_available_port(requested_port)
        
        try:
            print(f"    Starting iperf3 test: server on port {port}, duration {duration}s")
            
            # Start iperf3 server in background
            server_cmd = f"iperf3 -s -p {port} -1"
            server_proc = subprocess.Popen(
                server_cmd.split(),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            # Wait for server to start
            print(f"    iperf3 server starting...")
            time.sleep(2)
            
            # Run iperf3 client with progress monitoring
            print(f"    Starting iperf3 client test...")
            client_cmd = f"iperf3 -c {host} -p {port} -t {duration} -f g -i 1"  # Add -i 1 for interval reports
            
            # Run with real-time output
            client_proc = subprocess.Popen(
                client_cmd.split(),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Monitor progress
            output_lines = []
            while client_proc.poll() is None:
                line = client_proc.stdout.readline()
                if line:
                    output_lines.append(line)
                    # Display progress if it's an interval report
                    if 'Gbits/sec' in line and 'sender' not in line and 'receiver' not in line:
                        parts = line.split()
                        for i, part in enumerate(parts):
                            if 'Gbits/sec' in part and i > 0:
                                try:
                                    bandwidth = float(parts[i-1])
                                    interval = parts[0] if '-' in parts[0] else "current"
                                    print(f"    iperf3 progress [{interval}]: {bandwidth:.2f} Gbps")
                                    break
                                except:
                                    continue
            
            # Get final output
            stdout, stderr = client_proc.communicate()
            full_output = ''.join(output_lines) + stdout
            
            # Parse bandwidth from output
            bandwidth_gbps = self._parse_iperf3_output(full_output)
            print(f"    iperf3 test completed: {bandwidth_gbps:.3f} Gbps")
            
            # Clean up server
            server_proc.terminate()
            server_proc.wait(timeout=5)
            
            return torch.tensor([bandwidth_gbps], dtype=torch.float32)
            
        except Exception as e:
            print(f"    iperf3 test failed: {e}")
            return torch.tensor([0.0], dtype=torch.float32)
    
    def _parse_iperf3_output(self, output: str) -> float:
        """Parse iperf3 output to extract bandwidth"""
        lines = output.strip().split('\n')
        for line in lines:
            if 'sender' in line or 'receiver' in line:
                # Look for bandwidth value in Gbits/sec
                parts = line.split()
                for i, part in enumerate(parts):
                    if 'Gbits/sec' in part and i > 0:
                        try:
                            return float(parts[i-1])
                        except:
                            continue
        return 0.0
    
    def _tcp_netcat_test(self, inputs: List[torch.Tensor], 
                        params: Dict[str, Any]) -> torch.Tensor:
        """TCP bandwidth test using netcat"""
        # Check if netcat is available
        try:
            subprocess.run(['nc', '-h'], capture_output=True, check=False)
        except FileNotFoundError:
            print("netcat not available, skipping test")
            return torch.tensor([0.0], dtype=torch.float32)
        
        data = inputs[0]
        duration = params['duration']
        host = params['server_host']
        requested_port = params['server_port']
        
        # Find an available port
        port = self._find_available_port(requested_port)
        
        try:
            # Start netcat server
            server_cmd = f"nc -l -p {port}"
            server_proc = subprocess.Popen(
                server_cmd.split(),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            # Wait for server to start
            time.sleep(1.0)
            
            # Create data file
            data_file = f"/tmp/tcp_test_data_{port}.bin"
            with open(data_file, 'wb') as f:
                f.write(data.numpy().tobytes())
            
            # Send data using netcat with timeout control
            start_time = time.time()
            total_sent = 0
            iterations = 0
            max_iterations = duration * 5  # Limit iterations
            
            while time.time() - start_time < duration and iterations < max_iterations:
                try:
                    client_cmd = f"timeout 1 nc {host} {port} < {data_file}"
                    result = subprocess.run(
                        client_cmd,
                        shell=True,
                        capture_output=True,
                        timeout=2
                    )
                    if result.returncode == 0:
                        total_sent += len(data.numpy().tobytes())
                    iterations += 1
                except subprocess.TimeoutExpired:
                    break
                except Exception:
                    break
            
            elapsed_time = time.time() - start_time
            
            # Clean up
            server_proc.terminate()
            server_proc.wait(timeout=5)
            if os.path.exists(data_file):
                os.remove(data_file)
            
            # Calculate bandwidth in Gbps
            if elapsed_time > 0:
                bandwidth_gbps = (total_sent * 8) / (elapsed_time * 1e9)
            else:
                bandwidth_gbps = 0.0
            return torch.tensor([bandwidth_gbps], dtype=torch.float32)
            
        except Exception as e:
            print(f"netcat test failed: {e}")
            return torch.tensor([0.0], dtype=torch.float32)
    
    def _find_available_port(self, start_port: int = None) -> int:
        """Find an available port for testing"""
        import random
        if start_port is None:
            start_port = self.port_range_start
        
        for port in range(start_port, self.port_range_end):
            try:
                test_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                test_socket.bind(('localhost', port))
                test_socket.close()
                return port
            except OSError:
                continue
        
        # If no port found in range, use random high port
        return random.randint(20000, 30000)
