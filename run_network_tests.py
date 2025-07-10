#!/usr/bin/env python3
"""
Network Performance Testing Suite
Main entry point for running network performance tests, validation, and environment checks
Combines all network-related functionality into a single script
"""

import sys
import os
import argparse
import subprocess
import re
import shutil
import time
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass

# Add src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

@dataclass
class NetworkDevice:
    """网络设备信息"""
    name: str
    type: str  # 'ethernet', 'infiniband', 'roce'
    speed: str  # e.g., '400000' for 400Gbps
    state: str  # 'up', 'down'
    mtu: int
    numa_node: Optional[int] = None
    pci_address: Optional[str] = None
    driver: Optional[str] = None
    firmware: Optional[str] = None
    capabilities: List[str] = None

@dataclass
class RDMADevice:
    """RDMA设备信息"""
    name: str
    node_guid: str
    sys_image_guid: str
    node_type: str
    physical_state: str
    link_layer: str
    port_state: str
    rate: str  # Link rate
    numa_node: Optional[int] = None
    pci_address: Optional[str] = None

def print_status(color_name: str, message: str):
    """Print colored status message"""
    colors = {
        'RED': '\033[0;31m',
        'GREEN': '\033[0;32m',
        'YELLOW': '\033[1;33m',
        'NC': '\033[0m'  # No Color
    }
    color = colors.get(color_name, colors['NC'])
    nc = colors['NC']
    print(f"{color}[INFO]{nc} {message}")

def print_error(message: str):
    """Print error message"""
    print(f"\033[0;31m[ERROR]\033[0m {message}")

def print_success(message: str):
    """Print success message"""
    print(f"\033[0;32m[SUCCESS]\033[0m {message}")

def print_warning(message: str):
    """Print warning message"""
    print(f"\033[1;33m[WARNING]\033[0m {message}")

def check_command_available(command: str) -> bool:
    """Check if a command is available in the system"""
    return shutil.which(command) is not None

def run_command(command: List[str], timeout: int = 10) -> Tuple[bool, str, str]:
    """Run a command and return success status, stdout, stderr"""
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
        return result.returncode == 0, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return False, "", "Command timed out"
    except Exception as e:
        return False, "", str(e)

def check_environment_dependencies():
    """Check environment dependencies and tools availability"""
    print("=" * 60)
    print("Checking Environment Dependencies")
    print("=" * 60)
    
    issues = []
    warnings = []
    
    # Check Python and basic dependencies
    print_status("YELLOW", "Checking Python dependencies...")
    
    if not check_command_available("python3"):
        issues.append("Python3 is not installed")
    else:
        print_success("Python3 is available")
    
    # Check PyTorch
    try:
        import torch
        print_success("PyTorch is available")
    except ImportError:
        warnings.append("PyTorch is not available - some PCIe tests may not work")
        print_warning("PyTorch is not available - some PCIe tests may not work")
    
    # Check NumPy
    try:
        import numpy
        print_success("NumPy is available")
    except ImportError:
        warnings.append("NumPy is not available - some tests may not work")
        print_warning("NumPy is not available - some tests may not work")
    
    # Check network testing tools
    print_status("YELLOW", "Checking network testing tools...")
    
    # Check for iperf3
    if check_command_available("iperf3"):
        print_success("iperf3 is available")
    else:
        warnings.append("iperf3 is not available - TCP iperf3 tests will be skipped")
        print_warning("iperf3 is not available - TCP iperf3 tests will be skipped")
    
    # Check for netcat
    if check_command_available("nc"):
        print_success("netcat is available")
    else:
        warnings.append("netcat is not available - TCP netcat tests will be skipped")
        print_warning("netcat is not available - TCP netcat tests will be skipped")
    
    # Check for RDMA tools
    if check_command_available("ibv_devices"):
        print_success("RDMA tools are available")
        
        # Check for RDMA devices
        success, stdout, _ = run_command(["ibv_devices"])
        if success and "device" in stdout:
            print_success("RDMA devices found")
        else:
            warnings.append("No RDMA devices found - RDMA tests will be skipped")
            print_warning("No RDMA devices found - RDMA tests will be skipped")
    else:
        warnings.append("RDMA tools are not available - RDMA tests will be skipped")
        print_warning("RDMA tools are not available - RDMA tests will be skipped")
    
    # Check for RDMA perftest tools
    if check_command_available("ib_write_bw"):
        print_success("RDMA perftest tools are available")
    else:
        warnings.append("RDMA perftest tools are not available - some RDMA tests will be skipped")
        print_warning("RDMA perftest tools are not available - some RDMA tests will be skipped")
    
    # Check for nvidia-smi (GPU support)
    if check_command_available("nvidia-smi"):
        print_success("NVIDIA GPU tools are available")
        
        # Check for GPU devices
        success, stdout, _ = run_command(["nvidia-smi", "-L"])
        if success and "GPU" in stdout:
            print_success("GPU devices found")
        else:
            warnings.append("No GPU devices found - PCIe tests will be skipped")
            print_warning("No GPU devices found - PCIe tests will be skipped")
    else:
        warnings.append("NVIDIA GPU tools are not available - PCIe tests will be skipped")
        print_warning("NVIDIA GPU tools are not available - PCIe tests will be skipped")
    
    return issues, warnings

def validate_operator_files():
    """Validate that all operator files are present and can be imported"""
    print_status("YELLOW", "Validating operator files...")
    
    operators = [
        "src/operators/tcp_bandwidth_operator.py",
        "src/operators/rdma_bandwidth_operator.py", 
        "src/operators/pcie_bandwidth_operator.py",
        "src/operators/network_stress_operator.py"
    ]
    
    issues = []
    
    for operator in operators:
        if os.path.exists(operator):
            print_success(f"{os.path.basename(operator)} exists")
            
            # Check if file is not empty
            if os.path.getsize(operator) > 0:
                print_success(f"{os.path.basename(operator)} is not empty")
            else:
                issues.append(f"{os.path.basename(operator)} is empty")
                print_error(f"{os.path.basename(operator)} is empty")
        else:
            issues.append(f"{os.path.basename(operator)} is missing")
            print_error(f"{os.path.basename(operator)} is missing")
    
    return issues

def test_operator_imports():
    """Test that all operators can be imported successfully"""
    print_status("YELLOW", "Testing operator imports...")
    
    import_tests = [
        ("TCP", "operators.tcp_bandwidth_operator", "TCPBandwidthOperator"),
        ("RDMA", "operators.rdma_bandwidth_operator", "RDMABandwidthOperator"),
        ("PCIe", "operators.pcie_bandwidth_operator", "PCIeBandwidthOperator"),
        ("Network Stress", "operators.network_stress_operator", "NetworkStressOperator")
    ]
    
    issues = []
    
    for name, module_name, class_name in import_tests:
        try:
            module = __import__(module_name, fromlist=[class_name])
            operator_class = getattr(module, class_name)
            print_success(f"{name} operator imports correctly")
        except Exception as e:
            issues.append(f"{name} operator import failed: {e}")
            print_error(f"{name} operator import failed: {e}")
    
    return issues

def test_operator_instantiation():
    """Test that all operators can be instantiated successfully"""
    print_status("YELLOW", "Testing operator instantiation...")
    
    issues = []
    
    try:
        from operators.tcp_bandwidth_operator import TCPBandwidthOperator
        from operators.rdma_bandwidth_operator import RDMABandwidthOperator
        from operators.pcie_bandwidth_operator import PCIeBandwidthOperator
        from operators.network_stress_operator import NetworkStressOperator
        
        operators = [
            ("TCP", TCPBandwidthOperator),
            ("RDMA", RDMABandwidthOperator),
            ("PCIe", PCIeBandwidthOperator),
            ("Network Stress", NetworkStressOperator)
        ]
        
        for name, operator_class in operators:
            try:
                operator = operator_class()
                print_success(f"{name} operator instantiated")
            except Exception as e:
                issues.append(f"{name} operator instantiation failed: {e}")
                print_error(f"{name} operator instantiation failed: {e}")
                
    except ImportError as e:
        issues.append(f"Failed to import operators: {e}")
        print_error(f"Failed to import operators: {e}")
    
    return issues

def run_deployment_validation():
    """Run comprehensive deployment validation"""
    print("=" * 80)
    print("Network Operators Deployment Validation")
    print("=" * 80)
    
    # Check if we're in the correct directory
    if not os.path.exists("README.md") or not os.path.exists("src/operators"):
        print_error("Please run this script from the op_bench project root directory")
        return False
    
    print_status("GREEN", "Starting network operators deployment validation...")
    
    all_issues = []
    all_warnings = []
    
    # 1. Check environment dependencies
    issues, warnings = check_environment_dependencies()
    all_issues.extend(issues)
    all_warnings.extend(warnings)
    
    # 2. Validate operator files
    issues = validate_operator_files()
    all_issues.extend(issues)
    
    # 3. Test operator imports
    issues = test_operator_imports()
    all_issues.extend(issues)
    
    # 4. Test operator instantiation
    issues = test_operator_instantiation()
    all_issues.extend(issues)
    
    # 5. Check file permissions
    print_status("YELLOW", "Checking file permissions...")
    if not os.access(__file__, os.X_OK):
        print_warning("Making run_network_tests.py executable")
        try:
            os.chmod(__file__, 0o755)
        except Exception as e:
            all_warnings.append(f"Failed to make file executable: {e}")
    
    # Print summary
    print("\n" + "=" * 80)
    print("Deployment Validation Summary")
    print("=" * 80)
    
    if all_issues:
        print_error("Deployment validation failed with the following issues:")
        for issue in all_issues:
            print(f"  ✗ {issue}")
        return False
    else:
        print_success("Deployment validation completed successfully!")
        
        if all_warnings:
            print_warning("Note: The following warnings were encountered:")
            for warning in all_warnings:
                print(f"  ⚠ {warning}")
        
        print("\nNext steps:")
        print("1. Run network tests: python run_network_tests.py --all")
        print("2. Use quick mode: python run_network_tests.py --quick")
        print("3. Test specific protocols: python run_network_tests.py --tcp --rdma")
        print("4. Run with custom duration: python run_network_tests.py --duration 5")
        
        return True

def check_network_configuration():
    """Check detailed network configuration similar to network_config_checker.py"""
    print("=" * 80)
    print("Network Configuration Check")
    print("=" * 80)
    
    config = {
        'network_devices': _check_network_devices(),
        'rdma_devices': _check_rdma_devices(),
        'gpu_devices': _check_gpu_devices(),
        'system_info': _get_system_info()
    }
    
    _print_configuration_summary(config)
    return config

def _check_network_devices() -> List[Dict[str, Any]]:
    """Check network devices"""
    print_status("YELLOW", "Checking network devices...")
    devices = []
    
    try:
        success, stdout, _ = run_command(['ip', 'link', 'show'])
        if success:
            # Parse ip link output
            for line in stdout.split('\n'):
                if ':' in line and 'state' in line.lower():
                    parts = line.split(':')
                    if len(parts) >= 2:
                        name = parts[1].strip().split('@')[0]  # Remove @veth part if present
                        if name and not name.startswith('lo'):  # Skip loopback
                            device_info = {'name': name, 'state': 'unknown', 'speed': 'unknown', 'mtu': 1500}
                            
                            # Get more details with ethtool
                            success2, stdout2, _ = run_command(['ethtool', name], timeout=5)
                            if success2 and 'Speed:' in stdout2:
                                for ethtool_line in stdout2.split('\n'):
                                    if 'Speed:' in ethtool_line:
                                        speed_match = re.search(r'(\d+)', ethtool_line)
                                        if speed_match:
                                            device_info['speed'] = speed_match.group(1)
                            
                            devices.append(device_info)
    except Exception as e:
        print_warning(f"Error checking network devices: {e}")
    
    return devices

def _check_rdma_devices() -> List[Dict[str, Any]]:
    """Check RDMA devices"""
    print_status("YELLOW", "Checking RDMA devices...")
    devices = []
    
    try:
        if check_command_available('ibv_devices'):
            success, stdout, _ = run_command(['ibv_devices'])
            if success:
                for line in stdout.split('\n'):
                    line = line.strip()
                    if line and not line.startswith('device') and line != '':
                        devices.append({'name': line, 'status': 'available'})
        else:
            print_warning("RDMA tools not available")
    except Exception as e:
        print_warning(f"Error checking RDMA devices: {e}")
    
    return devices

def _check_gpu_devices() -> List[Dict[str, Any]]:
    """Check GPU devices"""
    print_status("YELLOW", "Checking GPU devices...")
    devices = []
    
    try:
        if check_command_available('nvidia-smi'):
            success, stdout, _ = run_command(['nvidia-smi', '-L'])
            if success:
                for line in stdout.split('\n'):
                    if 'GPU' in line and ':' in line:
                        gpu_match = re.search(r'GPU (\d+): (.+?) \(', line)
                        if gpu_match:
                            devices.append({
                                'index': gpu_match.group(1),
                                'name': gpu_match.group(2),
                                'status': 'available'
                            })
        else:
            print_warning("NVIDIA tools not available")
    except Exception as e:
        print_warning(f"Error checking GPU devices: {e}")
    
    return devices

def _get_system_info() -> Dict[str, Any]:
    """Get system information"""
    info = {}
    
    # Get kernel version
    try:
        success, stdout, _ = run_command(['uname', '-r'])
        if success:
            info['kernel_version'] = stdout.strip()
    except:
        pass
    
    # Get CPU info
    try:
        success, stdout, _ = run_command(['nproc'])
        if success:
            info['cpu_cores'] = int(stdout.strip())
    except:
        pass
    
    return info

def _print_configuration_summary(config: Dict[str, Any]):
    """Print configuration summary"""
    print("\n" + "=" * 80)
    print("Configuration Summary")
    print("=" * 80)
    
    # Network devices summary
    network_devices = config.get('network_devices', [])
    print(f"\nNetwork devices ({len(network_devices)} found):")
    for device in network_devices:
        speed_info = f"{device['speed']}Mb/s" if device['speed'] != 'unknown' else 'unknown speed'
        print(f"  • {device['name']}: {speed_info}")
    
    # RDMA devices summary
    rdma_devices = config.get('rdma_devices', [])
    print(f"\nRDMA devices ({len(rdma_devices)} found):")
    for device in rdma_devices:
        print(f"  • {device['name']}: {device['status']}")
    
    # GPU devices summary
    gpu_devices = config.get('gpu_devices', [])
    print(f"\nGPU devices ({len(gpu_devices)} found):")
    for device in gpu_devices:
        print(f"  • GPU{device['index']}: {device['name']}")
    
    # System info
    system_info = config.get('system_info', {})
    if system_info.get('kernel_version'):
        print(f"\nKernel version: {system_info['kernel_version']}")
    if system_info.get('cpu_cores'):
        print(f"CPU cores: {system_info['cpu_cores']}")

def import_operators():
    """Import operators with proper error handling"""
    try:
        from operators.tcp_bandwidth_operator import TCPBandwidthOperator
        from operators.rdma_bandwidth_operator import RDMABandwidthOperator
        from operators.pcie_bandwidth_operator import PCIeBandwidthOperator
        from operators.network_stress_operator import NetworkStressOperator
        return TCPBandwidthOperator, RDMABandwidthOperator, PCIeBandwidthOperator, NetworkStressOperator
    except ImportError as e:
        print(f"Error importing operators: {e}")
        print("Please ensure you're running in the correct Python environment with PyTorch installed.")
        sys.exit(1)
    """Run quick network benchmark across all available protocols"""
    print("\n" + "="*60)
    print("Running Quick Network Benchmark")
    print("="*60)
    
    try:
        from operators.network_stress_operator import NetworkStressOperator
        network_op = NetworkStressOperator()
        
        # Get system capabilities
        system_info = network_op.get_system_info()
        print("System capabilities:")
        print(f"  - TCP: {system_info['tcp_info']['available']}")
        print(f"  - RDMA: {system_info['rdma_info']['available']}")
        print(f"  - PCIe: {system_info['pcie_info']['available']}")
        
        # Run quick benchmark
        print("\nRunning quick benchmark tests...")
        benchmark_results = network_op.run_quick_benchmark()
        
        print("\nQuick benchmark results:")
        for protocol, bandwidth in benchmark_results.items():
            print(f"  - {protocol}: {bandwidth:.2f} Gbps")
        
        return benchmark_results
        
    except Exception as e:
        print(f"Quick benchmark failed: {e}")
        return {}

def test_operators():
    """Test all network operators functionality"""
    print("\n" + "="*60)
    print("Testing Network Operators Functionality")
    print("="*60)
    
    operators_config = [
        ('TCP Bandwidth', 'operators.tcp_bandwidth_operator', 'TCPBandwidthOperator'),
        ('RDMA Bandwidth', 'operators.rdma_bandwidth_operator', 'RDMABandwidthOperator'),
        ('PCIe Bandwidth', 'operators.pcie_bandwidth_operator', 'PCIeBandwidthOperator'),
        ('Network Stress', 'operators.network_stress_operator', 'NetworkStressOperator')
    ]
    
    all_passed = True
    
    for name, module_name, class_name in operators_config:
        print(f"\n{'='*50}")
        print(f"Testing {name} Operator")
        print(f"{'='*50}")
        
        try:
            # Import and instantiate
            module = __import__(module_name, fromlist=[class_name])
            operator_class = getattr(module, class_name)
            operator = operator_class()
            print(f"✓ {name} operator created successfully")
            
            # Get test cases
            test_cases = operator.get_test_cases()
            print(f"✓ Found {len(test_cases)} test cases")
            
            # Get implementations
            implementations = operator.get_available_implementations()
            print(f"✓ Available implementations: {implementations}")
            
            # Test first case if available
            if test_cases:
                test_case = test_cases[0]
                print(f"✓ Testing case: {test_case.name}")
                
                # Generate inputs
                inputs = operator.generate_inputs(test_case)
                print(f"✓ Generated inputs: {inputs[0].shape}")
                
                # Calculate FLOPS
                flops = operator.calculate_flops(test_case)
                print(f"✓ FLOPS calculation: {flops}")
                
                # For specific operators, run additional tests
                if name == 'RDMA Bandwidth':
                    devices = operator.rdma_devices
                    print(f"✓ Found {len(devices)} RDMA devices")
                    if devices:
                        print(f"✓ First device: {devices[0]}")
                        device_info = operator.get_rdma_device_info()
                        print(f"✓ Device info: {len(device_info)} devices")
                    else:
                        print("! No RDMA devices found - this is expected on non-RDMA systems")
                
                elif name == 'PCIe Bandwidth':
                    gpu_devices = operator.gpu_devices
                    print(f"✓ Found {len(gpu_devices)} GPU devices")
                    if gpu_devices:
                        print(f"✓ First GPU: {gpu_devices[0]['name']}")
                        topology = operator.get_pcie_topology()
                        print(f"✓ PCIe topology: {len(topology.get('pcie_devices', []))} devices")
                    else:
                        print("! No GPU devices found - this is expected on non-GPU systems")
                
                elif name == 'Network Stress':
                    system_info = operator.get_system_info()
                    print(f"✓ System info:")
                    print(f"  - TCP available: {system_info['tcp_info']['available']}")
                    print(f"  - RDMA available: {system_info['rdma_info']['available']}")
                    print(f"  - PCIe available: {system_info['pcie_info']['available']}")
            
            print(f"✓ {name} operator test completed")
            
        except Exception as e:
            print(f"✗ {name} operator test failed: {e}")
            all_passed = False
    
    print("\n" + "="*60)
    if all_passed:
        print("All operator tests completed successfully!")
    else:
        print("Some operator tests failed - see details above")
    print("="*60)
    
    return all_passed

def run_tcp_tests(quick_mode=False, test_duration=3):
    """Run TCP bandwidth tests"""
    print("\n" + "="*60)
    print("Running TCP Bandwidth Tests")
    print("="*60)
    
    try:
        from operators.tcp_bandwidth_operator import TCPBandwidthOperator
        tcp_op = TCPBandwidthOperator()
        test_cases = tcp_op.get_test_cases()
        
        for test_case in test_cases[:3]:  # Run first 3 test cases
            print(f"\nRunning test case: {test_case.name}")
            print(f"Description: {test_case.description}")
            
            # Reduce duration for faster testing
            original_duration = test_case.additional_params.get('duration', 10)
            test_case.additional_params['duration'] = min(test_duration, original_duration)
            duration = test_case.additional_params['duration']
            
            # Set test runs based on mode
            warmup_runs = 1 if quick_mode else 2
            test_runs = 2 if quick_mode else 3
            
            print(f"Test duration: {duration}s per implementation ({'quick mode' if quick_mode else 'standard mode'})")
            print(f"Runs: {warmup_runs} warmup + {test_runs} test runs per implementation")
            print(f"Progress will be shown below:")
            
            try:
                # Add timeout protection
                import signal
                
                def timeout_handler(signum, frame):
                    raise TimeoutError(f"Test timed out after {duration + 30}s")
                
                # Set timeout to test duration + 30s buffer
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(duration + 30)
                
                results = tcp_op.run_comparison(test_case, warmup_runs=warmup_runs, test_runs=test_runs)
                
                # Clear timeout
                signal.alarm(0)
                
                print(f"\nResults:")
                for result in results:
                    status = "✓" if result.available else "✗"
                    correctness = "✓" if result.correct else "✗"
                    # For TCP bandwidth tests, show only bandwidth (not GFLOPS)
                    if hasattr(result, 'result') and result.result is not None:
                        actual_bandwidth = float(result.result[0])
                        print(f"  {status} {result.name}: {actual_bandwidth:.2f} Gbps, "
                              f"time: {result.avg_time_ms:.0f}ms, correct: {correctness}")
                    else:
                        # For tests that don't return bandwidth (like iperf3/netcat when unavailable)
                        print(f"  {status} {result.name}: Test skipped or failed, "
                              f"time: {result.avg_time_ms:.0f}ms")
                    
            except TimeoutError as e:
                print(f"  ✗ Test timed out: {e}")
            except Exception as e:
                print(f"  ✗ Test failed: {e}")
    
    except ImportError as e:
        print(f"  ✗ Cannot import TCP operator: {e}")
    except Exception as e:
        print(f"  ✗ TCP tests failed: {e}")

def run_rdma_tests(quick_mode=False, test_duration=3):
    """Run RDMA bandwidth tests"""
    print("\n" + "="*60)
    print("Running RDMA Bandwidth Tests")
    print("="*60)
    
    try:
        from operators.rdma_bandwidth_operator import RDMABandwidthOperator
        rdma_op = RDMABandwidthOperator()
        
        if not rdma_op.rdma_devices:
            print("No RDMA devices found. Skipping RDMA tests.")
            return
        
        print(f"Found {len(rdma_op.rdma_devices)} RDMA device(s)")
        test_cases = rdma_op.get_test_cases()
        
        for test_case in test_cases[:2]:  # Run first 2 test cases
            print(f"\nRunning test case: {test_case.name}")
            print(f"Description: {test_case.description}")
            
            # Reduce duration for faster testing
            test_case.additional_params['duration'] = test_duration
            
            # Set test runs based on mode
            warmup_runs = 1 if quick_mode else 2
            test_runs = 2 if quick_mode else 3
            
            try:
                results = rdma_op.run_comparison(test_case, warmup_runs=warmup_runs, test_runs=test_runs)
                
                print(f"\nResults:")
                for result in results:
                    status = "✓" if result.available else "✗"
                    correctness = "✓" if result.correct else "✗"
                    if hasattr(result, 'result') and result.result is not None:
                        actual_bandwidth = float(result.result[0])
                        print(f"  {status} {result.name}: {actual_bandwidth:.2f} Gbps, "
                              f"time: {result.avg_time_ms:.0f}ms, correct: {correctness}")
                    else:
                        print(f"  {status} {result.name}: Test skipped or failed")
                        
            except Exception as e:
                print(f"  ✗ Test failed: {e}")
    
    except ImportError as e:
        print(f"  ✗ Cannot import RDMA operator: {e}")
    except Exception as e:
        print(f"  ✗ RDMA tests failed: {e}")

def run_pcie_tests(quick_mode=False, test_duration=3):
    """Run PCIe bandwidth tests"""
    print("\n" + "="*60)
    print("Running PCIe Bandwidth Tests")
    print("="*60)
    
    try:
        from operators.pcie_bandwidth_operator import PCIeBandwidthOperator
        pcie_op = PCIeBandwidthOperator()
        
        if not pcie_op.gpu_devices:
            print("No GPU devices found. Skipping PCIe tests.")
            return
        
        print(f"Found {len(pcie_op.gpu_devices)} GPU device(s)")
        test_cases = pcie_op.get_test_cases()
        
        for test_case in test_cases[:2]:  # Run first 2 test cases
            print(f"\nRunning test case: {test_case.name}")
            print(f"Description: {test_case.description}")
            
            # Set test runs based on mode
            warmup_runs = 1 if quick_mode else 2
            test_runs = 2 if quick_mode else 3
            
            try:
                results = pcie_op.run_comparison(test_case, warmup_runs=warmup_runs, test_runs=test_runs)
                
                print(f"\nResults:")
                for result in results:
                    status = "✓" if result.available else "✗"
                    correctness = "✓" if result.correct else "✗"
                    if hasattr(result, 'result') and result.result is not None:
                        actual_bandwidth = float(result.result[0])
                        print(f"  {status} {result.name}: {actual_bandwidth:.2f} Gbps, "
                              f"time: {result.avg_time_ms:.0f}ms, correct: {correctness}")
                    else:
                        print(f"  {status} {result.name}: Test skipped or failed")
                        
            except Exception as e:
                print(f"  ✗ Test failed: {e}")
    
    except ImportError as e:
        print(f"  ✗ Cannot import PCIe operator: {e}")
    except Exception as e:
        print(f"  ✗ PCIe tests failed: {e}")

def run_network_stress_tests(quick_mode=False, test_duration=3):
    """Run network stress tests"""
    print("\n" + "="*60)
    print("Running Network Stress Tests")
    print("="*60)
    
    try:
        from operators.network_stress_operator import NetworkStressOperator
        network_op = NetworkStressOperator()
        
        # Show system capabilities
        system_info = network_op.get_system_info()
        print("System capabilities:")
        print(f"  - TCP: {system_info['tcp_info']['available']}")
        print(f"  - RDMA: {system_info['rdma_info']['available']}")
        print(f"  - PCIe: {system_info['pcie_info']['available']}")
        
        test_cases = network_op.get_test_cases()
        
        for test_case in test_cases[:1]:  # Run first test case
            print(f"\nRunning test case: {test_case.name}")
            print(f"Description: {test_case.description}")
            
            # Set test runs based on mode
            warmup_runs = 1 if quick_mode else 2
            test_runs = 2 if quick_mode else 3
            
            try:
                results = network_op.run_comparison(test_case, warmup_runs=warmup_runs, test_runs=test_runs)
                
                print(f"\nResults:")
                for result in results:
                    status = "✓" if result.available else "✗"
                    correctness = "✓" if result.correct else "✗"
                    if hasattr(result, 'result') and result.result is not None:
                        actual_bandwidth = float(result.result[0])
                        print(f"  {status} {result.name}: {actual_bandwidth:.2f} Gbps, "
                              f"time: {result.avg_time_ms:.0f}ms, correct: {correctness}")
                    else:
                        print(f"  {status} {result.name}: Test skipped or failed")
                        
            except Exception as e:
                print(f"  ✗ Test failed: {e}")
    
    except ImportError as e:
        print(f"  ✗ Cannot import Network Stress operator: {e}")
    except Exception as e:
        print(f"  ✗ Network stress tests failed: {e}")
    
    test_cases = rdma_op.get_test_cases()
    
    for test_case in test_cases[:3]:  # Run first 3 test cases
        print(f"\nRunning test case: {test_case.name}")
        print(f"Description: {test_case.description}")
        
        try:
            results = rdma_op.run_comparison(test_case, warmup_runs=1, test_runs=3)
            
            print(f"Results:")
            for result in results:
                status = "✓" if result.available else "✗"
                correctness = "✓" if result.correct else "✗"
                # For RDMA bandwidth tests, show only bandwidth
                if hasattr(result, 'result') and result.result is not None:
                    actual_bandwidth = float(result.result[0])
                    print(f"  {status} {result.name}: {actual_bandwidth:.2f} Gbps, "
                          f"time: {result.avg_time_ms:.0f}ms, correct: {correctness}")
                else:
                    print(f"  {status} {result.name}: Test skipped or failed, "
                          f"time: {result.avg_time_ms:.0f}ms")
                
        except Exception as e:
            print(f"  ✗ Test failed: {e}")

def run_network_stress_tests():
    """Run comprehensive network stress tests"""
    print("\n" + "="*60)
    print("Running Network Stress Tests")
    print("="*60)
    
    _, _, _, NetworkStressOperator = import_operators()
    network_op = NetworkStressOperator()
    
    # Show system info
    system_info = network_op.get_system_info()
    print(f"System Information:")
    print(f"  - TCP available: {system_info['tcp_info']['available']}")
    print(f"  - RDMA available: {system_info['rdma_info']['available']}")
    print(f"  - PCIe available: {system_info['pcie_info']['available']}")
    
    # Run quick benchmark
    print(f"\nRunning quick benchmark...")
    benchmark_results = network_op.run_quick_benchmark()
    print(f"Quick benchmark results:")
    for protocol, bandwidth in benchmark_results.items():
        print(f"  - {protocol}: {bandwidth:.2f} Gbps")
    
    # Run stress tests
    test_cases = network_op.get_test_cases()
    
    for test_case in test_cases:
        print(f"\nRunning stress test: {test_case.name}")
        print(f"Description: {test_case.description}")
        
        try:
            results = network_op.run_comparison(test_case, warmup_runs=1, test_runs=3)
            
            print(f"Results:")
            for result in results:
                status = "✓" if result.available else "✗"
                correctness = "✓" if result.correct else "✗"
                # For network stress tests, show only bandwidth
                if hasattr(result, 'result') and result.result is not None:
                    actual_bandwidth = float(result.result[0])
                    print(f"  {status} {result.name}: {actual_bandwidth:.2f} Gbps, "
                          f"time: {result.avg_time_ms:.0f}ms, correct: {correctness}")
                else:
                    print(f"  {status} {result.name}: Test skipped or failed, "
                          f"time: {result.avg_time_ms:.0f}ms")
                
        except Exception as e:
            print(f"  ✗ Test failed: {e}")

def check_dependencies():
    """Check system dependencies and tools"""
    print("="*60)
    print("Checking System Dependencies")
    print("="*60)
    
    dependencies = {
        'python3': 'Python 3',
        'iperf3': 'iPerf3 (TCP bandwidth testing)',
        'nc': 'netcat (TCP testing)',
        'ibv_devices': 'RDMA tools (InfiniBand Verbs)',
        'ib_write_bw': 'RDMA perftest tools',
        'nvidia-smi': 'NVIDIA GPU tools',
        'lspci': 'PCI utilities',
        'numactl': 'NUMA control utilities'
    }
    
    results = {}
    for cmd, description in dependencies.items():
        try:
            result = subprocess.run([cmd, '--version' if cmd in ['python3', 'iperf3'] else '--help'], 
                                  capture_output=True, timeout=5)
            available = result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            if cmd == 'ibv_devices':
                # Special check for ibv_devices
                try:
                    result = subprocess.run([cmd], capture_output=True, timeout=5)
                    available = result.returncode == 0
                except:
                    available = False
            else:
                available = False
        
        status = "✓" if available else "✗"
        print(f"  {status} {description}: {'Available' if available else 'Not found'}")
        results[cmd] = available
    
    print(f"\nSummary: {sum(results.values())}/{len(results)} dependencies available")
    return results

def check_python_packages():
    """Check Python package dependencies"""
    print("\n" + "="*60)
    print("Checking Python Package Dependencies")
    print("="*60)
    
    packages = {
        'torch': 'PyTorch (GPU operations)',
        'numpy': 'NumPy (numerical operations)',
        'psutil': 'Process utilities'
    }
    
    results = {}
    for package, description in packages.items():
        try:
            __import__(package)
            available = True
        except ImportError:
            available = False
        
        status = "✓" if available else "✗"
        print(f"  {status} {description}: {'Available' if available else 'Not installed'}")
        results[package] = available
    
    return results

def check_hardware():
    """Check available hardware"""
    print("\n" + "="*60)
    print("Checking Hardware Availability")
    print("="*60)
    
    hardware = {}
    
    # Check RDMA devices
    try:
        result = subprocess.run(['ibv_devices'], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            rdma_devices = []
            for line in result.stdout.strip().split('\n')[1:]:  # Skip header
                if line.strip():
                    parts = line.split()
                    if len(parts) >= 1:
                        rdma_devices.append(parts[0])
            hardware['rdma_devices'] = rdma_devices
            print(f"  ✓ RDMA devices: {len(rdma_devices)} found ({', '.join(rdma_devices)})")
        else:
            hardware['rdma_devices'] = []
            print(f"  ✗ RDMA devices: None found")
    except:
        hardware['rdma_devices'] = []
        print(f"  ✗ RDMA devices: Cannot check (tools not available)")
    
    # Check GPU devices
    try:
        result = subprocess.run(['nvidia-smi', '-L'], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            gpu_devices = []
            for line in result.stdout.strip().split('\n'):
                if 'GPU' in line:
                    match = re.search(r'GPU \d+: (.+) \(', line)
                    if match:
                        gpu_devices.append(match.group(1))
            hardware['gpu_devices'] = gpu_devices
            print(f"  ✓ GPU devices: {len(gpu_devices)} found")
            for i, gpu in enumerate(gpu_devices):
                print(f"    - GPU {i}: {gpu}")
        else:
            hardware['gpu_devices'] = []
            print(f"  ✗ GPU devices: None found")
    except:
        hardware['gpu_devices'] = []
        print(f"  ✗ GPU devices: Cannot check (nvidia-smi not available)")
    
    # Check network interfaces
    try:
        import socket
        hostname = socket.gethostname()
        local_ip = socket.gethostbyname(hostname)
        hardware['network_interfaces'] = [local_ip]
        print(f"  ✓ Network interfaces: Available (local IP: {local_ip})")
    except:
        hardware['network_interfaces'] = []
        print(f"  ✗ Network interfaces: Cannot determine")
    
    return hardware

def validate_operators():
    """Validate that all network operators can be imported and instantiated"""
    print("\n" + "="*60)
    print("Validating Network Operators")
    print("="*60)
    
    operators = [
        ('TCP Bandwidth', 'operators.tcp_bandwidth_operator', 'TCPBandwidthOperator'),
        ('RDMA Bandwidth', 'operators.rdma_bandwidth_operator', 'RDMABandwidthOperator'),
        ('PCIe Bandwidth', 'operators.pcie_bandwidth_operator', 'PCIeBandwidthOperator'),
        ('Network Stress', 'operators.network_stress_operator', 'NetworkStressOperator')
    ]
    
    results = {}
    for name, module_name, class_name in operators:
        try:
            # Import the module
            module = __import__(module_name, fromlist=[class_name])
            operator_class = getattr(module, class_name)
            
            # Instantiate the operator
            operator = operator_class()
            
            # Check basic functionality
            test_cases = operator.get_test_cases()
            implementations = operator.get_available_implementations()
            
            print(f"  ✓ {name} Operator:")
            print(f"    - Import: Success")
            print(f"    - Instantiation: Success")
            print(f"    - Test cases: {len(test_cases)}")
            print(f"    - Implementations: {len(implementations)}")
            
            # Test input generation for first test case
            if test_cases:
                try:
                    inputs = operator.generate_inputs(test_cases[0])
                    print(f"    - Input generation: Success ({inputs[0].shape})")
                except Exception as e:
                    print(f"    - Input generation: Failed ({e})")
            
            results[name] = True
            
        except Exception as e:
            print(f"  ✗ {name} Operator: Failed ({e})")
            results[name] = False
    
    success_count = sum(results.values())
    total_count = len(results)
    print(f"\nValidation Summary: {success_count}/{total_count} operators working")
    
    return results

def check_environment():
    """Comprehensive environment check"""
    print("Network Testing Environment Check")
    print("="*60)
    
    # Check dependencies
    deps = check_dependencies()
    
    # Check Python packages
    packages = check_python_packages()
    
    # Check hardware
    hardware = check_hardware()
    
    # Validate operators
    operators = validate_operators()
    
    # Overall summary
    print("\n" + "="*60)
    print("Environment Check Summary")
    print("="*60)
    
    total_deps = len(deps)
    available_deps = sum(deps.values())
    print(f"System Dependencies: {available_deps}/{total_deps} available")
    
    total_packages = len(packages)
    available_packages = sum(packages.values())
    print(f"Python Packages: {available_packages}/{total_packages} available")
    
    print(f"RDMA Devices: {len(hardware.get('rdma_devices', []))}")
    print(f"GPU Devices: {len(hardware.get('gpu_devices', []))}")
    print(f"Network Interfaces: {len(hardware.get('network_interfaces', []))}")
    
    total_operators = len(operators)
    working_operators = sum(operators.values())
    print(f"Network Operators: {working_operators}/{total_operators} working")
    
    # Recommendations
    print("\n" + "="*60)
    print("Recommendations")
    print("="*60)
    
    if not deps.get('iperf3'):
        print("- Install iperf3 for TCP bandwidth testing: sudo apt install iperf3")
    
    if not deps.get('ibv_devices'):
        print("- Install RDMA tools for InfiniBand testing: sudo apt install infiniband-diags")
    
    if not packages.get('torch'):
        print("- Install PyTorch for GPU operations: pip install torch")
    
    if not hardware.get('rdma_devices'):
        print("- RDMA tests will be skipped (no RDMA devices found)")
    
    if not hardware.get('gpu_devices'):
        print("- PCIe/GPU tests will be skipped (no GPU devices found)")
    
    print(f"\nEnvironment readiness: {'Ready' if working_operators >= 2 else 'Needs setup'}")
    
    return {
        'dependencies': deps,
        'packages': packages,
        'hardware': hardware,
        'operators': operators
    }

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Network Performance Testing Suite - Unified Network Testing Tool')
    
    # Test selection options
    parser.add_argument('--tcp', action='store_true', help='Run TCP bandwidth tests')
    parser.add_argument('--rdma', action='store_true', help='Run RDMA bandwidth tests')
    parser.add_argument('--pcie', action='store_true', help='Run PCIe bandwidth tests')
    parser.add_argument('--stress', action='store_true', help='Run network stress tests')
    parser.add_argument('--all', action='store_true', help='Run all network tests')
    
    # Test mode options
    parser.add_argument('--quick', action='store_true', help='Run quick tests (faster, less accuracy)')
    parser.add_argument('--duration', type=int, default=3, help='Test duration in seconds (default: 3)')
    
    # Environment and validation options
    parser.add_argument('--validate', action='store_true', help='Run deployment validation (check environment)')
    parser.add_argument('--config', action='store_true', help='Check network configuration')
    parser.add_argument('--test-operators', action='store_true', help='Test operator functionality')
    parser.add_argument('--env-check', action='store_true', help='Run environment readiness check')
    
    # Legacy compatibility options
    parser.add_argument('--deploy', action='store_true', help='Run deployment validation (alias for --validate)')
    parser.add_argument('--check-env', action='store_true', help='Check environment (alias for --env-check)')
    
    args = parser.parse_args()
    
    # Handle legacy aliases
    if args.deploy:
        args.validate = True
    if args.check_env:
        args.env_check = True
    
    # If no specific action is requested, show help
    if not any([args.tcp, args.rdma, args.pcie, args.stress, args.all,
                args.validate, args.config, args.test_operators, args.env_check]):
        parser.print_help()
        print("\nCommon usage examples:")
        print("  python run_network_tests.py --all               # Run all network tests")
        print("  python run_network_tests.py --quick --tcp       # Quick TCP test")
        print("  python run_network_tests.py --validate          # Validate deployment")
        print("  python run_network_tests.py --config            # Check network config")
        print("  python run_network_tests.py --env-check         # Environment readiness")
        return
    
    print("Network Performance Testing Suite")
    print("="*60)
    print("Unified network testing, validation, and configuration tool")
    print("="*60)
    
    # Run validation/configuration checks if requested
    if args.validate:
        success = run_deployment_validation()
        if not success:
            print("\nDeployment validation failed. Please fix issues before running tests.")
            sys.exit(1)
        print()  # Add spacing before other operations
    
    if args.config:
        check_network_configuration()
        print()  # Add spacing before other operations
    
    if args.test_operators:
        test_operators()
        print()  # Add spacing before other operations
    
    if args.env_check:
        check_environment()
        print()  # Add spacing before other operations
    
    # Run network tests if requested
    test_requested = any([args.tcp, args.rdma, args.pcie, args.stress, args.all])
    if test_requested:
        print("Starting Network Performance Tests")
        print("="*60)
        
        if args.all or args.tcp:
            run_tcp_tests(quick_mode=args.quick, test_duration=args.duration)
        
        if args.all or args.rdma:
            run_rdma_tests()
        
        if args.all or args.pcie:
            run_pcie_tests()
        
        if args.all or args.stress:
            run_network_stress_tests()
        
        print("\n" + "="*60)
        print("All network tests completed!")
        print("="*60)

if __name__ == "__main__":
    main()
