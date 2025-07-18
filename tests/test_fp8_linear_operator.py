#!/usr/bin/env python3
"""
Test FP8 Linear Operator
Test basic functionality of the FP8 linear operator
"""

import sys
import os
import unittest
import torch

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from operators.fp8_linear_operator import FP8LinearOperator

class TestFP8LinearOperator(unittest.TestCase):
    """Test cases for FP8 Linear operator"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.operator = FP8LinearOperator()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def test_operator_initialization(self):
        """Test that the operator initializes correctly"""
        self.assertIsNotNone(self.operator)
        self.assertTrue(len(self.operator.available_backends) > 0)
        self.assertIn("pytorch_bf16", self.operator.available_backends)
        
    def test_get_test_cases(self):
        """Test that test cases are generated correctly"""
        test_cases = self.operator.get_test_cases()
        self.assertTrue(len(test_cases) > 0)
        
        for test_case in test_cases:
            self.assertIsNotNone(test_case.name)
            self.assertIsNotNone(test_case.input_shapes)
            self.assertIsNotNone(test_case.input_dtypes)
            self.assertIsNotNone(test_case.additional_params)
            self.assertIn("out_features", test_case.additional_params)
            
    def test_generate_inputs(self):
        """Test input generation"""
        test_cases = self.operator.get_test_cases()
        test_case = test_cases[0]  # Use first test case
        
        inputs = self.operator.generate_inputs(test_case)
        self.assertTrue(len(inputs) > 0)
        
        for input_tensor in inputs:
            self.assertIsInstance(input_tensor, torch.Tensor)
            self.assertEqual(input_tensor.device.type, self.device.type)
            
    def test_calculate_flops(self):
        """Test FLOP calculation"""
        test_cases = self.operator.get_test_cases()
        test_case = test_cases[0]  # Use first test case
        
        flops = self.operator.calculate_flops(test_case)
        self.assertGreater(flops, 0)
        
        # Verify FLOP calculation
        input_shape = test_case.input_shapes[0]
        out_features = test_case.additional_params["out_features"]
        expected_flops = 2 * input_shape[0] * input_shape[1] * out_features
        self.assertEqual(flops, expected_flops)
        
    def test_pytorch_bf16_implementation(self):
        """Test PyTorch BF16 baseline implementation"""
        test_cases = self.operator.get_test_cases()
        test_case = test_cases[0]  # Use first test case
        
        inputs = self.operator.generate_inputs(test_case)
        
        try:
            result = self.operator._pytorch_bf16_impl(inputs, test_case.additional_params)
            self.assertIsInstance(result, torch.Tensor)
            
            # Check output shape
            expected_out_features = test_case.additional_params["out_features"]
            expected_shape = (inputs[0].shape[0], expected_out_features)
            self.assertEqual(result.shape, expected_shape)
            
        except Exception as e:
            self.fail(f"PyTorch BF16 implementation failed: {e}")
            
    def test_available_implementations(self):
        """Test getting available implementations"""
        implementations = self.operator.get_available_implementations()
        self.assertIsInstance(implementations, list)
        self.assertIn("pytorch_bf16", implementations)
        
    def test_reference_implementation(self):
        """Test getting reference implementation"""
        ref_impl = self.operator.get_reference_implementation()
        self.assertEqual(ref_impl, "pytorch_bf16")

if __name__ == '__main__':
    unittest.main()
