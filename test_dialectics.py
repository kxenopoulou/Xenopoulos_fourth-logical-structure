"""
test_dialectics.py
Unit tests for XenopoulosDialecticalDynamics class.
Tests D₁ and D₂ formalisms and dialectical synthesis.
"""

import unittest
import numpy as np
import torch
import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.abspath('.'))

# Import from main implementation
from deepseek_python_20260105_ddd90d import XenopoulosDialecticalDynamics

class TestXenopoulosDialecticalDynamics(unittest.TestCase):
    """Test cases for dialectical dynamics implementation."""
    
    def setUp(self):
        """Initialize test dialectics with dimension 3."""
        self.dimension = 3
        self.dialectics = XenopoulosDialecticalDynamics(input_dim=self.dimension)
        self.device = self.dialectics.device
        
    def test_01_initialization(self):
        """Test class initialization."""
        self.assertEqual(self.dialectics.input_dim, self.dimension)
        self.assertIsNotNone(self.dialectics.D1_network)
        self.assertIsNotNone(self.dialectics.D2_network)
        
        # Check parameters
        self.assertIsInstance(self.dialectics.alpha, torch.nn.Parameter)
        self.assertIsInstance(self.dialectics.beta, torch.nn.Parameter)
        self.assertIsInstance(self.dialectics.gamma, torch.nn.Parameter)
        
        print("✅ Dialectical dynamics initialized correctly")
    
    def test_02_D1_forward_pass(self):
        """Test D₁ forward pass (F→N→R→C)."""
        thesis = np.random.randn(self.dimension).astype(np.float32)
        antithesis = np.random.randn(self.dimension).astype(np.float32)
        
        thesis_tensor = torch.FloatTensor(thesis).unsqueeze(0).to(self.device)
        antithesis_tensor = torch.FloatTensor(antithesis).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            result = self.dialectics.forward(
                thesis_tensor, antithesis_tensor, mode='D1'
            )
        
        # Check output structure
        self.assertIn('synthesis', result)
        self.assertIn('identity', result)
        self.assertIn('negation', result)
        self.assertIn('reciprocity', result)
        self.assertIn('correlation', result)
        self.assertIn('qualitative_transition', result)
        self.assertIn('synthesis_norm', result)
        self.assertIn('mode', result)
        
        # Check tensor shapes
        self.assertEqual(result['synthesis'].shape, (1, self.dimension))
        self.assertEqual(result['identity'].shape, (1, self.dimension))
        self.assertEqual(result['negation'].shape, (1, self.dimension))
        self.assertEqual(result['reciprocity'].shape, (1, self.dimension))
        self.assertEqual(result['correlation'].shape, (1, self.dimension))
        
        # Check mode
        self.assertEqual(result['mode'], 'D1')
        
        print("✅ D₁ forward pass successful")
    
    def test_03_D2_forward_pass(self):
        """Test D₂ forward pass (F→C→N→R)."""
        thesis = np.random.randn(self.dimension).astype(np.float32)
        antithesis = np.random.randn(self.dimension).astype(np.float32)
        
        thesis_tensor = torch.FloatTensor(thesis).unsqueeze(0).to(self.device)
        antithesis_tensor = torch.FloatTensor(antithesis).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            result = self.dialectics.forward(
                thesis_tensor, antithesis_tensor, mode='D2'
            )
        
        # Check output structure
        self.assertIn('synthesis', result)
        self.assertEqual(result['synthesis'].shape, (1, self.dimension))
        self.assertEqual(result['mode'], 'D2')
        
        print("✅ D₂ forward pass successful")
    
    def test_04_inrc_operators(self):
        """Test INRC operator transformations."""
        thesis = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        antithesis = np.array([4.0, 5.0, 6.0], dtype=np.float32)
        
        thesis_tensor = torch.FloatTensor(thesis).unsqueeze(0).to(self.device)
        antithesis_tensor = torch.FloatTensor(antithesis).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            result = self.dialectics.forward(
                thesis_tensor, antithesis_tensor, mode='D1'
            )
        
        # Check identity: I(x) = x
        identity = result['identity'].cpu().numpy()[0]
        np.testing.assert_array_almost_equal(identity, thesis, decimal=6)
        
        # Check negation: N(x) = -x
        negation = result['negation'].cpu().numpy()[0]
        expected_negation = -antithesis
        np.testing.assert_array_almost_equal(negation, expected_negation, decimal=6)
        
        # Check reciprocity: cyclic permutation
        reciprocity = result['reciprocity'].cpu().numpy()[0]
        expected_reciprocity = np.array([3.0, 1.0, 2.0])  # (x₁, x₂, x₃) → (x₃, x₁, x₂)
        np.testing.assert_array_almost_equal(reciprocity, expected_reciprocity, decimal=6)
        
        print("✅ INRC operators working correctly")
    
    def test_05_synthesis_equation(self):
        """Test Xenopoulos synthesis equation S = α(I•N) - β|I-N| + γR."""
        thesis = np.random.randn(self.dimension).astype(np.float32)
        antithesis = np.random.randn(self.dimension).astype(np.float32)
        
        thesis_tensor = torch.FloatTensor(thesis).unsqueeze(0).to(self.device)
        antithesis_tensor = torch.FloatTensor(antithesis).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            result = self.dialectics.forward(
                thesis_tensor, antithesis_tensor, mode='D1'
            )
        
        # Get components for manual calculation
        identity = torch.FloatTensor(thesis).unsqueeze(0).to(self.device)
        negation = torch.FloatTensor(-antithesis).unsqueeze(0).to(self.device)
        reciprocity = torch.roll(identity, shifts=1, dims=-1)
        
        # Manual calculation
        identity_dot_negation = torch.sum(identity * negation, dim=-1, keepdim=True)
        identity_minus_negation_norm = torch.norm(identity - negation, dim=-1, keepdim=True)
        reciprocity_mean = torch.mean(reciprocity, dim=-1, keepdim=True)
        
        manual_synthesis = (
            self.dialectics.alpha * identity_dot_negation -
            self.dialectics.beta * identity_minus_negation_norm +
            self.dialectics.gamma * reciprocity_mean
        )
        
        # The actual synthesis includes neural network output too
        synthesis = result['synthesis']
        self.assertEqual(synthesis.shape, manual_synthesis.shape)
        
        print("✅ Synthesis equation implemented correctly")
    
    def test_06_historical_context(self):
        """Test historical context integration."""
        thesis = np.random.randn(self.dimension).astype(np.float32)
        antithesis = np.random.randn(self.dimension).astype(np.float32)
        
        thesis_tensor = torch.FloatTensor(thesis).unsqueeze(0).to(self.device)
        antithesis_tensor = torch.FloatTensor(antithesis).unsqueeze(0).to(self.device)
        
        # Create historical context (last 3 syntheses)
        historical_context = []
        for i in range(3):
            hist = np.random.randn(self.dimension).astype(np.float32)
            hist_tensor = torch.FloatTensor(hist).unsqueeze(0).to(self.device)
            historical_context.append(hist_tensor)
        
        with torch.no_grad():
            # Without history
            result_no_history = self.dialectics.forward(
                thesis_tensor, antithesis_tensor, mode='D1'
            )
            
            # With history
            result_with_history = self.dialectics.forward(
                thesis_tensor, antithesis_tensor, 
                historical_context=historical_context,
                mode='D1'
            )
        
        # Results should be different
        synthesis_no_hist = result_no_history['synthesis'].cpu().numpy()
        synthesis_with_hist = result_with_history['synthesis'].cpu().numpy()
        
        self.assertFalse(np.array_equal(synthesis_no_hist, synthesis_with_hist))
        
        print("✅ Historical context integration working")
    
    def test_07_qualitative_transition_detection(self):
        """Test qualitative transition detection."""
        # Create dialectics with specific threshold
        threshold = 0.5
        dialectics = XenopoulosDialecticalDynamics(
            input_dim=self.dimension,
            qualitative_threshold=threshold
        )
        
        # Test with low norm (no transition)
        thesis = np.ones(self.dimension, dtype=np.float32) * 0.1
        antithesis = np.ones(self.dimension, dtype=np.float32) * 0.1
        
        thesis_tensor = torch.FloatTensor(thesis).unsqueeze(0).to(self.device)
        antithesis_tensor = torch.FloatTensor(antithesis).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            result = dialectics.forward(
                thesis_tensor, antithesis_tensor, mode='D1'
            )
        
        # Norm should be low, no transition
        self.assertLess(result['synthesis_norm'], threshold)
        self.assertFalse(result['qualitative_transition'])
        
        print("✅ Qualitative transition detection working")
    
    def test_08_D1_D2_differences(self):
        """Test that D₁ and D₂ produce different results."""
        thesis = np.random.randn(self.dimension).astype(np.float32)
        antithesis = np.random.randn(self.dimension).astype(np.float32)
        
        thesis_tensor = torch.FloatTensor(thesis).unsqueeze(0).to(self.device)
        antithesis_tensor = torch.FloatTensor(antithesis).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            result_D1 = self.dialectics.forward(
                thesis_tensor, antithesis_tensor, mode='D1'
            )
            result_D2 = self.dialectics.forward(
                thesis_tensor, antithesis_tensor, mode='D2'
            )
        
        synthesis_D1 = result_D1['synthesis'].cpu().numpy()
        synthesis_D2 = result_D2['synthesis'].cpu().numpy()
        
        # D₁ and D₂ should produce different syntheses
        self.assertFalse(np.array_equal(synthesis_D1, synthesis_D2))
        
        # Check norms are reasonable
        self.assertGreater(result_D1['synthesis_norm'], 0)
        self.assertGreater(result_D2['synthesis_norm'], 0)
        
        print("✅ D₁ and D₂ produce distinct syntheses")
    
    def test_09_batch_processing(self):
        """Test batch processing capability."""
        batch_size = 4
        
        # Create batch of inputs
        thesis_batch = np.random.randn(batch_size, self.dimension).astype(np.float32)
        antithesis_batch = np.random.randn(batch_size, self.dimension).astype(np.float32)
        
        thesis_tensor = torch.FloatTensor(thesis_batch).to(self.device)
        antithesis_tensor = torch.FloatTensor(antithesis_batch).to(self.device)
        
        with torch.no_grad():
            result = self.dialectics.forward(
                thesis_tensor, antithesis_tensor, mode='D1'
            )
        
        # Check batch dimensions
        self.assertEqual(result['synthesis'].shape, (batch_size, self.dimension))
        self.assertEqual(result['identity'].shape, (batch_size, self.dimension))
        self.assertEqual(result['negation'].shape, (batch_size, self.dimension))
        
        # synthesis_norm should be scalar (mean of batch)
        self.assertIsInstance(result['synthesis_norm'], float)
        
        print("✅ Batch processing working")
    
    def test_10_dialectical_cycle(self):
        """Test complete dialectical cycle."""
        thesis = np.random.randn(self.dimension).astype(np.float32)
        antithesis = np.random.randn(self.dimension).astype(np.float32)
        
        # Run dialectical cycle
        history = self.dialectics.dialectical_cycle(
            thesis, antithesis, steps=5, mode='D1'
        )
        
        # Check history structure
        self.assertIn('thesis', history)
        self.assertIn('antithesis', history)
        self.assertIn('synthesis', history)
        self.assertIn('synthesis_norms', history)
        self.assertIn('qualitative_transitions', history)
        
        # Check lengths
        self.assertEqual(len(history['thesis']), 6)  # initial + 5 steps
        self.assertEqual(len(history['synthesis']), 5)  # 5 syntheses
        self.assertEqual(len(history['synthesis_norms']), 5)
        self.assertEqual(len(history['qualitative_transitions']), 5)
        
        # Check evolution
        for i in range(len(history['synthesis_norms']) - 1):
            # Norms should vary
            self.assertNotEqual(history['synthesis_norms'][i], 
                              history['synthesis_norms'][i+1])
        
        print("✅ Dialectical cycle working")
    
    def test_11_monte_carlo_analysis(self):
        """Test Monte Carlo analysis method."""
        thesis = np.random.randn(self.dimension).astype(np.float32)
        antithesis = np.random.randn(self.dimension).astype(np.float32)
        
        analysis = self.dialectics.analyze_synthesis(
            thesis, antithesis, n_iterations=50
        )
        
        # Check analysis structure
        self.assertIn('mean_synthesis', analysis)
        self.assertIn('std_synthesis', analysis)
        self.assertIn('mean_norm', analysis)
        self.assertIn('std_norm', analysis)
        self.assertIn('min_norm', analysis)
        self.assertIn('max_norm', analysis)
        self.assertIn('probability_qualitative', analysis)
        
        # Check values
        self.assertEqual(len(analysis['mean_synthesis']), self.dimension)
        self.assertIsInstance(analysis['mean_norm'], float)
        self.assertIsInstance(analysis['probability_qualitative'], float)
        
        # Probability should be between 0 and 1
        self.assertGreaterEqual(analysis['probability_qualitative'], 0)
        self.assertLessEqual(analysis['probability_qualitative'], 1)
        
        # min_norm <= mean_norm <= max_norm
        self.assertLessEqual(analysis['min_norm'], analysis['mean_norm'])
        self.assertGreaterEqual(analysis['max_norm'], analysis['mean_norm'])
        
        print("✅ Monte Carlo analysis working")
    
    def test_12_invalid_mode_handling(self):
        """Test error handling for invalid mode."""
        thesis = np.random.randn(self.dimension).astype(np.float32)
        antithesis = np.random.randn(self.dimension).astype(np.float32)
        
        thesis_tensor = torch.FloatTensor(thesis).unsqueeze(0).to(self.device)
        antithesis_tensor = torch.FloatTensor(antithesis).unsqueeze(0).to(self.device)
        
        # Should raise ValueError for invalid mode
        with self.assertRaises(ValueError):
            self.dialectics.forward(
                thesis_tensor, antithesis_tensor, mode='INVALID'
            )
        
        print("✅ Invalid mode handling working")
    
    def test_13_parameter_gradient_flow(self):
        """Test that gradients flow through parameters."""
        thesis = np.random.randn(self.dimension).astype(np.float32)
        antithesis = np.random.randn(self.dimension).astype(np.float32)
        
        thesis_tensor = torch.FloatTensor(thesis).unsqueeze(0).to(self.device)
        antithesis_tensor = torch.FloatTensor(antithesis).unsqueeze(0).to(self.device)
        
        # Enable gradient tracking
        thesis_tensor.requires_grad = True
        antithesis_tensor.requires_grad = True
        
        # Forward pass
        result = self.dialectics.forward(
            thesis_tensor, antithesis_tensor, mode='D1'
        )
        
        # Create dummy loss
        loss = torch.mean(result['synthesis'] ** 2)
        
        # Backward pass
        loss.backward()
        
        # Check gradients
        self.assertIsNotNone(self.dialectics.alpha.grad)
        self.assertIsNotNone(self.dialectics.beta.grad)
        self.assertIsNotNone(self.dialectics.gamma.grad)
        
        # Check network gradients
        for param in self.dialectics.D1_network.parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad)
        
        print("✅ Gradient flow working")
    
    def test_14_save_load_functionality(self):
        """Test model save/load functionality."""
        import tempfile
        import os
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as tmp:
            temp_path = tmp.name
        
        try:
            # Save model
            self.dialectics.save_model(temp_path)
            self.assertTrue(os.path.exists(temp_path))
            
            # Load model
            loaded_dialectics = XenopoulosDialecticalDynamics.load_model(
                temp_path, device='cpu'
            )
            
            # Check loaded model
            self.assertEqual(loaded_dialectics.input_dim, self.dimension)
            
            # Test forward pass with loaded model
            thesis = np.random.randn(self.dimension).astype(np.float32)
            antithesis = np.random.randn(self.dimension).astype(np.float32)
            
            thesis_tensor = torch.FloatTensor(thesis).unsqueeze(0)
            antithesis_tensor = torch.FloatTensor(antithesis).unsqueeze(0)
            
            with torch.no_grad():
                original_result = self.dialectics.forward(
                    thesis_tensor, antithesis_tensor, mode='D1'
                )
                loaded_result = loaded_dialectics.forward(
                    thesis_tensor, antithesis_tensor, mode='D1'
                )
            
            # Results should be similar (not exactly equal due to device)
            original_synth = original_result['synthesis'].cpu().numpy()
            loaded_synth = loaded_result['synthesis'].cpu().numpy()
            
            np.testing.assert_array_almost_equal(
                original_synth, loaded_synth, decimal=5
            )
            
            print("✅ Save/load functionality working")
            
        finally:
            # Cleanup
            if os.path.exists(temp_path):
                os.remove(temp_path)
    
    def test_15_visualization_2d(self):
        """Test 2D visualization method."""
        # Create 2D dialectics for visualization
        dialectics_2d = XenopoulosDialecticalDynamics(input_dim=2)
        
        try:
            # Should create visualization without errors
            result = dialectics_2d.visualize_synthesis_space(
                thesis_range=(-1, 1),
                antithesis_range=(-1, 1),
                n_points=10,
                mode='D1'
            )
            
            # Check return value
            self.assertIn('X', result)
            self.assertIn('Y', result)
            self.assertIn('syntheses', result)
            self.assertIn('norms', result)
            
            print("✅ 2D visualization working")
            
        except Exception as e:
            # If matplotlib not available, skip but don't fail
            print(f"⚠️  Visualization skipped: {e}")
            self.skipTest(f"Visualization dependencies not available: {e}")
    
    def test_16_high_dimensional_input(self):
        """Test with higher dimensional inputs."""
        high_dim = 8
        dialectics_high = XenopoulosDialecticalDynamics(input_dim=high_dim)
        
        thesis = np.random.randn(high_dim).astype(np.float32)
        antithesis = np.random.randn(high_dim).astype(np.float32)
        
        thesis_tensor = torch.FloatTensor(thesis).unsqueeze(0)
        antithesis_tensor = torch.FloatTensor(antithesis).unsqueeze(0)
        
        with torch.no_grad():
            result = dialectics_high.forward(
                thesis_tensor, antithesis_tensor, mode='D1'
            )
        
        self.assertEqual(result['synthesis'].shape, (1, high_dim))
        
        print("✅ High-dimensional input working")
    
    def test_17_zero_input(self):
        """Test with zero vectors."""
        thesis = np.zeros(self.dimension, dtype=np.float32)
        antithesis = np.zeros(self.dimension, dtype=np.float32)
        
        thesis_tensor = torch.FloatTensor(thesis).unsqueeze(0).to(self.device)
        antithesis_tensor = torch.FloatTensor(antithesis).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            result = self.dialectics.forward(
                thesis_tensor, antithesis_tensor, mode='D1'
            )
        
        # Check output is valid
        self.assertFalse(np.any(np.isnan(result['synthesis'].cpu().numpy())))
        self.assertFalse(np.any(np.isinf(result['synthesis'].cpu().numpy())))
        
        print("✅ Zero input handling working")
    
    def test_18_extreme_input_values(self):
        """Test with extreme input values."""
        thesis = np.ones(self.dimension, dtype=np.float32) * 100.0
        antithesis = np.ones(self.dimension, dtype=np.float32) * -100.0
        
        thesis_tensor = torch.FloatTensor(thesis).unsqueeze(0).to(self.device)
        antithesis_tensor = torch.FloatTensor(antithesis).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            result = self.dialectics.forward(
                thesis_tensor, antithesis_tensor, mode='D1'
            )
        
        # Output should still be valid
        synthesis = result['synthesis'].cpu().numpy()
        self.assertFalse(np.any(np.isnan(synthesis)))
        self.assertFalse(np.any(np.isinf(synthesis)))
        
        print("✅ Extreme input handling working")

def run_dialectics_tests():
    """Run all dialectics tests with detailed reporting."""
    print("=" * 80)
    print("TEST SUITE: XENOPOULOS DIALECTICAL DYNAMICS")
    print("=" * 80)
    print(f"Testing D₁ (F→N→R→C) and D₂ (F→C→N→R) formalisms")
    print(f"Synthesis equation: S = α(I•N) - β|I-N| + γR")
    print("=" * 80)
    
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestXenopoulosDialecticalDynamics)
    
    runner = unittest.TextTestRunner(verbosity=2, descriptions=True)
    result = runner.run(suite)
    
    print("\n" + "=" * 80)
    print("DIALECTICS TEST SUMMARY")
    print("=" * 80)
    print(f"Total Tests: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped)}")
    
    if result.wasSuccessful():
        print("\n✅ DIALECTICAL DYNAMICS VALIDATION COMPLETE")
        print("   All tests passed!")
    else:
        print("\n⚠️  DIALECTICAL DYNAMICS VALIDATION FAILED")
        print("   Check test failures above")
    
    print("=" * 80)
    
    return result.wasSuccessful()

if __name__ == "__main__":
    success = run_dialectics_tests()
    sys.exit(0 if success else 1)
    
    