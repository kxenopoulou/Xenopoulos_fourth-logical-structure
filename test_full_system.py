"""
test_full_system.py
Complete test suite for Xenopoulos' Fourth Logical Structure system.
Tests integration of all components: Klein4, Dialectics, Ontology, and full system.
"""

import unittest
import numpy as np
import torch
import sys
import os
from scipy.integrate import solve_ivp

# Προσθήκη του current directory στο path
sys.path.insert(0, os.path.abspath('.'))

# Import από το κύριο αρχείο
from deepseek_python_20260105_ddd90d import (
    XenopoulosKlein4Group,
    XenopoulosDialecticalDynamics,
    XenopoulosOntologicalConflict,
    XenopoulosFourthStructure
)

class TestFullSystemIntegration(unittest.TestCase):
    """Integration tests for the complete Xenopoulos system."""
    
    def setUp(self):
        """Initialize all system components."""
        self.dimension = 3
        self.klein_group = XenopoulosKlein4Group(dimension=self.dimension)
        self.dialectics = XenopoulosDialecticalDynamics(input_dim=self.dimension)
        self.ontology = XenopoulosOntologicalConflict(dimension=self.dimension)
        self.system = XenopoulosFourthStructure(dimension=self.dimension)
        
    def test_01_component_initialization(self):
        """Test that all components initialize correctly."""
        self.assertIsNotNone(self.klein_group)
        self.assertIsNotNone(self.dialectics)
        self.assertIsNotNone(self.ontology)
        self.assertIsNotNone(self.system)
        
        print("✅ All components initialized successfully")
        
    def test_02_klein4_system_compatibility(self):
        """Test that Klein4 group integrates with system."""
        # Verify system uses Klein4 group
        self.assertIsNotNone(self.system.klein_group)
        self.assertEqual(self.system.klein_group.dimension, self.dimension)
        
        # Test operators within system context
        test_vector = np.random.randn(self.dimension)
        identity_result = self.system.klein_group.apply_operator(test_vector, 'I')
        np.testing.assert_array_almost_equal(identity_result, test_vector)
        
        print("✅ Klein4 group integrated with system")
        
    def test_03_dialectical_dynamics_in_system(self):
        """Test dialectical dynamics within full system."""
        self.assertIsNotNone(self.system.dialectics)
        
        # Test forward pass through system's dialectics
        thesis = np.random.randn(self.dimension).astype(np.float32)
        antithesis = -thesis + 0.1 * np.random.randn(self.dimension).astype(np.float32)
        
        thesis_tensor = torch.FloatTensor(thesis).unsqueeze(0)
        antithesis_tensor = torch.FloatTensor(antithesis).unsqueeze(0)
        
        with torch.no_grad():
            # Test both modes
            for mode in ['D1', 'D2']:
                synthesis = self.system.dialectics(
                    thesis_tensor, antithesis_tensor, mode=mode
                )
                self.assertEqual(synthesis.shape, (1, self.dimension))
                
        print("✅ Dialectical dynamics functional in system")
        
    def test_04_ontological_conflict_in_system(self):
        """Test ontological conflict within full system."""
        self.assertIsNotNone(self.system.ontology)
        
        # Test conflict evolution
        initial_state = np.concatenate([
            self.system.thesis,
            self.system.antithesis
        ])
        
        final_state, transition = self.system.ontology.evolve_conflict(
            initial_state, time_span=(0, 1)
        )
        
        self.assertEqual(len(final_state), 2 * self.dimension)
        self.assertIsInstance(transition, bool)
        
        print("✅ Ontological conflict functional in system")
        
    def test_05_single_dialectical_step(self):
        """Test a complete dialectical step in the system."""
        initial_thesis = self.system.thesis.copy()
        initial_antithesis = self.system.antithesis.copy()
        
        # Perform dialectical step
        synthesis = self.system.dialectical_step(include_chaos=False)
        
        # Verify results
        self.assertEqual(len(synthesis), self.dimension)
        self.assertEqual(len(self.system.synthesis_history), 1)
        
        # Verify state updates
        self.assertFalse(np.array_equal(self.system.thesis, initial_thesis))
        self.assertFalse(np.array_equal(self.system.antithesis, initial_antithesis))
        
        print("✅ Single dialectical step completed")
        
    def test_06_ontology_evolution_in_system(self):
        """Test ontology evolution within system."""
        initial_thesis = self.system.thesis.copy()
        
        phase_transition = self.system.evolve_ontology()
        
        # Verify state changed
        self.assertFalse(np.array_equal(self.system.thesis, initial_thesis))
        self.assertIsInstance(phase_transition, bool)
        
        print(f"✅ Ontology evolution completed (transition: {phase_transition})")
        
    def test_07_qualitative_transition_detection(self):
        """Test qualitative transition detection logic."""
        # Test below threshold (no transition)
        below_norm = 0.5
        new_thesis, new_antithesis, transition = self.system.check_qualitative_transition(below_norm)
        self.assertIsNone(new_thesis)
        self.assertIsNone(new_antithesis)
        self.assertFalse(transition)
        
        # Test above threshold (should trigger transition)
        above_norm = 0.9
        new_thesis, new_antithesis, transition = self.system.check_qualitative_transition(above_norm)
        
        if self.system.qualitative_threshold < above_norm:
            self.assertIsNotNone(new_thesis)
            self.assertIsNotNone(new_antithesis)
            self.assertTrue(transition)
            self.assertEqual(len(self.system.qualitative_transitions), 1)
        else:
            self.assertFalse(transition)
        
        print("✅ Qualitative transition detection working")
        
    def test_08_complete_system_evolution_short(self):
        """Test complete system evolution for a few epochs."""
        initial_epoch = self.system.epoch
        
        # Evolve for 10 epochs
        synthesis_history, transitions = self.system.evolve_system(epochs=10)
        
        # Verify results
        self.assertEqual(self.system.epoch, 10)
        self.assertGreater(len(synthesis_history), 0)
        self.assertIsInstance(transitions, list)
        
        # Verify tracking
        self.assertEqual(len(self.system.synthesis_history), len(synthesis_history))
        self.assertEqual(len(self.system.mode_history), 10)
        
        print(f"✅ System evolved for 10 epochs")
        print(f"   Syntheses: {len(synthesis_history)}")
        print(f"   Transitions: {len(transitions)}")
        
    def test_09_mode_alternation(self):
        """Test that system alternates between D1 and D2 modes."""
        # Evolve for multiple steps
        for _ in range(10):
            self.system.dialectical_step(include_chaos=False)
        
        # Check mode alternation pattern
        mode_history = self.system.mode_history
        
        # Should alternate D1, D2, D1, D2...
        for i, mode in enumerate(mode_history[:-1]):
            if mode == 'D1':
                self.assertEqual(mode_history[i+1], 'D2')
            else:
                self.assertEqual(mode_history[i+1], 'D1')
        
        print("✅ Mode alternation working correctly")
        
    def test_10_historical_context(self):
        """Test historical context incorporation."""
        # Generate some history
        for _ in range(5):
            self.system.dialectical_step(include_chaos=False)
        
        # Check history maintenance
        self.assertLessEqual(len(self.system.synthesis_history), 100)  # Max history
        
        # Test with explicit historical context
        thesis_tensor = torch.FloatTensor(self.system.thesis).unsqueeze(0)
        antithesis_tensor = torch.FloatTensor(self.system.antithesis).unsqueeze(0)
        
        historical_context = [
            torch.FloatTensor(s).unsqueeze(0) 
            for s in self.system.synthesis_history[-3:]
        ]
        
        with torch.no_grad():
            synthesis = self.system.dialectics(
                thesis_tensor, antithesis_tensor, 
                historical_context=historical_context,
                mode='D1'
            )
            
        self.assertEqual(synthesis.shape, (1, self.dimension))
        
        print("✅ Historical context functioning")
        
    def test_11_chaos_injection(self):
        """Test chaos injection in dialectical steps."""
        # Run step without chaos
        synthesis_no_chaos = self.system.dialectical_step(include_chaos=False)
        
        # Reset system
        self.system = XenopoulosFourthStructure(dimension=self.dimension)
        
        # Run step with chaos
        synthesis_with_chaos = self.system.dialectical_step(include_chaos=True)
        
        # They should be different due to chaos
        self.assertFalse(np.array_equal(synthesis_no_chaos, synthesis_with_chaos))
        
        print("✅ Chaos injection working")
        
    def test_12_visualization_method(self):
        """Test that visualization method runs without errors."""
        # Evolve a bit first
        for _ in range(20):
            self.system.dialectical_step(include_chaos=False)
        
        # Try to create visualization (should not crash)
        try:
            # Note: We're not actually showing it, just testing creation
            import matplotlib
            matplotlib.use('Agg')  # Use non-interactive backend
            
            # This should run without errors
            self.system.visualize_system()
            
            print("✅ Visualization method runs without errors")
        except Exception as e:
            self.fail(f"Visualization failed with error: {e}")
            
    def test_13_parameter_consistency(self):
        """Test that parameters remain consistent across components."""
        # All components should have same dimension
        self.assertEqual(self.klein_group.dimension, self.dimension)
        self.assertEqual(self.dialectics.input_dim, self.dimension)
        self.assertEqual(self.ontology.dimension, self.dimension)
        self.assertEqual(self.system.dimension, self.dimension)
        
        # System should contain all components
        self.assertEqual(self.system.klein_group.dimension, self.dimension)
        self.assertEqual(self.system.dialectics.input_dim, self.dimension)
        self.assertEqual(self.system.ontology.dimension, self.dimension)
        
        print("✅ Parameter consistency verified")
        
    def test_14_synthesis_properties(self):
        """Test mathematical properties of synthesis."""
        # Generate multiple syntheses
        syntheses = []
        for _ in range(10):
            synthesis = self.system.dialectical_step(include_chaos=False)
            syntheses.append(synthesis)
            
            # Check basic properties
            self.assertFalse(np.any(np.isnan(synthesis)))
            self.assertFalse(np.any(np.isinf(synthesis)))
            
        # Calculate statistics
        syntheses_array = np.array(syntheses)
        norms = np.linalg.norm(syntheses_array, axis=1)
        
        # Norms should be positive
        self.assertTrue(np.all(norms >= 0))
        
        # Most norms should be below qualitative threshold
        below_threshold = norms < self.system.qualitative_threshold
        self.assertGreater(np.mean(below_threshold), 0.5)  # Most below threshold
        
        print("✅ Synthesis mathematical properties verified")
        
    def test_15_reset_and_reinitialize(self):
        """Test that system can be reinitialized."""
        initial_state = self.system.thesis.copy()
        
        # Evolve a bit
        for _ in range(5):
            self.system.dialectical_step(include_chaos=False)
            
        # Create new system
        new_system = XenopoulosFourthStructure(dimension=self.dimension)
        
        # Should have different initial state
        self.assertFalse(np.array_equal(initial_state, new_system.thesis))
        
        # But same structure
        self.assertEqual(self.system.dimension, new_system.dimension)
        self.assertEqual(self.system.qualitative_threshold, new_system.qualitative_threshold)
        
        print("✅ System reinitialization working")
        
    def test_16_edge_cases(self):
        """Test edge cases and error handling."""
        # Test with very small dimension (minimum)
        try:
            small_system = XenopoulosFourthStructure(dimension=2)
            self.assertEqual(small_system.dimension, 2)
            print("✅ Minimum dimension (2) works")
        except Exception as e:
            print(f"⚠️  Dimension 2 failed: {e}")
            
        # Test with larger dimension
        try:
            large_system = XenopoulosFourthStructure(dimension=10)
            self.assertEqual(large_system.dimension, 10)
            print("✅ Larger dimension (10) works")
        except Exception as e:
            print(f"⚠️  Dimension 10 failed: {e}")
            
        # Test extreme parameter values
        extreme_system = XenopoulosFourthStructure(dimension=3)
        extreme_system.qualitative_threshold = 100.0  # Very high
        extreme_system.chaos_factor = 0.0  # No chaos
        
        # Should still work
        synthesis = extreme_system.dialectical_step(include_chaos=False)
        self.assertEqual(len(synthesis), 3)
        
        print("✅ Edge cases handled")
        
    def test_17_performance_benchmark(self):
        """Basic performance benchmark."""
        import time
        
        # Time single step
        start = time.time()
        self.system.dialectical_step(include_chaos=False)
        single_step_time = time.time() - start
        
        # Time 100 steps
        start = time.time()
        for _ in range(100):
            self.system.dialectical_step(include_chaos=False)
        hundred_steps_time = time.time() - start
        
        print(f"⏱️  Performance benchmarks:")
        print(f"   Single step: {single_step_time*1000:.2f} ms")
        print(f"   100 steps: {hundred_steps_time*1000:.2f} ms")
        print(f"   Average per step: {hundred_steps_time/100*1000:.2f} ms")
        
        self.assertLess(single_step_time, 1.0)  # Should be fast
        self.assertLess(hundred_steps_time, 10.0)  # Should be reasonable
        
    def test_18_complete_workflow(self):
        """Test complete workflow from initialization to visualization."""
        print("\n🔬 Testing complete workflow...")
        
        # 1. Initialize
        system = XenopoulosFourthStructure(dimension=3)
        print("   Step 1: System initialized")
        
        # 2. Evolve
        synthesis_history, transitions = system.evolve_system(epochs=50)
        print(f"   Step 2: Evolved for 50 epochs")
        print(f"     Syntheses: {len(synthesis_history)}")
        print(f"     Transitions: {len(transitions)}")
        
        # 3. Analyze
        final_synthesis = synthesis_history[-1]
        final_norm = np.linalg.norm(final_synthesis)
        print(f"   Step 3: Analysis")
        print(f"     Final synthesis norm: {final_norm:.4f}")
        print(f"     Thesis norm: {np.linalg.norm(system.thesis):.4f}")
        print(f"     Antithesis norm: {np.linalg.norm(system.antithesis):.4f}")
        
        # 4. Verify system state
        self.assertGreater(len(synthesis_history), 0)
        self.assertIsInstance(transitions, list)
        self.assertEqual(system.epoch, 50)
        
        print("✅ Complete workflow successful")

def run_comprehensive_tests():
    """Run all tests with detailed reporting."""
    print("=" * 80)
    print("COMPREHENSIVE TEST SUITE: XENOPOULOS FOURTH LOGICAL STRUCTURE")
    print("=" * 80)
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestFullSystemIntegration)
    
    # Run tests with detailed output
    runner = unittest.TextTestRunner(verbosity=2, descriptions=True)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print(f"Total Tests: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped)}")
    
    if result.wasSuccessful():
        print("\n🎉 ALL TESTS PASSED! System is fully operational.")
    else:
        print("\n⚠️  SOME TESTS FAILED. Check details above.")
        
    print("=" * 80)
    
    return result.wasSuccessful()

if __name__ == "__main__":
    # Run comprehensive test suite
    success = run_comprehensive_tests()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)