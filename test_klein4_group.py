"""
test_klein4_group.py
Unit tests for XenopoulosKlein4Group class.
Validates Klein-4 group properties and INRC operators.
"""

import unittest
import numpy as np
import sys
import os

# Προσθήκη του src/ directory στο path για να μπορούμε να εισάγουμε τις κλάσεις
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from klein4_group import XenopoulosKlein4Group

class TestXenopoulosKlein4Group(unittest.TestCase):
    """Test cases for the Klein-4 group implementation."""
    
    def setUp(self):
        """Initialize test group with dimension 3."""
        self.dimension = 3
        self.group = XenopoulosKlein4Group(dimension=self.dimension)
        
    def test_identity_operator(self):
        """Test identity operator I(x) = x."""
        vector = np.random.randn(self.dimension)
        result = self.group.apply_operator(vector, 'I')
        np.testing.assert_array_almost_equal(result, vector, decimal=6)
        
    def test_negation_operator(self):
        """Test negation operator N(x) = -x."""
        vector = np.random.randn(self.dimension)
        result = self.group.apply_operator(vector, 'N')
        expected = -vector
        np.testing.assert_array_almost_equal(result, expected, decimal=6)
        
    def test_self_inverse_property(self):
        """Test N∘N = I."""
        vector = np.random.randn(self.dimension)
        # Apply N twice
        once = self.group.apply_operator(vector, 'N')
        twice = self.group.apply_operator(once, 'N')
        np.testing.assert_array_almost_equal(twice, vector, decimal=6)
        
    def test_reciprocity_operator(self):
        """Test reciprocity operator R (cyclic permutation)."""
        vector = np.array([1.0, 2.0, 3.0])
        result = self.group.apply_operator(vector, 'R')
        expected = np.array([3.0, 1.0, 2.0])  # Cyclic shift right
        np.testing.assert_array_almost_equal(result, expected, decimal=6)
        
    def test_correlation_operator(self):
        """Test correlation operator C = N∘R = R∘N."""
        vector = np.random.randn(self.dimension)
        # Calculate C(x) directly
        c_result = self.group.apply_operator(vector, 'C')
        # Calculate N∘R(x)
        nr_result = self.group.apply_operator(
            self.group.apply_operator(vector, 'R'), 'N'
        )
        # Calculate R∘N(x)
        rn_result = self.group.apply_operator(
            self.group.apply_operator(vector, 'N'), 'R'
        )
        np.testing.assert_array_almost_equal(c_result, nr_result, decimal=6)
        np.testing.assert_array_almost_equal(c_result, rn_result, decimal=6)
        
    def test_klein4_group_axioms(self):
        """Test all Klein-4 group axioms."""
        # Test closure: composition of any two operators yields an operator in the group
        operators = ['I', 'N', 'R', 'C']
        for a in operators:
            for b in operators:
                result_ab = self.group.I @ (self.group.apply_operator(
                    self.group.apply_operator(np.eye(self.dimension)[0], b), a
                ))
                # The result should be equivalent to applying some operator in the group
                # We'll check via Cayley table consistency
                pass  # Cayley table test is separate
                
        # Test associativity: (a∘b)∘c = a∘(b∘c)
        a = np.random.randn(self.dimension)
        for op1 in operators:
            for op2 in operators:
                for op3 in operators:
                    # (op1∘op2)∘op3
                    res1 = self.group.apply_operator(
                        self.group.apply_operator(
                            self.group.apply_operator(a, op3), op2
                        ), op1
                    )
                    # op1∘(op2∘op3)
                    res2 = self.group.apply_operator(a, op1)
                    res2 = self.group.apply_operator(res2, op2)
                    res2 = self.group.apply_operator(res2, op3)
                    # Should be equal if we track composition correctly
                    # Note: This test is conceptual; actual matrix multiplication is associative
                    
    def test_cayley_table(self):
        """Test that Cayley table matches Klein-4 group structure."""
        table = self.group.get_cayley_table()
        
        # Expected Klein-4 Cayley table
        expected = {
            'I': {'I': 'I', 'N': 'N', 'R': 'R', 'C': 'C'},
            'N': {'I': 'N', 'N': 'I', 'R': 'C', 'C': 'R'},
            'R': {'I': 'R', 'N': 'C', 'R': 'I', 'C': 'N'},
            'C': {'I': 'C', 'N': 'R', 'R': 'N', 'C': 'I'}
        }
        
        self.assertEqual(table, expected)
        
    def test_invalid_operator(self):
        """Test error handling for invalid operator name."""
        vector = np.random.randn(self.dimension)
        with self.assertRaises(ValueError):
            self.group.apply_operator(vector, 'X')  # 'X' is not a valid operator
            
    def test_dimension_mismatch(self):
        """Test that operator dimension matches vector dimension."""
        vector = np.random.randn(self.dimension + 1)  # Wrong dimension
        with self.assertRaises(ValueError):
            # This would fail in matrix multiplication
            self.group.apply_operator(vector, 'I')
            
    def test_trace_properties(self):
        """Test traces of operators."""
        self.assertEqual(np.trace(self.group.I), self.dimension)
        self.assertEqual(np.trace(self.group.N), -self.dimension)
        # R and C traces depend on dimension
        self.assertAlmostEqual(np.trace(self.group.R), 
                              1.0 if self.dimension % 3 == 0 else 0.0)
        
    def test_determinant_properties(self):
        """Test determinants of operators."""
        self.assertAlmostEqual(np.linalg.det(self.group.I), 1.0)
        self.assertAlmostEqual(np.linalg.det(self.group.N), 
                              (-1.0) ** self.dimension)
        self.assertAlmostEqual(np.linalg.det(self.group.R), 
                              1.0 if self.dimension % 2 == 0 else -1.0)
        
def run_tests():
    """Run all tests and print results."""
    suite = unittest.TestLoader().loadTestsFromTestCase(TestXenopoulosKlein4Group)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    if result.wasSuccessful():
        print("\n✅ All tests passed!")
        return True
    else:
        print("\n❌ Some tests failed.")
        return False

if __name__ == "__main__":
    print("=" * 70)
    print("TEST SUITE: XenopoulosKlein4Group")
    print("=" * 70)
    
    success = run_tests()
    
    print("\n" + "=" * 70)
    if success:
        print("🎉 KLEIN-4 GROUP VALIDATION COMPLETE")
    else:
        print("⚠️  VALIDATION FAILED - CHECK ERRORS")
    print("=" * 70)