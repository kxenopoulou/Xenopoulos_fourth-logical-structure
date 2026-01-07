"""
Klein-4 Group Implementation for Xenopoulos' INRC Operators
Core mathematical framework for Piaget's operators: Identity (I), Negation (N), Reciprocity (R), Correlation (C)

Author: Epameinondas Xenopoulos (Theoretical Framework)
Implementation: [Your Name]
Date: 2024

Based on: Xenopoulos, E. (2024). Epistemology of Logic: Logic-Dialectic or Theory of Knowledge (2nd ed.)
DOI: https://doi.org/10.5281/zenodo.14929817
"""

import numpy as np
from typing import Dict, List, Tuple, Union, Optional
import warnings

class XenopoulosKlein4Group:
    """
    Complete implementation of Piaget's INRC operators forming a Klein-4 group.
    
    The Klein-4 group is the smallest non-cyclic group with 4 elements:
        V₄ = {I, N, R, C} where:
        - I: Identity operator
        - N: Negation operator  
        - R: Reciprocity operator
        - C: Correlation operator (C = N∘R = R∘N)
    
    Group Properties (Theorem 3.1 in Xenopoulos):
        1. Closure: For all a,b ∈ V₄, a∘b ∈ V₄
        2. Associativity: (a∘b)∘c = a∘(b∘c)
        3. Identity: I∘a = a∘I = a
        4. Inverses: a∘a = I for all a ∈ V₄
        5. Commutativity: a∘b = b∘a (abelian group)
    """
    
    def __init__(self, dimension: int = 3, dtype=np.float64):
        """
        Initialize the Klein-4 group for given dimension.
        
        Parameters:
        -----------
        dimension : int
            Dimension of the vector space operators act upon
        dtype : numpy dtype
            Data type for matrices (default: np.float64)
        
        Raises:
        -------
        ValueError: If dimension < 2
        """
        if dimension < 2:
            raise ValueError(f"Dimension must be ≥ 2, got {dimension}")
        
        self.dimension = dimension
        self.dtype = dtype
        
        # Initialize operators
        self.I = self._create_identity()
        self.N = self._create_negation()
        self.R = self._create_reciprocity()
        self.C = self._create_correlation()
        
        # Cayley table for quick lookups
        self._cayley_table = self._generate_cayley_table()
        
        # Validate group structure
        self._validate_group_structure()
        
        print(f"✅ Klein-4 Group initialized (dimension={dimension})")
    
    def _create_identity(self) -> np.ndarray:
        """Create Identity operator: I(x) = x"""
        return np.eye(self.dimension, dtype=self.dtype)
    
    def _create_negation(self) -> np.ndarray:
        """Create Negation operator: N(x) = -x"""
        return -np.eye(self.dimension, dtype=self.dtype)
    
    def _create_reciprocity(self) -> np.ndarray:
        """
        Create Reciprocity operator: R(x₁, x₂, ..., xₙ) = (xₙ, x₁, ..., xₙ₋₁)
        Cyclic permutation that preserves structure while reversing order
        """
        R = np.zeros((self.dimension, self.dimension), dtype=self.dtype)
        for i in range(self.dimension):
            R[i, (i + 1) % self.dimension] = 1.0
        return R
    
    def _create_correlation(self) -> np.ndarray:
        """
        Create Correlation operator: C = N∘R = R∘N
        Following Klein-4 group composition rule
        """
        # C = N ∘ R (matrix multiplication)
        return self.N @ self.R
    
    def _generate_cayley_table(self) -> Dict[str, Dict[str, np.ndarray]]:
        """Generate Cayley table for the Klein-4 group"""
        operators = {'I': self.I, 'N': self.N, 'R': self.R, 'C': self.C}
        table = {}
        
        for op1_name, op1 in operators.items():
            table[op1_name] = {}
            for op2_name, op2 in operators.items():
                table[op1_name][op2_name] = op1 @ op2
        
        return table
    
    def _validate_group_structure(self) -> None:
        """
        Validate all Klein-4 group properties.
        Raises ValueError if any property is violated.
        """
        operators = {'I': self.I, 'N': self.N, 'R': self.R, 'C': self.C}
        tolerance = 1e-10
        
        print("\n🔬 Validating Klein-4 Group Properties:")
        print("-" * 50)
        
        # Property 1: Self-inverses (a∘a = I)
        for name, op in operators.items():
            result = op @ op
            if not np.allclose(result, self.I, atol=tolerance):
                raise ValueError(f"Self-inverse violated for {name}: {name}∘{name} ≠ I")
            print(f"✓ {name}² = I")
        
        # Property 2: N∘R = C
        if not np.allclose(self.N @ self.R, self.C, atol=tolerance):
            raise ValueError("N∘R ≠ C")
        print("✓ N∘R = C")
        
        # Property 3: R∘N = C
        if not np.allclose(self.R @ self.N, self.C, atol=tolerance):
            raise ValueError("R∘N ≠ C")
        print("✓ R∘N = C")
        
        # Property 4: R∘C = N
        if not np.allclose(self.R @ self.C, self.N, atol=tolerance):
            raise ValueError("R∘C ≠ N")
        print("✓ R∘C = N")
        
        # Property 5: C∘R = N
        if not np.allclose(self.C @ self.R, self.N, atol=tolerance):
            raise ValueError("C∘R ≠ N")
        print("✓ C∘R = N")
        
        # Property 6: N∘C = R
        if not np.allclose(self.N @ self.C, self.R, atol=tolerance):
            raise ValueError("N∘C ≠ R")
        print("✓ N∘C = R")
        
        # Property 7: C∘N = R
        if not np.allclose(self.C @ self.N, self.R, atol=tolerance):
            raise ValueError("C∘N ≠ R")
        print("✓ C∘N = R")
        
        # Property 8: Associativity test (random vectors)
        test_vector = np.random.randn(self.dimension)
        for a_name, a in operators.items():
            for b_name, b in operators.items():
                for c_name, c in operators.items():
                    left = (a @ b) @ c @ test_vector
                    right = a @ (b @ c) @ test_vector
                    if not np.allclose(left, right, atol=tolerance):
                        raise ValueError(f"Associativity violated for {a_name}∘{b_name}∘{c_name}")
        print("✓ Associativity: (a∘b)∘c = a∘(b∘c)")
        
        # Property 9: Commutativity (abelian group)
        for a_name, a in operators.items():
            for b_name, b in operators.items():
                if not np.allclose(a @ b, b @ a, atol=tolerance):
                    raise ValueError(f"Commutativity violated: {a_name}∘{b_name} ≠ {b_name}∘{a_name}")
        print("✓ Commutativity: a∘b = b∘a (abelian)")
        
        print("-" * 50)
        print("✅ ALL Klein-4 group properties validated successfully!")
    
    def apply_operator(self, vector: np.ndarray, operator: str) -> np.ndarray:
        """
        Apply a single INRC operator to a vector.
        
        Parameters:
        -----------
        vector : np.ndarray
            Input vector of shape (dimension,) or (n, dimension)
        operator : str
            Operator to apply: 'I', 'N', 'R', or 'C'
        
        Returns:
        --------
        np.ndarray : Transformed vector
        
        Raises:
        -------
        ValueError: If operator not in {'I', 'N', 'R', 'C'}
        ValueError: If vector dimension doesn't match group dimension
        """
        if operator not in ['I', 'N', 'R', 'C']:
            raise ValueError(f"Operator must be 'I', 'N', 'R', or 'C', got '{operator}'")
        
        vector = np.asarray(vector, dtype=self.dtype)
        
        # Handle batched input
        if vector.ndim == 1:
            if vector.shape[0] != self.dimension:
                raise ValueError(f"Vector dimension {vector.shape[0]} doesn't match group dimension {self.dimension}")
            return self._apply_single(vector, operator)
        elif vector.ndim == 2:
            if vector.shape[1] != self.dimension:
                raise ValueError(f"Vector dimension {vector.shape[1]} doesn't match group dimension {self.dimension}")
            return np.array([self._apply_single(v, operator) for v in vector])
        else:
            raise ValueError(f"Vector must be 1D or 2D, got shape {vector.shape}")
    
    def _apply_single(self, vector: np.ndarray, operator: str) -> np.ndarray:
        """Apply operator to a single vector"""
        if operator == 'I':
            return vector.copy()
        elif operator == 'N':
            return -vector
        elif operator == 'R':
            return self.R @ vector
        elif operator == 'C':
            return self.C @ vector
    
    def apply_sequence(self, vector: np.ndarray, sequence: List[str]) -> np.ndarray:
        """
        Apply a sequence of INRC operators to a vector.
        
        Parameters:
        -----------
        vector : np.ndarray
            Input vector
        sequence : List[str]
            List of operators to apply in order
        
        Returns:
        --------
        np.ndarray : Result after applying all operators
        
        Example:
        --------
        >>> group.apply_sequence(vector, ['I', 'N', 'R', 'C'])
        # Equivalent to: C(R(N(I(vector))))
        """
        result = vector.copy()
        for op in sequence:
            result = self.apply_operator(result, op)
        return result
    
    def compose_operators(self, operator1: str, operator2: str) -> np.ndarray:
        """
        Compose two operators and return the resulting operator matrix.
        
        Parameters:
        -----------
        operator1, operator2 : str
            Operators to compose (must be 'I', 'N', 'R', or 'C')
        
        Returns:
        --------
        np.ndarray : Matrix representing operator1 ∘ operator2
        """
        if operator1 not in ['I', 'N', 'R', 'C'] or operator2 not in ['I', 'N', 'R', 'C']:
            raise ValueError("Operators must be 'I', 'N', 'R', or 'C'")
        
        return self._cayley_table[operator1][operator2]
    
    def get_operator_matrix(self, operator: str) -> np.ndarray:
        """
        Get the matrix representation of an operator.
        
        Parameters:
        -----------
        operator : str
            Operator name: 'I', 'N', 'R', or 'C'
        
        Returns:
        --------
        np.ndarray : Matrix representation
        """
        operators = {'I': self.I, 'N': self.N, 'R': self.R, 'C': self.C}
        return operators[operator].copy()
    
    def get_all_operators(self) -> Dict[str, np.ndarray]:
        """Get all four operators as a dictionary"""
        return {
            'I': self.I.copy(),
            'N': self.N.copy(),
            'R': self.R.copy(),
            'C': self.C.copy()
        }
    
    def get_cayley_table_symbolic(self) -> Dict[str, Dict[str, str]]:
        """
        Get Cayley table with symbolic operator names.
        
        Returns:
        --------
        Dict[str, Dict[str, str]] : Cayley table where table[a][b] = a∘b
        """
        operators = {'I': self.I, 'N': self.N, 'R': self.R, 'C': self.C}
        symbolic_table = {}
        
        print("\n📊 Cayley Table for Klein-4 Group:")
        print("   " + " ".join(['I', 'N', 'R', 'C']))
        print("  " + "-" * 13)
        
        for op1_name, op1 in operators.items():
            symbolic_table[op1_name] = {}
            row = f"{op1_name}: "
            for op2_name, op2 in operators.items():
                result = op1 @ op2
                # Find which operator this corresponds to
                for op_name, op in operators.items():
                    if np.allclose(result, op, atol=1e-10):
                        symbolic_table[op1_name][op2_name] = op_name
                        row += f"{op_name} "
                        break
            print(row)
        
        return symbolic_table
    
    def analyze_operator_properties(self, operator: str) -> Dict[str, Union[float, bool]]:
        """
        Analyze mathematical properties of an operator.
        
        Parameters:
        -----------
        operator : str
            Operator name: 'I', 'N', 'R', or 'C'
        
        Returns:
        --------
        Dict[str, Union[float, bool]] : Dictionary of properties
        """
        if operator not in ['I', 'N', 'R', 'C']:
            raise ValueError(f"Operator must be 'I', 'N', 'R', or 'C', got '{operator}'")
        
        op_matrix = self.get_operator_matrix(operator)
        
        properties = {
            'trace': float(np.trace(op_matrix)),
            'determinant': float(np.linalg.det(op_matrix)),
            'norm': float(np.linalg.norm(op_matrix)),
            'is_orthogonal': np.allclose(op_matrix @ op_matrix.T, np.eye(self.dimension)),
            'is_symmetric': np.allclose(op_matrix, op_matrix.T),
            'is_skew_symmetric': np.allclose(op_matrix, -op_matrix.T),
            'eigenvalues': np.linalg.eigvals(op_matrix).tolist()
        }
        
        return properties
    
    def generate_dialectical_cycle(self, initial_vector: np.ndarray, 
                                   sequence_type: str = 'standard') -> Dict[str, np.ndarray]:
        """
        Generate a complete dialectical cycle using INRC operators.
        
        Parameters:
        -----------
        initial_vector : np.ndarray
            Initial state (thesis)
        sequence_type : str
            Type of dialectical cycle:
            - 'standard': I → N → R → C
            - 'reverse': I → R → N → C
            - 'extended': I → N → R → C → N → R → I
        
        Returns:
        --------
        Dict[str, np.ndarray] : Dictionary mapping operator names to results
        """
        sequences = {
            'standard': ['I', 'N', 'R', 'C'],
            'reverse': ['I', 'R', 'N', 'C'],
            'extended': ['I', 'N', 'R', 'C', 'N', 'R', 'I']
        }
        
        if sequence_type not in sequences:
            raise ValueError(f"sequence_type must be one of {list(sequences.keys())}")
        
        results = {}
        current = initial_vector.copy()
        
        for op in sequences[sequence_type]:
            current = self.apply_operator(current, op)
            results[op] = current.copy()
        
        return results
    
    def visualize_operators(self, save_path: Optional[str] = None) -> None:
        """
        Create visualization of all operators.
        
        Parameters:
        -----------
        save_path : str, optional
            Path to save the visualization
        """
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        
        fig = plt.figure(figsize=(15, 10))
        
        # Generate test vectors
        if self.dimension == 2:
            theta = np.linspace(0, 2*np.pi, 8, endpoint=False)
            test_vectors = np.array([np.cos(theta), np.sin(theta)]).T
        elif self.dimension >= 3:
            # Use vertices of a cube/hypercube
            test_vectors = np.array([
                [1, 0, 0] if self.dimension >= 3 else [1, 0],
                [0, 1, 0] if self.dimension >= 3 else [0, 1],
                [0, 0, 1] if self.dimension >= 3 else [1, 1],
                [-1, 0, 0] if self.dimension >= 3 else [-1, 0],
                [0, -1, 0] if self.dimension >= 3 else [0, -1],
                [0, 0, -1] if self.dimension >= 3 else [-1, -1]
            ])
            if self.dimension > 3:
                # Pad with zeros for higher dimensions
                padding = np.zeros((test_vectors.shape[0], self.dimension - 3))
                test_vectors = np.hstack([test_vectors, padding])
        
        operators = self.get_all_operators()
        
        for idx, (op_name, op_matrix) in enumerate(operators.items(), 1):
            ax = fig.add_subplot(2, 2, idx, projection='3d' if self.dimension >= 3 else None)
            
            # Apply operator to test vectors
            transformed = np.array([op_matrix @ v for v in test_vectors])
            
            if self.dimension == 2:
                # 2D plot
                ax.scatter(test_vectors[:, 0], test_vectors[:, 1], 
                          c='blue', s=100, alpha=0.6, label='Original')
                ax.scatter(transformed[:, 0], transformed[:, 1], 
                          c='red', s=100, alpha=0.6, label=f'{op_name}-transformed')
                
                # Draw arrows from original to transformed
                for orig, trans in zip(test_vectors, transformed):
                    ax.arrow(orig[0], orig[1], 
                            trans[0]-orig[0], trans[1]-orig[1],
                            head_width=0.05, head_length=0.1, 
                            fc='green', ec='green', alpha=0.3)
                
                ax.set_xlim(-2, 2)
                ax.set_ylim(-2, 2)
                ax.grid(True)
                
            elif self.dimension >= 3:
                # 3D plot (show first 3 dimensions)
                ax.scatter(test_vectors[:, 0], test_vectors[:, 1], test_vectors[:, 2],
                          c='blue', s=100, alpha=0.6, label='Original')
                ax.scatter(transformed[:, 0], transformed[:, 1], transformed[:, 2],
                          c='red', s=100, alpha=0.6, label=f'{op_name}-transformed')
                
                ax.set_xlim(-2, 2)
                ax.set_ylim(-2, 2)
                ax.set_zlim(-2, 2)
                ax.grid(True)
            
            ax.set_title(f'Operator {op_name}', fontsize=12, fontweight='bold')
            ax.legend()
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            if self.dimension >= 3:
                ax.set_zlabel('Z')
        
        plt.suptitle(f'Klein-4 Group Operators (Dimension {self.dimension})', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Visualization saved to {save_path}")
        
        plt.show()
    
    def __repr__(self) -> str:
        """String representation of the Klein-4 group"""
        return (f"XenopoulosKlein4Group(dimension={self.dimension}, "
                f"dtype={self.dtype.__name__})")
    
    def __str__(self) -> str:
        """Human-readable description"""
        return (f"Klein-4 Group of INRC operators\n"
                f"• Dimension: {self.dimension}\n"
                f"• Operators: I (Identity), N (Negation), R (Reciprocity), C (Correlation)\n"
                f"• Properties: Abelian, self-inverse, associative")


# Utility functions for working with Klein-4 groups
def create_dialectical_transformation(thesis: np.ndarray, 
                                      antithesis: np.ndarray,
                                      group: XenopoulosKlein4Group) -> Dict[str, np.ndarray]:
    """
    Create a complete dialectical transformation using INRC operators.
    
    Parameters:
    -----------
    thesis : np.ndarray
        Initial thesis vector
    antithesis : np.ndarray  
        Antithesis vector
    group : XenopoulosKlein4Group
        Klein-4 group instance
    
    Returns:
    --------
    Dict[str, np.ndarray] : Dictionary of transformed vectors
    """
    results = {}
    
    # Apply each operator to thesis
    for op_name in ['I', 'N', 'R', 'C']:
        results[f'thesis_{op_name}'] = group.apply_operator(thesis, op_name)
    
    # Apply each operator to antithesis
    for op_name in ['I', 'N', 'R', 'C']:
        results[f'antithesis_{op_name}'] = group.apply_operator(antithesis, op_name)
    
    # Create synthesis as combination
    synthesis = 0.7 * results['thesis_I'] + 0.3 * results['antithesis_N']
    results['synthesis'] = synthesis
    
    return results


def validate_klein4_composition(a: str, b: str, expected: str, 
                                group: XenopoulosKlein4Group) -> bool:
    """
    Validate a composition in the Klein-4 group.
    
    Parameters:
    -----------
    a, b : str
        Operators to compose
    expected : str
        Expected result operator
    group : XenopoulosKlein4Group
        Klein-4 group instance
    
    Returns:
    --------
    bool : True if composition matches expected
    """
    result = group.compose_operators(a, b)
    expected_matrix = group.get_operator_matrix(expected)
    return np.allclose(result, expected_matrix, atol=1e-10)


# Example usage
if __name__ == "__main__":
    print("=" * 70)
    print("KLEIN-4 GROUP IMPLEMENTATION - DEMONSTRATION")
    print("=" * 70)
    
    # Create a 3D Klein-4 group
    print("\n1. Creating Klein-4 group (dimension=3)...")
    group = XenopoulosKlein4Group(dimension=3)
    
    # Display Cayley table
    print("\n2. Cayley table (composition rules):")
    table = group.get_cayley_table_symbolic()
    
    # Analyze operator properties
    print("\n3. Operator properties:")
    for op in ['I', 'N', 'R', 'C']:
        props = group.analyze_operator_properties(op)
        print(f"\n   Operator {op}:")
        print(f"   • Trace: {props['trace']:.2f}")
        print(f"   • Determinant: {props['determinant']:.2f}")
        print(f"   • Orthogonal: {props['is_orthogonal']}")
    
    # Demonstrate dialectical cycle
    print("\n4. Dialectical cycle demonstration:")
    initial_vector = np.array([1.0, 0.5, -0.5])
    print(f"   Initial vector: {initial_vector}")
    
    cycle = group.generate_dialectical_cycle(initial_vector, 'standard')
    for op, result in cycle.items():
        print(f"   {op}: {result}")
    
    # Test composition validation
    print("\n5. Composition validation:")
    tests = [('N', 'R', 'C'), ('R', 'N', 'C'), ('C', 'C', 'I')]
    for a, b, expected in tests:
        valid = validate_klein4_composition(a, b, expected, group)
        print(f"   {a}∘{b} = {expected}: {valid}")
    
    print("\n" + "=" * 70)
    print("DEMONSTRATION COMPLETE")
    print("=" * 70)
    
    