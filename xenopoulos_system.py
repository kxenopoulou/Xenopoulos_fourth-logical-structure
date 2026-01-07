"""
Xenopoulos INRC-Klein4 Dialectical System
Complete implementation of Epameinondas Xenopoulos' Fourth Logical Structure
Mathematization of Hegelian-Marxist dialectics through Piaget's INRC operators
"""

import numpy as np
import torch
import torch.nn as nn
from scipy.integrate import solve_ivp
from scipy.linalg import expm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class XenopoulosKlein4Group:
    """Complete Klein-4 group implementation of Piaget's INRC operators"""
    
    def __init__(self, dimension=3):
        self.dimension = dimension
        
        # Identity operator (I): x → x
        self.I = np.eye(dimension, dtype=np.float64)
        
        # Negation operator (N): x → -x (self-inverse: N ∘ N = I)
        self.N = -np.eye(dimension, dtype=np.float64)
        
        # Reciprocity operator (R): cyclic permutation
        self.R = self._create_reciprocity_operator()
        
        # Correlation operator (C): C = N ∘ R = R ∘ N
        self.C = self.N @ self.R
        
        # Verify Klein-4 group properties
        self._validate_klein4_group()
    
    def _create_reciprocity_operator(self):
        """Create reciprocity as cyclic permutation matrix"""
        R = np.zeros((self.dimension, self.dimension), dtype=np.float64)
        for i in range(self.dimension):
            R[i, (i + 1) % self.dimension] = 1.0
        return R
    
    def _validate_klein4_group(self):
        """Validate all Klein-4 group axioms"""
        validations = {
            "N² = I": np.allclose(self.N @ self.N, self.I),
            "R² = I": np.allclose(self.R @ self.R, self.I),
            "C² = I": np.allclose(self.C @ self.C, self.I),
            "N∘R = C": np.allclose(self.N @ self.R, self.C),
            "R∘N = C": np.allclose(self.R @ self.N, self.C),
            "R∘C = N": np.allclose(self.R @ self.C, self.N),
            "C∘R = N": np.allclose(self.C @ self.R, self.N),
            "N∘C = R": np.allclose(self.N @ self.C, self.R),
            "C∘N = R": np.allclose(self.C @ self.N, self.R)
        }
        
        print("Xenopoulos Klein-4 Group Validation:")
        for property_name, is_valid in validations.items():
            status = "✓" if is_valid else "✗"
            print(f"  {status} {property_name}")
        
        if all(validations.values()):
            print("✅ Group structure verified successfully")
        else:
            raise ValueError("Klein-4 group validation failed")
    
    def apply_operator(self, vector, operator_name):
        """Apply specific INRC operator to a vector"""
        operators = {
            'I': self.I,
            'N': self.N,
            'R': self.R,
            'C': self.C
        }
        
        if operator_name not in operators:
            raise ValueError(f"Operator must be one of {list(operators.keys())}")
        
        return operators[operator_name] @ vector
    
    def get_cayley_table(self):
        """Generate Cayley table for the Klein-4 group"""
        operators = {'I': self.I, 'N': self.N, 'R': self.R, 'C': self.C}
        table = {}
        
        for op1_name, op1 in operators.items():
            table[op1_name] = {}
            for op2_name, op2 in operators.items():
                result = op1 @ op2
                # Find which operator this corresponds to
                for op_name, op in operators.items():
                    if np.allclose(result, op):
                        table[op1_name][op2_name] = op_name
                        break
        
        return table

class XenopoulosDialecticalDynamics(nn.Module):
    """Implementation of Xenopoulos' D₁ and D₂ formalisms"""
    
    def __init__(self, input_dim=3, hidden_dim=16):
        super().__init__()
        
        # D₁: F → N → R → C (Multidimensional Synthesis)
        self.D1_network = nn.Sequential(
            nn.Linear(input_dim * 4, hidden_dim * 2),
            nn.Tanh(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Tanh()
        )
        
        # D₂: F → C → N → R (Dialectical Reversal)
        self.D2_network = nn.Sequential(
            nn.Linear(input_dim * 4, hidden_dim * 2),
            nn.ELU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Tanh()
        )
        
        # Xenopoulos synthesis parameters: S = α(I•N) - β|I-N| + γR
        self.alpha = nn.Parameter(torch.tensor(0.7, dtype=torch.float32))
        self.beta = nn.Parameter(torch.tensor(0.3, dtype=torch.float32))
        self.gamma = nn.Parameter(torch.tensor(0.4, dtype=torch.float32))
        
        # Historical memory weights
        self.historical_weights = nn.Parameter(
            torch.tensor([0.5, 0.3, 0.2], dtype=torch.float32)
        )
    
    def forward(self, thesis, antithesis, historical_context=None, mode='D1'):
        """Perform dialectical synthesis using specified formalism"""
        # Apply INRC operators
        identity = thesis  # I(x) = x
        negation = -antithesis  # N(x) = -x
        
        # Reciprocity: cyclic transformation
        reciprocity = torch.roll(thesis, shifts=1, dims=-1)
        
        # Correlation: combined effect
        correlation = negation + reciprocity
        
        if mode == 'D1':
            # D₁: F → N → R → C
            inputs = torch.cat([identity, negation, reciprocity, correlation], dim=-1)
            raw_synthesis = self.D1_network(inputs)
        else:
            # D₂: F → C → N → R
            inputs = torch.cat([thesis, correlation, negation, reciprocity], dim=-1)
            raw_synthesis = self.D2_network(inputs)
        
        # Apply Xenopoulos synthesis equation
        identity_dot_negation = torch.sum(identity * negation, dim=-1, keepdim=True)
        identity_minus_negation_norm = torch.norm(identity - negation, dim=-1, keepdim=True)
        
        xenopoulos_synthesis = (
            self.alpha * identity_dot_negation -
            self.beta * identity_minus_negation_norm +
            self.gamma * torch.mean(reciprocity, dim=-1, keepdim=True)
        )
        
        # Incorporate historical context if provided
        if historical_context is not None:
            historical_effect = torch.zeros_like(xenopoulos_synthesis)
            for i, weight in enumerate(self.historical_weights[:len(historical_context)]):
                historical_effect += weight * historical_context[-(i+1)]
            xenopoulos_synthesis += 0.2 * historical_effect
        
        final_synthesis = raw_synthesis + 0.3 * xenopoulos_synthesis
        
        return final_synthesis

class XenopoulosOntologicalConflict:
    """Model ontological contradictions as dynamical system"""
    
    def __init__(self, dimension=3):
        self.dimension = dimension
        
        # Ontological parameters
        self.growth_rate = 1.2
        self.competition_strength = 0.4
        self.cooperation_factor = 0.1
        self.noise_intensity = 0.02
        
        # Phase transition threshold
        self.phase_transition_threshold = 0.85
        
        # History tracking
        self.conflict_history = []
    
    def conflict_dynamics(self, t, state):
        """Differential equations for ontological conflict"""
        thesis = state[:self.dimension]
        antithesis = state[self.dimension:2*self.dimension]
        
        # Thesis dynamics: growth - competition + cooperation
        dthesis = (
            self.growth_rate * thesis -
            self.competition_strength * thesis * antithesis +
            self.cooperation_factor * antithesis
        )
        
        # Antithesis dynamics: similar but with phase shift
        dantithesis = (
            self.growth_rate * antithesis -
            self.competition_strength * antithesis * thesis +
            self.cooperation_factor * thesis
        )
        
        # Add stochastic noise
        noise = self.noise_intensity * np.random.randn(2 * self.dimension)
        
        return np.concatenate([dthesis, dantithesis]) + noise
    
    def evolve_conflict(self, initial_state, time_span=(0, 10)):
        """Evolve ontological conflict over time"""
        solution = solve_ivp(
            self.conflict_dynamics,
            time_span,
            initial_state,
            method='RK45',
            max_step=0.1,
            dense_output=True
        )
        
        final_state = solution.y[:, -1]
        self.conflict_history.append(final_state)
        
        # Check for phase transition
        conflict_magnitude = np.linalg.norm(final_state[:self.dimension] - 
                                          final_state[self.dimension:])
        
        phase_transition = conflict_magnitude > self.phase_transition_threshold
        
        return final_state, phase_transition

class XenopoulosFourthStructure:
    """Complete implementation of Xenopoulos' Fourth Logical Structure"""
    
    def __init__(self, dimension=3):
        self.dimension = dimension
        
        # Core components
        self.klein_group = XenopoulosKlein4Group(dimension)
        self.dialectics = XenopoulosDialecticalDynamics(dimension)
        self.ontology = XenopoulosOntologicalConflict(dimension)
        
        # System state
        self.thesis = np.random.randn(dimension).astype(np.float32)
        self.antithesis = -self.thesis + 0.1 * np.random.randn(dimension).astype(np.float32)
        self.synthesis_history = []
        
        # Control parameters
        self.qualitative_threshold = 0.8
        self.history_depth = 3
        self.chaos_factor = 0.03
        
        # Tracking
        self.epoch = 0
        self.qualitative_transitions = []
        self.mode_history = []
    
    def dialectical_step(self, include_chaos=True):
        """Execute one step of dialectical evolution"""
        # Convert to tensors
        thesis_tensor = torch.FloatTensor(self.thesis).unsqueeze(0)
        antithesis_tensor = torch.FloatTensor(self.antithesis).unsqueeze(0)
        
        # Get historical context
        historical_context = None
        if len(self.synthesis_history) >= self.history_depth:
            historical_context = [
                torch.FloatTensor(s).unsqueeze(0) 
                for s in self.synthesis_history[-self.history_depth:]
            ]
        
        # Choose dialectical mode (alternate between D1 and D2)
        mode = 'D1' if self.epoch % 2 == 0 else 'D2'
        self.mode_history.append(mode)
        
        # Perform dialectical synthesis
        with torch.no_grad():
            synthesis_tensor = self.dialectics(
                thesis_tensor, 
                antithesis_tensor, 
                historical_context,
                mode=mode
            )
        
        synthesis = synthesis_tensor.numpy().flatten()
        
        # Add chaos if requested
        if include_chaos:
            chaos = self.chaos_factor * np.random.randn(self.dimension)
            synthesis += chaos
        
        # Update history
        self.synthesis_history.append(synthesis.copy())
        
        # Truncate history if too long
        if len(self.synthesis_history) > 100:
            self.synthesis_history = self.synthesis_history[-100:]
        
        return synthesis
    
    def evolve_ontology(self):
        """Evolve ontological contradictions"""
        initial_state = np.concatenate([self.thesis, self.antithesis])
        final_state, phase_transition = self.ontology.evolve_conflict(initial_state)
        
        # Update states
        self.thesis = final_state[:self.dimension]
        self.antithesis = final_state[self.dimension:]
        
        return phase_transition
    
    def check_qualitative_transition(self, synthesis_norm):
        """Check if quantitative changes trigger qualitative transition"""
        if synthesis_norm > self.qualitative_threshold:
            print(f"[Epoch {self.epoch}] ⚡ QUALITATIVE TRANSITION: "
                  f"{synthesis_norm:.3f} > {self.qualitative_threshold}")
            
            # Negation of negation: new thesis emerges from synthesis
            new_thesis = 0.6 * self.thesis + 0.4 * self.synthesis_history[-1]
            
            # New antithesis emerges
            new_antithesis = -0.7 * new_thesis + 0.2 * np.random.randn(self.dimension)
            
            # Store transition
            self.qualitative_transitions.append({
                'epoch': self.epoch,
                'synthesis_norm': synthesis_norm,
                'new_thesis_norm': np.linalg.norm(new_thesis)
            })
            
            return new_thesis, new_antithesis, True
        
        return None, None, False
    
    def evolve_system(self, epochs=500):
        """Main evolution loop"""
        print("=" * 70)
        print("XENOPOULOS FOURTH LOGICAL STRUCTURE - FULL SYSTEM EVOLUTION")
        print("=" * 70)
        print(f"• Dimension: {self.dimension}")
        print(f"• Initial Thesis: {self.thesis}")
        print(f"• Initial Antithesis: {self.antithesis}")
        print(f"• Qualitative Threshold: {self.qualitative_threshold}")
        print("-" * 70)
        
        for epoch in range(epochs):
            self.epoch = epoch
            
            # 1. Dialectical synthesis
            synthesis = self.dialectical_step(include_chaos=True)
            synthesis_norm = np.linalg.norm(synthesis)
            
            # 2. Ontological evolution
            ontological_transition = self.evolve_ontology()
            
            # 3. Check for qualitative transition
            new_thesis, new_antithesis, transition = self.check_qualitative_transition(synthesis_norm)
            
            if transition:
                self.thesis = new_thesis
                self.antithesis = new_antithesis
            
            # 4. Progress reporting
            if epoch % 100 == 0 and epoch > 0:
                print(f"[Epoch {epoch}] Synthesis: {synthesis_norm:.4f} | "
                      f"Transitions: {len(self.qualitative_transitions)} | "
                      f"Mode: {self.mode_history[-1]}")
        
        print("-" * 70)
        print(f"✅ EVOLUTION COMPLETE")
        print(f"• Total epochs: {epochs}")
        print(f"• Qualitative transitions: {len(self.qualitative_transitions)}")
        print(f"• Final synthesis norm: {synthesis_norm:.4f}")
        print("=" * 70)
        
        return self.synthesis_history, self.qualitative_transitions
    
    def visualize_system(self):
        """Comprehensive visualization of the system dynamics"""
        if not self.synthesis_history:
            print("No data available for visualization")
            return
        
        synthesis_array = np.array(self.synthesis_history)
        
        fig = plt.figure(figsize=(22, 14))
        fig.suptitle('XENOPOULOS FOURTH LOGICAL STRUCTURE - COMPLETE ANALYSIS', 
                    fontsize=16, fontweight='bold', y=1.02)
        
        # 1. Dialectical Evolution
        ax1 = plt.subplot(3, 3, 1)
        synthesis_norms = [np.linalg.norm(s) for s in self.synthesis_history]
        ax1.plot(synthesis_norms, 'r-', linewidth=2, alpha=0.8, label='Synthesis Norm')
        ax1.axhline(self.qualitative_threshold, color='g', linestyle='--', 
                   linewidth=1.5, label=f'Threshold ({self.qualitative_threshold})')
        
        # Mark qualitative transitions
        if self.qualitative_transitions:
            transition_epochs = [t['epoch'] for t in self.qualitative_transitions]
            transition_values = [t['synthesis_norm'] for t in self.qualitative_transitions]
            ax1.scatter(transition_epochs, transition_values, 
                       color='gold', s=100, zorder=5, label='Qualitative Transitions')
        
        ax1.set_title('Dialectical Evolution (Theorem 4.2)')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('||Synthesis||')
        ax1.legend(loc='upper right')
        ax1.grid(True, alpha=0.3)
        
        # 2. Phase Space 3D
        ax2 = plt.subplot(3, 3, 2, projection='3d')
        if len(synthesis_array) > 10:
            ax2.plot(synthesis_array[:, 0], synthesis_array[:, 1], synthesis_array[:, 2],
                    'b-', alpha=0.6, linewidth=1)
            scatter = ax2.scatter(synthesis_array[:, 0], synthesis_array[:, 1], synthesis_array[:, 2],
                                 c=range(len(synthesis_array)), cmap='viridis', s=15, alpha=0.7)
            plt.colorbar(scatter, ax=ax2, label='Temporal Progression')
        ax2.set_title('Dialectical Phase Space')
        ax2.set_xlabel('Component 1')
        ax2.set_ylabel('Component 2')
        ax2.set_zlabel('Component 3')
        
        # 3. INRC Operators Visualization
        ax3 = plt.subplot(3, 3, 3)
        operators = ['Identity (I)', 'Negation (N)', 'Reciprocity (R)', 'Correlation (C)']
        traces = [
            np.trace(self.klein_group.I),
            np.trace(self.klein_group.N),
            np.trace(self.klein_group.R),
            np.trace(self.klein_group.C)
        ]
        bars = ax3.bar(operators, traces, color=['blue', 'red', 'green', 'purple'])
        ax3.set_title('INRC Operators (Klein-4 Group)')
        ax3.set_ylabel('Trace Value')
        for bar, trace in zip(bars, traces):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    f'{trace:.2f}', ha='center', fontsize=9)
        
        # 4. Synthesis Distribution
        ax4 = plt.subplot(3, 3, 4)
        ax4.hist(synthesis_norms, bins=30, density=True, alpha=0.7, 
                color='darkorange', edgecolor='black')
        ax4.axvline(np.mean(synthesis_norms), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(synthesis_norms):.3f}')
        ax4.axvline(np.median(synthesis_norms), color='blue', linestyle=':', 
                   label=f'Median: {np.median(synthesis_norms):.3f}')
        ax4.set_title('Synthesis Distribution')
        ax4.set_xlabel('Synthesis Norm')
        ax4.set_ylabel('Probability Density')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # 5. Parameter Evolution
        ax5 = plt.subplot(3, 3, 5)
        epochs_range = range(len(synthesis_norms))
        ax5.plot(epochs_range, synthesis_norms, 'b-', alpha=0.6, label='Synthesis')
        
        # Moving average
        window = 50
        if len(synthesis_norms) > window:
            moving_avg = np.convolve(synthesis_norms, np.ones(window)/window, mode='valid')
            ax5.plot(range(window-1, len(synthesis_norms)), moving_avg, 
                    'r-', linewidth=2, label=f'{window}-epoch MA')
        
        ax5.set_title('Parameter Evolution & Trends')
        ax5.set_xlabel('Epoch')
        ax5.set_ylabel('Synthesis Norm')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # 6. Mode Usage
        ax6 = plt.subplot(3, 3, 6)
        if self.mode_history:
            mode_counts = {'D1': 0, 'D2': 0}
            for mode in self.mode_history:
                mode_counts[mode] += 1
            
            modes = list(mode_counts.keys())
            counts = list(mode_counts.values())
            colors = ['green', 'orange']
            wedges, texts, autotexts = ax6.pie(counts, labels=modes, colors=colors,
                                              autopct='%1.1f%%', startangle=90)
            
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontweight('bold')
            
            ax6.set_title('Dialectical Mode Usage')
        
        # 7. Autocorrelation Analysis
        ax7 = plt.subplot(3, 3, 7)
        if len(synthesis_norms) > 50:
            autocorr = np.correlate(synthesis_norms, synthesis_norms, mode='full')
            autocorr = autocorr[len(synthesis_norms)-1:] / autocorr[len(synthesis_norms)-1]
            lags = range(min(50, len(autocorr)))
            ax7.plot(lags, autocorr[:len(lags)], 'k-', linewidth=2)
            ax7.axhline(0, color='r', linestyle='--', alpha=0.5)
            ax7.set_title('Synthesis Autocorrelation')
            ax7.set_xlabel('Lag')
            ax7.set_ylabel('Correlation')
            ax7.grid(True, alpha=0.3)
        
        # 8. Transition Analysis
        ax8 = plt.subplot(3, 3, 8)
        if self.qualitative_transitions:
            transition_epochs = [t['epoch'] for t in self.qualitative_transitions]
            transition_sizes = [t['synthesis_norm'] for t in self.qualitative_transitions]
            ax8.scatter(transition_epochs, transition_sizes, 
                       c=range(len(transition_epochs)), cmap='hot', s=80)
            ax8.set_title('Qualitative Transitions')
            ax8.set_xlabel('Epoch of Transition')
            ax8.set_ylabel('Synthesis Norm at Transition')
            ax8.grid(True, alpha=0.3)
        
        # 9. Component Analysis
        ax9 = plt.subplot(3, 3, 9)
        if len(synthesis_array) > 0:
            components = ['Comp 1', 'Comp 2', 'Comp 3'][:self.dimension]
            mean_values = np.mean(synthesis_array, axis=0)
            std_values = np.std(synthesis_array, axis=0)
            
            x_pos = np.arange(len(components))
            ax9.bar(x_pos, mean_values, yerr=std_values, 
                   capsize=5, alpha=0.7, color='teal')
            ax9.set_xticks(x_pos)
            ax9.set_xticklabels(components)
            ax9.set_title('Component Statistics')
            ax9.set_ylabel('Mean Value ± Std')
            ax9.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.show()

# Main execution
if __name__ == "__main__":
    print("🚀 INITIALIZING XENOPOULOS DIALECTICAL SYSTEM")
    print("=" * 70)
    
    # Create system
    system = XenopoulosFourthStructure(dimension=3)
    
    # Display Klein-4 group info
    cayley_table = system.klein_group.get_cayley_table()
    print("\nKlein-4 Cayley Table:")
    print("    I  N  R  C")
    for row in ['I', 'N', 'R', 'C']:
        print(f"{row}: ", end="")
        for col in ['I', 'N', 'R', 'C']:
            print(f"{cayley_table[row][col]} ", end="")
        print()
    
    # Evolve system
    synthesis_history, transitions = system.evolve_system(epochs=500)
    
    # Visualize
    system.visualize_system()
    
    # Final statistics
    print("\n📊 FINAL STATISTICS:")
    print(f"• Total synthesis states: {len(synthesis_history)}")
    print(f"• Qualitative transitions: {len(transitions)}")
    if len(transitions) > 0:
        avg_transition_epoch = np.mean([t['epoch'] for t in transitions])
        avg_transition_magnitude = np.mean([t['synthesis_norm'] for t in transitions])
        print(f"• Average transition epoch: {avg_transition_epoch:.1f}")
        print(f"• Average transition magnitude: {avg_transition_magnitude:.3f}")
    print("=" * 70)
    
    
