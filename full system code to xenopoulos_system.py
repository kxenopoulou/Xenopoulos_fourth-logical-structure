"""
Xenopoulos INRC-Klein4 Dialectical System
Complete implementation of Epameinondas Xenopoulos' Fourth Logical Structure
Mathematization of Hegelian-Marxist dialectics through Piaget's INRC operators
Full integration with neural networks, differential equations, and chaos theory
"""

import numpy as np
import torch
import torch.nn as nn
from scipy.integrate import solve_ivp
from scipy.linalg import expm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys
import os

# ============================================================================
# 1. XENOPOULOS KLEIN-4 GROUP (INRC OPERATORS)
# ============================================================================

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
    
    def get_all_transformations(self, vector):
        """Apply all INRC operators to a vector and return results"""
        return {
            'I': self.apply_operator(vector, 'I'),
            'N': self.apply_operator(vector, 'N'),
            'R': self.apply_operator(vector, 'R'),
            'C': self.apply_operator(vector, 'C')
        }

# ============================================================================
# 2. XENOPOULOS DIALECTICAL DYNAMICS (D₁ & D₂ FORMALISMS)
# ============================================================================

class XenopoulosDialecticalDynamics(nn.Module):
    """Implementation of Xenopoulos' D₁ and D₂ formalisms"""
    
    def __init__(self, input_dim=3, hidden_dim=16, qualitative_threshold=0.8, device='auto'):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.qualitative_threshold = qualitative_threshold
        
        # Automatic device selection
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
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
        
        # Historical memory weights (Xenopoulos: last 3 states influence)
        self.historical_weights = nn.Parameter(
            torch.tensor([0.5, 0.3, 0.2], dtype=torch.float32)
        )
        
        # Move to device
        self.to(self.device)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights using Xavier initialization"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def _apply_inrc_operators(self, thesis, antithesis):
        """Apply all four INRC operators to thesis and antithesis"""
        # I(x) = x (Identity)
        identity = thesis
        
        # N(x) = -x (Negation)
        negation = -antithesis
        
        # R(x): cyclic transformation (Reciprocity)
        reciprocity = torch.roll(thesis, shifts=1, dims=-1)
        
        # C(x) = N∘R(x) = R∘N(x) (Correlation)
        correlation = negation + reciprocity
        
        return identity, negation, reciprocity, correlation
    
    def forward(self, thesis, antithesis, historical_context=None, mode='D1'):
        """Perform dialectical synthesis using Xenopoulos' formalisms"""
        if mode not in ['D1', 'D2']:
            raise ValueError(f"Mode must be 'D1' or 'D2', got '{mode}'")
        
        # 1. APPLY INRC OPERATORS
        identity, negation, reciprocity, correlation = self._apply_inrc_operators(thesis, antithesis)
        
        # 2. APPLY XENOPOULOS FORMALISM D₁ OR D₂
        if mode == 'D1':
            # D₁: F → N → R → C (Multidimensional Synthesis)
            inputs = torch.cat([identity, negation, reciprocity, correlation], dim=-1)
            raw_synthesis = self.D1_network(inputs)
        else:
            # D₂: F → C → N → R (Dialectical Reversal)
            inputs = torch.cat([thesis, correlation, negation, reciprocity], dim=-1)
            raw_synthesis = self.D2_network(inputs)
        
        # 3. APPLY XENOPOULOS SYNTHESIS EQUATION (Theorem 4.2)
        identity_dot_negation = torch.sum(identity * negation, dim=-1, keepdim=True)
        identity_minus_negation_norm = torch.norm(identity - negation, dim=-1, keepdim=True)
        
        xenopoulos_synthesis = (
            self.alpha * identity_dot_negation -
            self.beta * identity_minus_negation_norm +
            self.gamma * torch.mean(reciprocity, dim=-1, keepdim=True)
        )
        
        # 4. INCORPORATE HISTORICAL CONTEXT (Xenopoulos: historical retrospection)
        if historical_context is not None and len(historical_context) > 0:
            historical_effect = torch.zeros_like(xenopoulos_synthesis)
            num_context = min(len(historical_context), len(self.historical_weights))
            
            for i in range(num_context):
                weight = self.historical_weights[i]
                context_value = historical_context[-(i+1)]
                
                # Ensure context has correct shape
                if context_value.shape != historical_effect.shape:
                    if context_value.dim() == 1:
                        context_value = context_value.unsqueeze(0)
                    if context_value.shape[0] != historical_effect.shape[0]:
                        context_value = context_value.expand(historical_effect.shape[0], -1)
                
                historical_effect += weight * context_value
            
            xenopoulos_synthesis += 0.2 * historical_effect
        
        # 5. COMBINE RAW SYNTHESIS WITH XENOPOULOS EQUATION
        final_synthesis = raw_synthesis + 0.3 * xenopoulos_synthesis
        
        # 6. CALCULATE METRICS
        synthesis_norm = torch.norm(final_synthesis, dim=-1).mean().item()
        qualitative_transition = synthesis_norm > self.qualitative_threshold
        
        return {
            'synthesis': final_synthesis,
            'identity': identity,
            'negation': negation,
            'reciprocity': reciprocity,
            'correlation': correlation,
            'qualitative_transition': qualitative_transition,
            'synthesis_norm': synthesis_norm,
            'mode': mode
        }
    
    def dialectical_cycle(self, thesis, antithesis, steps=5, mode='D1'):
        """Perform a complete dialectical cycle over multiple steps"""
        # Convert to tensors
        thesis_tensor = torch.FloatTensor(thesis).unsqueeze(0).to(self.device)
        antithesis_tensor = torch.FloatTensor(antithesis).unsqueeze(0).to(self.device)
        
        history = {
            'thesis': [thesis.copy()],
            'antithesis': [antithesis.copy()],
            'synthesis': [],
            'synthesis_norms': [],
            'qualitative_transitions': []
        }
        
        historical_context = []
        
        for step in range(steps):
            with torch.no_grad():
                result = self.forward(
                    thesis_tensor, antithesis_tensor,
                    historical_context, mode=mode
                )
            
            # Extract results
            synthesis = result['synthesis'].cpu().numpy()[0]
            synthesis_norm = result['synthesis_norm']
            transition = result['qualitative_transition']
            
            # Update history
            history['synthesis'].append(synthesis)
            history['synthesis_norms'].append(synthesis_norm)
            history['qualitative_transitions'].append(transition)
            
            # Update historical context
            historical_context.append(result['synthesis'].detach())
            if len(historical_context) > 3:  # Keep only last 3
                historical_context = historical_context[-3:]
            
            # Update thesis/antithesis for next step (dialectical progression)
            if step < steps - 1:
                thesis_tensor = result['synthesis'].detach()
                antithesis_tensor = -thesis_tensor + 0.1 * torch.randn_like(thesis_tensor)
                
                history['thesis'].append(thesis_tensor.cpu().numpy()[0])
                history['antithesis'].append(antithesis_tensor.cpu().numpy()[0])
        
        return history
    
    def analyze_synthesis(self, thesis, antithesis, n_iterations=100):
        """Analyze synthesis properties with Monte Carlo sampling"""
        thesis_tensor = torch.FloatTensor(thesis).unsqueeze(0).to(self.device)
        antithesis_tensor = torch.FloatTensor(antithesis).unsqueeze(0).to(self.device)
        
        syntheses = []
        norms = []
        
        for _ in range(n_iterations):
            with torch.no_grad():
                # Add small noise for sampling
                noisy_thesis = thesis_tensor + 0.01 * torch.randn_like(thesis_tensor)
                noisy_antithesis = antithesis_tensor + 0.01 * torch.randn_like(antithesis_tensor)
                
                # Alternate between D1 and D2
                mode = 'D1' if np.random.random() > 0.5 else 'D2'
                result = self.forward(noisy_thesis, noisy_antithesis, mode=mode)
                
                syntheses.append(result['synthesis'].cpu().numpy())
                norms.append(result['synthesis_norm'])
        
        syntheses_array = np.array(syntheses).squeeze()
        norms_array = np.array(norms)
        
        return {
            'mean_synthesis': np.mean(syntheses_array, axis=0),
            'std_synthesis': np.std(syntheses_array, axis=0),
            'mean_norm': np.mean(norms_array),
            'std_norm': np.std(norms_array),
            'min_norm': np.min(norms_array),
            'max_norm': np.max(norms_array),
            'probability_qualitative': np.mean(np.array(norms_array) > self.qualitative_threshold)
        }

# ============================================================================
# 3. XENOPOULOS ONTOLOGICAL CONFLICT
# ============================================================================

class XenopoulosOntologicalConflict:
    """Model ontological contradictions as dynamical system"""
    
    def __init__(self, dimension=3, growth_rate=1.2, competition_strength=0.4, 
                 phase_transition_threshold=0.85):
        self.dimension = dimension
        self.growth_rate = growth_rate
        self.competition_strength = competition_strength
        self.phase_transition_threshold = phase_transition_threshold
        
        # Additional parameters
        self.cooperation_factor = 0.1
        self.noise_intensity = 0.02
        
        # History tracking
        self.conflict_history = []
        self.transition_history = []
    
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
    
    def evolve_conflict(self, initial_state, time_span=(0, 5)):
        """Evolve ontological conflict over time"""
        try:
            solution = solve_ivp(
                self.conflict_dynamics,
                time_span,
                initial_state,
                method='RK45',
                max_step=0.1,
                dense_output=True
            )
            
            final_state = solution.y[:, -1]
        except Exception as e:
            # Fallback to simple integration if solve_ivp fails
            print(f"⚠️  solve_ivp failed, using simple integration: {e}")
            t0, t1 = time_span
            dt = 0.01
            state = initial_state.copy()
            for t in np.arange(t0, t1, dt):
                derivative = self.conflict_dynamics(t, state)
                state = state + derivative * dt
            final_state = state
        
        self.conflict_history.append(final_state)
        
        # Check for phase transition
        conflict_magnitude = np.linalg.norm(
            final_state[:self.dimension] - final_state[self.dimension:]
        )
        
        phase_transition = conflict_magnitude > self.phase_transition_threshold
        
        # Record transition if it occurred
        if phase_transition:
            self.transition_history.append({
                'time': time_span[1],
                'magnitude': conflict_magnitude,
                'state': final_state.copy()
            })
        
        return final_state, phase_transition
    
    def get_stability_metrics(self):
        """Calculate stability metrics from conflict history"""
        if not self.conflict_history:
            return {}
        
        states = np.array(self.conflict_history)
        thesis_states = states[:, :self.dimension]
        antithesis_states = states[:, self.dimension:]
        
        # Calculate conflict magnitudes
        conflicts = np.linalg.norm(thesis_states - antithesis_states, axis=1)
        
        return {
            'mean_conflict': np.mean(conflicts),
            'std_conflict': np.std(conflicts),
            'max_conflict': np.max(conflicts),
            'min_conflict': np.min(conflicts),
            'transition_count': len(self.transition_history)
        }

# ============================================================================
# 4. XENOPOULOS FOURTH LOGICAL STRUCTURE (COMPLETE SYSTEM)
# ============================================================================

class XenopoulosFourthStructure:
    """Complete implementation of Xenopoulos' Fourth Logical Structure"""
    
    def __init__(self, dimension=3, chaos_factor=0.03, 
                 qualitative_threshold=0.8, history_depth=3):
        self.dimension = dimension
        
        # Core components
        self.klein_group = XenopoulosKlein4Group(dimension)
        self.dialectics = XenopoulosDialecticalDynamics(
            input_dim=dimension,
            qualitative_threshold=qualitative_threshold
        )
        self.ontology = XenopoulosOntologicalConflict(dimension=dimension)
        
        # System state
        self.thesis = np.random.randn(dimension).astype(np.float32)
        self.antithesis = -self.thesis + 0.1 * np.random.randn(dimension).astype(np.float32)
        self.synthesis_history = []
        
        # Control parameters
        self.qualitative_threshold = qualitative_threshold
        self.history_depth = history_depth
        self.chaos_factor = chaos_factor
        
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
                'new_thesis_norm': np.linalg.norm(new_thesis),
                'thesis_before': self.thesis.copy(),
                'thesis_after': new_thesis.copy()
            })
            
            return new_thesis, new_antithesis, True
        
        return None, None, False
    
    def evolve_system(self, epochs=500, verbose=True):
        """Main evolution loop for complete dialectical process"""
        if verbose:
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
            if verbose and epoch % 100 == 0 and epoch > 0:
                print(f"[Epoch {epoch}] Synthesis: {synthesis_norm:.4f} | "
                      f"Transitions: {len(self.qualitative_transitions)} | "
                      f"Mode: {self.mode_history[-1]}")
        
        if verbose:
            print("-" * 70)
            print(f"✅ EVOLUTION COMPLETE")
            print(f"• Total epochs: {epochs}")
            print(f"• Qualitative transitions: {len(self.qualitative_transitions)}")
            print(f"• Final synthesis norm: {synthesis_norm:.4f}")
            print("=" * 70)
        
        return self.synthesis_history, self.qualitative_transitions
    
    def visualize_complete_system(self):
        """Create comprehensive visualization of system dynamics"""
        if not self.synthesis_history:
            print("No data available for visualization")
            return None
        
        synthesis_array = np.array(self.synthesis_history)
        
        # Set matplotlib backend to avoid display issues
        import matplotlib
        matplotlib.use('Agg')
        
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
        try:
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
        except:
            # Fallback to 2D plot if 3D fails
            ax2 = plt.subplot(3, 3, 2)
            if len(synthesis_array) > 10:
                ax2.plot(synthesis_array[:, 0], synthesis_array[:, 1],
                        'b-', alpha=0.6, linewidth=1)
                ax2.set_title('2D Phase Space Projection')
                ax2.set_xlabel('Component 1')
                ax2.set_ylabel('Component 2')
                ax2.grid(True, alpha=0.3)
        
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
        
        # Save figure
        os.makedirs('images', exist_ok=True)
        plt.savefig('images/xenopoulos_complete_analysis.png', dpi=300, bbox_inches='tight')
        
        return fig
    
    def get_system_metrics(self):
        """Get comprehensive system metrics"""
        if not self.synthesis_history:
            return {}
        
        synthesis_array = np.array(self.synthesis_history)
        synthesis_norms = [np.linalg.norm(s) for s in self.synthesis_history]
        
        return {
            'dimension': self.dimension,
            'total_epochs': self.epoch,
            'synthesis_count': len(self.synthesis_history),
            'qualitative_transitions': len(self.qualitative_transitions),
            'mean_synthesis_norm': np.mean(synthesis_norms),
            'std_synthesis_norm': np.std(synthesis_norms),
            'max_synthesis_norm': np.max(synthesis_norms),
            'min_synthesis_norm': np.min(synthesis_norms),
            'mode_usage': {
                'D1': self.mode_history.count('D1'),
                'D2': self.mode_history.count('D2')
            },
            'final_thesis_norm': np.linalg.norm(self.thesis),
            'final_antithesis_norm': np.linalg.norm(self.antithesis),
            'final_synthesis_norm': synthesis_norms[-1] if synthesis_norms else 0
        }

# ============================================================================
# 5. UTILITY FUNCTIONS
# ============================================================================

def create_dialectical_transformation(thesis, antithesis, group):
    """Create a complete dialectical transformation using INRC operators"""
    results = {}
    
    # Apply all operators
    for op_name in ['I', 'N', 'R', 'C']:
        results[f'thesis_{op_name}'] = group.apply_operator(thesis, op_name)
        results[f'antithesis_{op_name}'] = group.apply_operator(antithesis, op_name)
    
    # Create synthesis as combination
    results['synthesis'] = 0.5 * (results['thesis_I'] + results['antithesis_N'])
    