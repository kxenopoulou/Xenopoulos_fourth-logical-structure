"""
XenopoulosOntologicalConflict - Dynamical systems model for ontological contradictions
Implementation of Hegelian-Marxist dialectical materialism as differential equations

Author: Epameinondas Xenopoulos (Theoretical Framework)
Implementation: [Your Name]
Date: 2024

Based on: Xenopoulos' Theorem 4.3: Ontological conflicts as dynamical systems
Differential equations modeling the master-slave dialectic and class struggle dynamics
"""

import numpy as np
from scipy.integrate import solve_ivp, odeint
from typing import Dict, List, Tuple, Optional, Union, Callable
import warnings
from dataclasses import dataclass, field
from enum import Enum
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class ConflictType(Enum):
    """Types of ontological conflicts modeled by the system"""
    MASTER_SLAVE = "master_slave"      # Hegelian recognition struggle
    CLASS_STRUGGLE = "class_struggle"  # Marxist economic conflict
    NATURE_CULTURE = "nature_culture"  # Ecological contradiction
    INDIVIDUAL_SOCIETY = "individual_society"  # Social contradiction
    QUANTITATIVE_QUALITATIVE = "quantitative_qualitative"  # Dialectical transition

@dataclass
class ConflictHistory:
    """Data structure for tracking conflict evolution"""
    time_points: List[float] = field(default_factory=list)
    thesis_states: List[np.ndarray] = field(default_factory=list)
    antithesis_states: List[np.ndarray] = field(default_factory=list)
    synthesis_norms: List[float] = field(default_factory=list)
    phase_transitions: List[Dict] = field(default_factory=list)
    conflict_intensities: List[float] = field(default_factory=list)

class XenopoulosOntologicalConflict:
    """
    Dynamical systems model for ontological contradictions based on Hegelian-Marxist dialectics.
    
    Models contradictions as coupled differential equations:
        dT/dt = growth×T - competition×T×A + cooperation×A + noise
        dA/dt = growth×A - competition×A×T + cooperation×T + noise
    
    Where:
        T: Thesis state (e.g., master, bourgeoisie, culture)
        A: Antithesis state (e.g., slave, proletariat, nature)
    """
    
    def __init__(self, 
                 dimension: int = 3,
                 conflict_type: Union[str, ConflictType] = ConflictType.MASTER_SLAVE,
                 growth_rate: float = 1.2,
                 competition_strength: float = 0.4,
                 cooperation_factor: float = 0.1,
                 noise_intensity: float = 0.02,
                 phase_transition_threshold: float = 0.85,
                 history_capacity: int = 1000):
        """
        Initialize ontological conflict system.
        
        Parameters:
        -----------
        dimension : int
            Dimension of state space (number of contradictory aspects)
        conflict_type : str or ConflictType
            Type of ontological conflict to model
        growth_rate : float
            Intrinsic growth rate of both thesis and antithesis
        competition_strength : float
            Strength of competitive interaction (negative feedback)
        cooperation_factor : float
            Strength of cooperative interaction (positive feedback)
        noise_intensity : float
            Intensity of stochastic noise (dialectical uncertainty)
        phase_transition_threshold : float
            Threshold for detecting qualitative phase transitions
        history_capacity : int
            Maximum number of states to store in history
        """
        self.dimension = dimension
        
        # Convert string to ConflictType if needed
        if isinstance(conflict_type, str):
            try:
                self.conflict_type = ConflictType(conflict_type)
            except ValueError:
                warnings.warn(f"Unknown conflict type: {conflict_type}. Using MASTER_SLAVE.")
                self.conflict_type = ConflictType.MASTER_SLAVE
        else:
            self.conflict_type = conflict_type
        
        # Dynamical parameters
        self.growth_rate = growth_rate
        self.competition_strength = competition_strength
        self.cooperation_factor = cooperation_factor
        self.noise_intensity = noise_intensity
        self.phase_transition_threshold = phase_transition_threshold
        
        # Adjust parameters based on conflict type
        self._adjust_parameters_for_conflict_type()
        
        # History tracking
        self.history = ConflictHistory()
        self.history_capacity = history_capacity
        
        # Current state
        self.current_thesis = np.random.randn(dimension) * 0.5
        self.current_antithesis = np.random.randn(dimension) * 0.5
        
        # Phase transition counter
        self.phase_transition_count = 0
        
        # Integration method
        self.integration_method = 'RK45'  # Runge-Kutta 4/5
        
        print(f"✅ XenopoulosOntologicalConflict initialized")
        print(f"   Type: {self.conflict_type.value}")
        print(f"   Dimension: {dimension}")
        print(f"   Growth: {self.growth_rate}, Competition: {self.competition_strength}")
        print(f"   Cooperation: {self.cooperation_factor}, Noise: {self.noise_intensity}")
        print(f"   Phase transition threshold: {self.phase_transition_threshold}")
    
    def _adjust_parameters_for_conflict_type(self):
        """Adjust dynamical parameters based on conflict type"""
        if self.conflict_type == ConflictType.MASTER_SLAVE:
            # Master-slave: High competition, low cooperation
            self.competition_strength = 0.6
            self.cooperation_factor = 0.05
            self.growth_rate = 1.0
            
        elif self.conflict_type == ConflictType.CLASS_STRUGGLE:
            # Class struggle: Medium competition, some cooperation
            self.competition_strength = 0.5
            self.cooperation_factor = 0.15
            self.growth_rate = 1.1
            self.noise_intensity = 0.03  # Higher uncertainty
            
        elif self.conflict_type == ConflictType.NATURE_CULTURE:
            # Nature-culture: Cooperation emphasized
            self.competition_strength = 0.3
            self.cooperation_factor = 0.25
            self.growth_rate = 0.9
            self.phase_transition_threshold = 0.7
            
        elif self.conflict_type == ConflictType.INDIVIDUAL_SOCIETY:
            # Individual-society: Balanced competition/cooperation
            self.competition_strength = 0.4
            self.cooperation_factor = 0.2
            self.growth_rate = 1.2
            
        elif self.conflict_type == ConflictType.QUANTITATIVE_QUALITATIVE:
            # Quantitative-qualitative: High growth, high thresholds
            self.growth_rate = 1.5
            self.phase_transition_threshold = 0.9
            self.noise_intensity = 0.01  # Lower noise for clearer transitions
    
    def conflict_dynamics(self, t: float, state: np.ndarray) -> np.ndarray:
        """
        Differential equations for ontological conflict (Hegelian master-slave dialectic).
        
        Parameters:
        -----------
        t : float
            Time parameter (used for time-dependent dynamics)
        state : np.ndarray
            Combined state vector: [thesis, antithesis]
            Shape: (2 * dimension,)
        
        Returns:
        --------
        np.ndarray : Time derivatives d(state)/dt
        """
        thesis = state[:self.dimension]
        antithesis = state[self.dimension:]
        
        # THESIS DYNAMICS: growth - competition + cooperation
        # dT/dt = αT - βT∘A + γA
        dthesis = (
            self.growth_rate * thesis -
            self.competition_strength * thesis * np.abs(antithesis) +
            self.cooperation_factor * antithesis
        )
        
        # ANTITHESIS DYNAMICS: similar structure but with possible asymmetry
        # dA/dt = α'A - β'A∘T + γ'T
        dantithesis = (
            self.growth_rate * antithesis -
            self.competition_strength * antithesis * np.abs(thesis) +
            self.cooperation_factor * thesis
        )
        
        # ADD STOCHASTIC NOISE (dialectical uncertainty)
        if self.noise_intensity > 0:
            noise = self.noise_intensity * np.random.randn(2 * self.dimension)
        else:
            noise = np.zeros(2 * self.dimension)
        
        # Add time-dependent perturbation for certain conflict types
        if self.conflict_type == ConflictType.CLASS_STRUGGLE:
            # Class struggle has periodic crises
            crisis_factor = 0.1 * np.sin(0.5 * t)  # Periodic crisis
            noise[:self.dimension] += crisis_factor * np.random.randn(self.dimension)
        
        return np.concatenate([dthesis, dantithesis]) + noise
    
    def _calculate_synthesis(self, thesis: np.ndarray, antithesis: np.ndarray) -> np.ndarray:
        """
        Calculate synthesis from thesis and antithesis.
        
        Following Xenopoulos' synthesis equation:
            S = 0.7*Thesis + 0.3*Negation(Antithesis) + historical_terms
        """
        # Basic synthesis: weighted combination
        basic_synthesis = 0.7 * thesis + 0.3 * (-antithesis)
        
        # Add nonlinear terms
        nonlinear = 0.1 * thesis * antithesis * np.sin(np.sum(thesis + antithesis))
        
        # Add historical influence if available
        historical_influence = np.zeros_like(thesis)
        if len(self.history.thesis_states) > 0:
            # Use exponentially weighted moving average
            decay = 0.8
            for i, (past_thesis, past_antithesis) in enumerate(
                zip(reversed(self.history.thesis_states[-3:]), 
                    reversed(self.history.antithesis_states[-3:]))
            ):
                weight = decay ** (i + 1)
                historical_influence += weight * (past_thesis + past_antithesis)
        
        return basic_synthesis + nonlinear + 0.2 * historical_influence
    
    def _check_phase_transition(self, synthesis_norm: float, 
                                conflict_intensity: float) -> bool:
        """
        Check if conditions warrant a qualitative phase transition.
        
        Phase transitions occur when:
        1. Synthesis norm exceeds threshold, OR
        2. Conflict intensity is very high, OR
        3. System reaches bifurcation point
        """
        condition1 = synthesis_norm > self.phase_transition_threshold
        condition2 = conflict_intensity > 0.95
        condition3 = synthesis_norm > 0.7 and conflict_intensity > 0.8
        
        return condition1 or condition2 or condition3
    
    def evolve_conflict(self, 
                        time_span: Tuple[float, float] = (0, 10),
                        initial_state: Optional[np.ndarray] = None,
                        n_steps: int = 100,
                        record_history: bool = True) -> Dict[str, Union[np.ndarray, bool]]:
        """
        Evolve ontological conflict over specified time span.
        
        Parameters:
        -----------
        time_span : Tuple[float, float]
            Time span for evolution (start, end)
        initial_state : np.ndarray, optional
            Initial combined state [thesis, antithesis]. If None, uses current state.
        n_steps : int
            Number of time steps for integration
        record_history : bool
            Whether to record states in history
        
        Returns:
        --------
        Dict containing:
            - final_state: Final combined state
            - phase_transition: Whether phase transition occurred
            - synthesis_norm: Norm of final synthesis
            - conflict_intensity: Intensity of conflict
            - solution: Full integration solution object
        """
        # Prepare initial state
        if initial_state is None:
            initial_state = np.concatenate([self.current_thesis, self.current_antithesis])
        
        # Check dimensions
        if len(initial_state) != 2 * self.dimension:
            raise ValueError(
                f"Initial state must have length {2 * self.dimension}, "
                f"got {len(initial_state)}"
            )
        
        # Integrate differential equations
        try:
            solution = solve_ivp(
                self.conflict_dynamics,
                time_span,
                initial_state,
                method=self.integration_method,
                max_step=(time_span[1] - time_span[0]) / n_steps,
                dense_output=True,
                vectorized=False
            )
            
            if not solution.success:
                warnings.warn(f"Integration failed: {solution.message}")
                # Fallback to simpler integration
                t_eval = np.linspace(time_span[0], time_span[1], n_steps)
                solution.y = odeint(
                    lambda y, t: self.conflict_dynamics(t, y),
                    initial_state,
                    t_eval
                ).T
                solution.t = t_eval
                solution.success = True
        
        except Exception as e:
            warnings.warn(f"Integration error: {e}. Using Euler method.")
            # Simple Euler integration as fallback
            dt = (time_span[1] - time_span[0]) / n_steps
            t_points = np.linspace(time_span[0], time_span[1], n_steps)
            y_points = np.zeros((2 * self.dimension, n_steps))
            y_points[:, 0] = initial_state
            
            for i in range(1, n_steps):
                dy = self.conflict_dynamics(t_points[i-1], y_points[:, i-1])
                y_points[:, i] = y_points[:, i-1] + dy * dt
            
            solution = type('obj', (object,), {
                'y': y_points,
                't': t_points,
                'success': True,
                'message': 'Euler integration (fallback)'
            })()
        
        # Extract final state
        final_state = solution.y[:, -1]
        final_thesis = final_state[:self.dimension]
        final_antithesis = final_state[self.dimension:]
        
        # Update current state
        self.current_thesis = final_thesis.copy()
        self.current_antithesis = final_antithesis.copy()
        
        # Calculate synthesis and metrics
        synthesis = self._calculate_synthesis(final_thesis, final_antithesis)
        synthesis_norm = np.linalg.norm(synthesis)
        conflict_intensity = np.linalg.norm(final_thesis - final_antithesis)
        
        # Check for phase transition
        phase_transition = self._check_phase_transition(synthesis_norm, conflict_intensity)
        
        if phase_transition:
            self.phase_transition_count += 1
            transition_data = {
                'time': time_span[1],
                'synthesis_norm': synthesis_norm,
                'conflict_intensity': conflict_intensity,
                'thesis': final_thesis.copy(),
                'antithesis': final_antithesis.copy(),
                'transition_number': self.phase_transition_count
            }
            
            if record_history:
                self.history.phase_transitions.append(transition_data)
        
        # Record history if requested
        if record_history:
            self._record_history(
                solution.t, 
                solution.y[:self.dimension, :], 
                solution.y[self.dimension:, :],
                synthesis_norm,
                conflict_intensity
            )
        
        return {
            'final_state': final_state,
            'phase_transition': phase_transition,
            'synthesis_norm': synthesis_norm,
            'conflict_intensity': conflict_intensity,
            'synthesis': synthesis,
            'solution': solution
        }
    
    def _record_history(self, 
                        time_points: np.ndarray,
                        thesis_states: np.ndarray,
                        antithesis_states: np.ndarray,
                        synthesis_norm: float,
                        conflict_intensity: float):
        """Record evolution history"""
        # Convert to lists for easier handling
        times = time_points.tolist()
        theses = [thesis_states[:, i] for i in range(thesis_states.shape[1])]
        antitheses = [antithesis_states[:, i] for i in range(antithesis_states.shape[1])]
        
        # Append to history
        self.history.time_points.extend(times)
        self.history.thesis_states.extend(theses)
        self.history.antithesis_states.extend(antitheses)
        self.history.synthesis_norms.extend([synthesis_norm] * len(times))
        self.history.conflict_intensities.extend([conflict_intensity] * len(times))
        
        # Trim history if exceeds capacity
        if len(self.history.time_points) > self.history_capacity:
            excess = len(self.history.time_points) - self.history_capacity
            self.history.time_points = self.history.time_points[excess:]
            self.history.thesis_states = self.history.thesis_states[excess:]
            self.history.antithesis_states = self.history.antithesis_states[excess:]
            self.history.synthesis_norms = self.history.synthesis_norms[excess:]
            self.history.conflict_intensities = self.history.conflict_intensities[excess:]
    
    def evolve_multiple_epochs(self, 
                               n_epochs: int = 5,
                               time_per_epoch: float = 5.0,
                               initial_state: Optional[np.ndarray] = None) -> Dict[str, List]:
        """
        Evolve conflict through multiple epochs with possible phase transitions.
        
        Parameters:
        -----------
        n_epochs : int
            Number of epochs to simulate
        time_per_epoch : float
            Time duration per epoch
        initial_state : np.ndarray, optional
            Initial state
        
        Returns:
        --------
        Dict with epoch-by-epoch results
        """
        if initial_state is None:
            current_state = np.concatenate([self.current_thesis, self.current_antithesis])
        else:
            current_state = initial_state.copy()
        
        epoch_results = {
            'final_states': [],
            'phase_transitions': [],
            'synthesis_norms': [],
            'conflict_intensities': [],
            'transition_times': []
        }
        
        total_time = 0
        for epoch in range(n_epochs):
            time_span = (total_time, total_time + time_per_epoch)
            
            result = self.evolve_conflict(
                time_span=time_span,
                initial_state=current_state,
                record_history=True
            )
            
            # Store results
            epoch_results['final_states'].append(result['final_state'])
            epoch_results['phase_transitions'].append(result['phase_transition'])
            epoch_results['synthesis_norms'].append(result['synthesis_norm'])
            epoch_results['conflict_intensities'].append(result['conflict_intensity'])
            
            if result['phase_transition']:
                epoch_results['transition_times'].append(time_span[1])
            
            # Update for next epoch
            current_state = result['final_state']
            total_time += time_per_epoch
            
            # Adjust parameters after phase transition (dialectical progression)
            if result['phase_transition']:
                self._adjust_after_transition()
        
        return epoch_results
    
    def _adjust_after_transition(self):
        """Adjust parameters after a phase transition (dialectical progression)"""
        # Reduce competition after resolution
        self.competition_strength *= 0.8
        
        # Increase cooperation
        self.cooperation_factor = min(0.5, self.cooperation_factor * 1.2)
        
        # Adjust growth rate
        self.growth_rate *= 1.05
        
        # Increase threshold for next transition
        self.phase_transition_threshold *= 1.1
    
    def get_conflict_metrics(self) -> Dict[str, float]:
        """Calculate various metrics about the current conflict state"""
        thesis_norm = np.linalg.norm(self.current_thesis)
        antithesis_norm = np.linalg.norm(self.current_antithesis)
        difference_norm = np.linalg.norm(self.current_thesis - self.current_antithesis)
        
        # Calculate synthesis
        synthesis = self._calculate_synthesis(self.current_thesis, self.current_antithesis)
        synthesis_norm = np.linalg.norm(synthesis)
        
        # Calculate asymmetry
        asymmetry = np.abs(thesis_norm - antithesis_norm) / max(thesis_norm, antithesis_norm)
        
        # Calculate correlation
        if thesis_norm > 0 and antithesis_norm > 0:
            correlation = np.dot(self.current_thesis, self.current_antithesis) / \
                         (thesis_norm * antithesis_norm)
        else:
            correlation = 0.0
        
        return {
            'thesis_norm': thesis_norm,
            'antithesis_norm': antithesis_norm,
            'difference_norm': difference_norm,
            'synthesis_norm': synthesis_norm,
            'asymmetry': asymmetry,
            'correlation': correlation,
            'phase_transition_count': self.phase_transition_count,
            'conflict_intensity': difference_norm / (thesis_norm + antithesis_norm + 1e-10),
            'stability_index': synthesis_norm / (difference_norm + 1e-10)
        }
    
    def visualize_conflict_dynamics(self, 
                                    save_path: Optional[str] = None,
                                    figsize: Tuple[int, int] = (15, 10)):
        """
        Create comprehensive visualization of conflict dynamics.
        
        Parameters:
        -----------
        save_path : str, optional
            Path to save the figure
        figsize : Tuple[int, int]
            Figure size
        """
        if len(self.history.time_points) == 0:
            print("No history data available for visualization")
            return
        
        # Convert history to arrays for plotting
        times = np.array(self.history.time_points)
        
        # Extract first component of thesis and antithesis for plotting
        if self.dimension >= 1:
            thesis_component = np.array([t[0] for t in self.history.thesis_states])
            antithesis_component = np.array([a[0] for a in self.history.antithesis_states])
        
        # Create figure
        fig = plt.figure(figsize=figsize)
        
        # 1. Time series of thesis and antithesis
        ax1 = plt.subplot(2, 3, 1)
        ax1.plot(times, thesis_component, 'b-', label='Thesis (Component 1)', linewidth=2)
        ax1.plot(times, antithesis_component, 'r-', label='Antithesis (Component 1)', linewidth=2)
        
        # Mark phase transitions
        for transition in self.history.phase_transitions:
            ax1.axvline(transition['time'], color='g', linestyle='--', alpha=0.5, 
                       label='Phase Transition' if transition == self.history.phase_transitions[0] else "")
        
        ax1.set_title(f'{self.conflict_type.value.replace("_", " ").title()} Dynamics')
        ax1.set_xlabel('Time')
        ax1.set_ylabel('State Value')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Phase portrait (first two dimensions)
        if self.dimension >= 2:
            ax2 = plt.subplot(2, 3, 2)
            thesis_comp1 = np.array([t[0] for t in self.history.thesis_states[-100:]])
            thesis_comp2 = np.array([t[1] for t in self.history.thesis_states[-100:]])
            antithesis_comp1 = np.array([a[0] for a in self.history.antithesis_states[-100:]])
            antithesis_comp2 = np.array([a[1] for a in self.history.antithesis_states[-100:]])
            
            ax2.scatter(thesis_comp1, thesis_comp2, c='blue', alpha=0.6, 
                       s=20, label='Thesis Trajectory')
            ax2.scatter(antithesis_comp1, antithesis_comp2, c='red', alpha=0.6, 
                       s=20, label='Antithesis Trajectory')
            
            # Draw lines connecting points
            ax2.plot(thesis_comp1, thesis_comp2, 'b-', alpha=0.3, linewidth=0.5)
            ax2.plot(antithesis_comp1, antithesis_comp2, 'r-', alpha=0.3, linewidth=0.5)
            
            ax2.set_title('Phase Portrait (Components 1 & 2)')
            ax2.set_xlabel('Component 1')
            ax2.set_ylabel('Component 2')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            ax2.axis('equal')
        
        # 3. Conflict intensity over time
        ax3 = plt.subplot(2, 3, 3)
        conflict_intensities = np.array(self.history.conflict_intensities)
        ax3.plot(times, conflict_intensities, 'purple', linewidth=2)
        ax3.axhline(self.phase_transition_threshold, color='orange', 
                   linestyle='--', label='Transition Threshold')
        ax3.set_title('Conflict Intensity Over Time')
        ax3.set_xlabel('Time')
        ax3.set_ylabel('Conflict Intensity')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Synthesis norms over time
        ax4 = plt.subplot(2, 3, 4)
        synthesis_norms = np.array(self.history.synthesis_norms)
        ax4.plot(times, synthesis_norms, 'green', linewidth=2)
        ax4.set_title('Synthesis Norm Evolution')
        ax4.set_xlabel('Time')
        ax4.set_ylabel('||Synthesis||')
        ax4.grid(True, alpha=0.3)
        
        # 5. 3D trajectory if dimension >= 3
        if self.dimension >= 3:
            ax5 = plt.subplot(2, 3, 5, projection='3d')
            thesis_3d = np.array(self.history.thesis_states[-200:])
            antithesis_3d = np.array(self.history.antithesis_states[-200:])
            
            ax5.plot(thesis_3d[:, 0], thesis_3d[:, 1], thesis_3d[:, 2], 
                    'b-', alpha=0.6, linewidth=1, label='Thesis')
            ax5.plot(antithesis_3d[:, 0], antithesis_3d[:, 1], antithesis_3d[:, 2],
                    'r-', alpha=0.6, linewidth=1, label='Antithesis')
            
            ax5.set_title('3D State Space Trajectory')
            ax5.set_xlabel('Component 1')
            ax5.set_ylabel('Component 2')
            ax5.set_zlabel('Component 3')
            ax5.legend()
        
        # 6. Current state metrics
        ax6 = plt.subplot(2, 3, 6)
        metrics = self.get_conflict_metrics()
        
        metric_names = [
            'Thesis Norm', 'Antithesis Norm', 
            'Conflict Intensity', 'Synthesis Norm',
            'Phase Transitions', 'Stability Index'
        ]
        metric_values = [
            metrics['thesis_norm'],
            metrics['antithesis_norm'],
            metrics['conflict_intensity'],
            metrics['synthesis_norm'],
            metrics['phase_transition_count'],
            metrics['stability_index']
        ]
        
        colors = ['blue', 'red', 'purple', 'green', 'orange', 'brown']
        bars = ax6.bar(metric_names, metric_values, color=colors, alpha=0.7)
        
        ax6.set_title('Current State Metrics')
        ax6.set_ylabel('Value')
        ax6.set_xticklabels(metric_names, rotation=45, ha='right')
        
        # Add value labels on bars
        for bar, value in zip(bars, metric_values):
            height = bar.get_height()
            ax6.text(bar.get_x() + bar.get_width()/2., height,
                    f'{value:.3f}', ha='center', va='bottom')
        
        plt.suptitle(f'Ontological Conflict Analysis: {self.conflict_type.value}', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Visualization saved to {save_path}")
        
        plt.show()
        
        return fig
    
    def reset_history(self):
        """Reset conflict history while keeping current state"""
        self.history = ConflictHistory()
        print("📊 Conflict history reset")
    
    def set_state(self, thesis: np.ndarray, antithesis: np.ndarray):
        """
        Set the current thesis and antithesis states.
        
        Parameters:
        -----------
        thesis, antithesis : np.ndarray
            New state vectors
        """
        if len(thesis) != self.dimension or len(antithesis) != self.dimension:
            raise ValueError(
                f"State vectors must have dimension {self.dimension}"
            )
        
        self.current_thesis = thesis.copy()
        self.current_antithesis = antithesis.copy()
        print(f"🔄 State set: thesis_norm={np.linalg.norm(thesis):.3f}, "
              f"antithesis_norm={np.linalg.norm(antithesis):.3f}")
    
    def __repr__(self) -> str:
        return (f"XenopoulosOntologicalConflict(dimension={self.dimension}, "
                f"type={self.conflict_type.value}, "
                f"transitions={self.phase_transition_count})")


# Utility functions for conflict analysis
def analyze_conflict_bifurcation(conflict_system: XenopoulosOntologicalConflict,
                                 parameter_name: str,
                                 parameter_range: Tuple[float, float],
                                 n_points: int = 20) -> Dict[str, np.ndarray]:
    """
    Analyze how conflict dynamics change with parameter variations.
    
    Parameters:
    -----------
    conflict_system : XenopoulosOntologicalConflict
        Conflict system to analyze
    parameter_name : str
        Parameter to vary: 'growth_rate', 'competition_strength', 'cooperation_factor'
    parameter_range : Tuple[float, float]
        Range of parameter values
    n_points : int
        Number of parameter points
    
    Returns:
    --------
    Dict with bifurcation analysis results
    """
    parameter_values = np.linspace(parameter_range[0], parameter_range[1], n_points)
    
    results = {
        'parameter_values': parameter_values,
        'final_thesis_norms': [],
        'final_antithesis_norms': [],
        'synthesis_norms': [],
        'phase_transitions': []
    }
    
    original_value = getattr(conflict_system, parameter_name)
    
    for param_value in parameter_values:
        # Set parameter
        setattr(conflict_system, parameter_name, param_value)
        
        # Evolve conflict
        result = conflict_system.evolve_conflict(
            time_span=(0, 10),
            record_history=False
        )
        
        # Record results
        final_state = result['final_state']
        thesis = final_state[:conflict_system.dimension]
        antithesis = final_state[conflict_system.dimension:]
        
        results['final_thesis_norms'].append(np.linalg.norm(thesis))
        results['final_antithesis_norms'].append(np.linalg.norm(antithesis))
        results['synthesis_norms'].append(result['synthesis_norm'])
        results['phase_transitions'].append(result['phase_transition'])
    
    # Restore original parameter
    setattr(conflict_system, parameter_name, original_value)
    
    return results


def compare_conflict_types(dimension: int = 3, 
                          time_span: Tuple[float, float] = (0, 10)) -> Dict:
    """
    Compare dynamics of different conflict types.
    """
    results = {}
    
    for conflict_type in ConflictType:
        print(f"Analyzing {conflict_type.value}...")
        
        system = XenopoulosOntologicalConflict(
            dimension=dimension,
            conflict_type=conflict_type
        )
        
        result = system.evolve_conflict(time_span=time_span)
        
        results[conflict_type.value] = {
            'final_synthesis_norm': result['synthesis_norm'],
            'conflict_intensity': result['conflict_intensity'],
            'phase_transition': result['phase_transition'],
            'metrics': system.get_conflict_metrics()
        }
    
    return results


# Example usage for Visual Studio Code
if __name__ == "__main__":
    print("=" * 70)
    print("XENOPOULOS ONTOLOGICAL CONFLICT - VISUAL STUDIO CODE DEMO")
    print("=" * 70)
    
    # Create conflict system
    print("\n1. Creating Master-Slave dialectical conflict system...")
    conflict = XenopoulosOntologicalConflict(
        dimension=3,
        conflict_type=ConflictType.MASTER_SLAVE,
        growth_rate=1.2,
        competition_strength=0.6,
        cooperation_factor=0.05
    )
    
    # Set initial state (master strong, slave weak)
    print("\n2. Setting initial state (Hegelian master-slave relation)...")
    thesis = np.array([1.0, 0.5, -0.5])  # Master position
    antithesis = np.array([-0.5, 0.5, 0.3])  # Slave position
    conflict.set_state(thesis, antithesis)
    
    # Evolve conflict
    print("\n3. Evolving conflict over time...")
    result = conflict.evolve_conflict(time_span=(0, 15), n_steps=200)
    
    print(f"   Final synthesis norm: {result['synthesis_norm']:.4f}")
    print(f"   Conflict intensity: {result['conflict_intensity']:.4f}")
    print(f"   Phase transition occurred: {result['phase_transition']}")
    
    # Show metrics
    print("\n4. Current conflict metrics:")
    metrics = conflict.get_conflict_metrics()
    for name, value in metrics.items():
        print(f"   {name}: {value:.4f}")
    
    # Multiple epochs with possible transitions
    print("\n5. Simulating multiple epochs...")
    epoch_results = conflict.evolve_multiple_epochs(
        n_epochs=4,
        time_per_epoch=8.0
    )
    
    print(f"   Total phase transitions: {conflict.phase_transition_count}")
    print(f"   Final epoch synthesis norms: {epoch_results['synthesis_norms']}")
    
    # Bifurcation analysis
    print("\n6. Performing bifurcation analysis (varying competition strength)...")
    bifurcation = analyze_conflict_bifurcation(
        conflict,
        parameter_name='competition_strength',
        parameter_range=(0.1, 1.0),
        n_points=10
    )
    
    print(f"   Parameter range: {bifurcation['parameter_values'][0]:.2f} to {bifurcation['parameter_values'][-1]:.2f}")
    print(f"   Average synthesis: {np.mean(bifurcation['synthesis_norms']):.4f}")
    
    # Compare different conflict types
    print("\n7. Comparing different conflict types...")
    comparison = compare_conflict_types(dimension=2, time_span=(0, 8))
    
    for ctype, data in comparison.items():
        print(f"   {ctype}: synthesis={data['final_synthesis_norm']:.3f}, "
              f"transition={data['phase_transition']}")
    
    # Visualize
    print("\n8. Generating comprehensive visualization...")
    conflict.visualize_conflict_dynamics(
        save_path="ontological_conflict_analysis.png",
        figsize=(16, 10)
    )
    
    print("\n" + "=" * 70)
    print("DEMO COMPLETE - Ready for use in Visual Studio Code!")
    print("=" * 70)

    