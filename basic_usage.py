"""
basic_usage.py - Basic usage examples for Xenopoulos Dialectical System
Complete implementation with practical examples
"""

import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Import the main system
try:
    from deepseek_python_20260105_ddd90d import (
        XenopoulosKlein4Group,
        XenopoulosDialecticalDynamics,
        XenopoulosOntologicalConflict,
        XenopoulosFourthStructure
    )
    print("✅ Successfully imported Xenopoulos system")
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Please ensure deepseek_python_20260105_ddd90d.py is in the same directory")
    exit(1)

def example_1_quick_start():
    """
    Example 1: Quick Start - Minimal working example
    """
    print("\n" + "="*80)
    print("EXAMPLE 1: QUICK START")
    print("="*80)
    
    # 1. Initialize the system
    print("\n1. Initializing Xenopoulos Fourth Logical Structure...")
    system = XenopoulosFourthStructure(
        dimension=3,              # State space dimension
        chaos_factor=0.03,        # Chaos injection factor
        qualitative_threshold=0.8, # Threshold for qualitative transitions
        history_depth=3           # Historical context depth
    )
    
    print(f"   • Dimension: {system.dimension}")
    print(f"   • Initial Thesis: {system.thesis}")
    print(f"   • Initial Antithesis: {system.antithesis}")
    
    # 2. Evolve the system
    print("\n2. Evolving system for 50 epochs...")
    synthesis_history, transitions = system.evolve_system(
        epochs=50,               # Number of evolution steps
        verbose=False            # Don't show detailed progress
    )
    
    # 3. Display results
    print("\n3. Results:")
    print(f"   • Total syntheses: {len(synthesis_history)}")
    print(f"   • Qualitative transitions: {len(transitions)}")
    print(f"   • Final synthesis: {synthesis_history[-1].round(4)}")
    
    if transitions:
        print(f"   • First transition at epoch: {transitions[0]['epoch']}")
        print(f"   • First transition magnitude: {transitions[0]['synthesis_norm']:.4f}")
    
    return system, synthesis_history, transitions

def example_2_individual_components():
    """
    Example 2: Working with individual components
    """
    print("\n" + "="*80)
    print("EXAMPLE 2: INDIVIDUAL COMPONENTS")
    print("="*80)
    
    # 1. Klein-4 Group (INRC Operators)
    print("\n1. Klein-4 Group Operations:")
    group = XenopoulosKlein4Group(dimension=3)
    
    # Test vector
    test_vector = np.array([1.0, 2.0, 3.0])
    print(f"   Test vector: {test_vector}")
    
    # Apply all operators
    identity = group.apply_operator(test_vector, 'I')
    negation = group.apply_operator(test_vector, 'N')
    reciprocity = group.apply_operator(test_vector, 'R')
    correlation = group.apply_operator(test_vector, 'C')
    
    print(f"   I(x) = {identity.round(4)}")
    print(f"   N(x) = {negation.round(4)}")
    print(f"   R(x) = {reciprocity.round(4)} (cyclic permutation)")
    print(f"   C(x) = {correlation.round(4)} (N∘R = R∘N)")
    
    # 2. Dialectical Dynamics
    print("\n2. Dialectical Dynamics (D₁ and D₂):")
    dynamics = XenopoulosDialecticalDynamics(
        input_dim=3,
        hidden_dim=16,
        qualitative_threshold=0.8
    )
    
    # Create thesis and antithesis
    thesis = np.array([1.0, 0.5, -0.5])
    antithesis = np.array([-0.5, 0.5, 1.0])
    
    print(f"   Thesis: {thesis}")
    print(f"   Antithesis: {antithesis}")
    
    # Perform synthesis using both formalisms
    thesis_tensor = np.array([thesis])
    antithesis_tensor = np.array([antithesis])
    
    result_D1 = dynamics.forward(thesis_tensor, antithesis_tensor, mode='D1')
    result_D2 = dynamics.forward(thesis_tensor, antithesis_tensor, mode='D2')
    
    print(f"   D₁ Synthesis norm: {result_D1['synthesis_norm']:.4f}")
    print(f"   D₂ Synthesis norm: {result_D2['synthesis_norm']:.4f}")
    print(f"   D₁ Result shape: {result_D1['synthesis'].shape}")
    
    # 3. Ontological Conflict
    print("\n3. Ontological Conflict Dynamics:")
    conflict = XenopoulosOntologicalConflict(dimension=3)
    
    initial_state = np.concatenate([thesis, antithesis])
    print(f"   Initial state shape: {initial_state.shape}")
    
    final_state, phase_transition = conflict.evolve_conflict(
        initial_state,
        time_span=(0, 5)  # Evolve from time 0 to 5
    )
    
    print(f"   Phase transition occurred: {phase_transition}")
    print(f"   Final state (first 3 values): {final_state[:3].round(4)}...")
    
    return group, dynamics, conflict

def example_3_complete_workflow():
    """
    Example 3: Complete workflow with visualization
    """
    print("\n" + "="*80)
    print("EXAMPLE 3: COMPLETE WORKFLOW")
    print("="*80)
    
    # Initialize system
    print("\n1. Initializing system...")
    system = XenopoulosFourthStructure(dimension=3)
    
    # Single steps demonstration
    print("\n2. Demonstrating single steps:")
    
    # Single dialectical step
    print("   • Performing single dialectical step...")
    synthesis = system.dialectical_step(include_chaos=True)
    print(f"     Synthesis: {synthesis.round(4)}")
    print(f"     Synthesis norm: {np.linalg.norm(synthesis):.4f}")
    
    # Single ontology evolution
    print("   • Evolving ontology...")
    transition = system.evolve_ontology()
    print(f"     Phase transition occurred: {transition}")
    print(f"     Updated thesis: {system.thesis.round(4)}")
    
    # Complete evolution
    print("\n3. Running complete evolution (100 epochs)...")
    synthesis_history, transitions = system.evolve_system(
        epochs=100,
        verbose=False
    )
    
    # Analysis
    print("\n4. Analysis:")
    norms = [np.linalg.norm(s) for s in synthesis_history]
    
    print(f"   • Total epochs: {system.epoch}")
    print(f"   • Final synthesis norm: {norms[-1]:.4f}")
    print(f"   • Mean synthesis norm: {np.mean(norms):.4f}")
    print(f"   • Max synthesis norm: {np.max(norms):.4f}")
    print(f"   • Qualitative transitions: {len(transitions)}")
    
    if transitions:
        transition_epochs = [t['epoch'] for t in transitions]
        print(f"   • Transition epochs: {transition_epochs}")
    
    # Mode usage
    if hasattr(system, 'mode_history'):
        mode_counts = {'D1': 0, 'D2': 0}
        for mode in system.mode_history:
            mode_counts[mode] += 1
        
        print(f"   • D1 mode usage: {mode_counts['D1']} ({mode_counts['D1']/len(system.mode_history)*100:.1f}%)")
        print(f"   • D2 mode usage: {mode_counts['D2']} ({mode_counts['D2']/len(system.mode_history)*100:.1f}%)")
    
    return system, synthesis_history, transitions

def example_4_parameter_exploration():
    """
    Example 4: Exploring different parameters
    """
    print("\n" + "="*80)
    print("EXAMPLE 4: PARAMETER EXPLORATION")
    print("="*80)
    
    # Test different chaos factors
    chaos_factors = [0.01, 0.03, 0.05, 0.1]
    results = []
    
    for chaos_factor in chaos_factors:
        print(f"\nTesting chaos_factor = {chaos_factor}")
        
        system = XenopoulosFourthStructure(
            dimension=3,
            chaos_factor=chaos_factor,
            qualitative_threshold=0.8
        )
        
        synthesis_history, transitions = system.evolve_system(
            epochs=50,
            verbose=False
        )
        
        norms = [np.linalg.norm(s) for s in synthesis_history]
        
        results.append({
            'chaos_factor': chaos_factor,
            'mean_norm': np.mean(norms),
            'std_norm': np.std(norms),
            'transitions': len(transitions),
            'final_norm': norms[-1]
        })
        
        print(f"  • Mean norm: {np.mean(norms):.4f}")
        print(f"  • Transitions: {len(transitions)}")
    
    # Display summary
    print("\n" + "-"*40)
    print("SUMMARY OF PARAMETER EXPLORATION:")
    print("-"*40)
    
    for result in results:
        print(f"chaos_factor={result['chaos_factor']}: "
              f"mean={result['mean_norm']:.4f}, "
              f"transitions={result['transitions']}")
    
    return results

def example_5_dialectical_cycles():
    """
    Example 5: Dialectical cycles and historical context
    """
    print("\n" + "="*80)
    print("EXAMPLE 5: DIALECTICAL CYCLES")
    print("="*80)
    
    # Initialize dialectical dynamics
    dynamics = XenopoulosDialecticalDynamics(
        input_dim=3,
        hidden_dim=16,
        qualitative_threshold=0.7
    )
    
    # Create initial thesis and antithesis
    thesis = np.array([1.0, 0.0, 0.0])
    antithesis = np.array([0.0, 1.0, 0.0])
    
    print(f"Initial thesis: {thesis}")
    print(f"Initial antithesis: {antithesis}")
    
    # Run dialectical cycle
    print("\nRunning dialectical cycle (10 steps, D1 mode)...")
    history = dynamics.dialectical_cycle(
        thesis=thesis,
        antithesis=antithesis,
        steps=10,
        mode='D1'
    )
    
    # Analyze results
    print(f"\nCycle completed with {len(history['synthesis'])} syntheses")
    print(f"Synthesis norms: {[round(n, 4) for n in history['synthesis_norms']]}")
    
    # Count qualitative transitions
    transitions = sum(history['qualitative_transitions'])
    print(f"Qualitative transitions during cycle: {transitions}")
    
    # Show evolution
    print("\nEvolution of thesis:")
    for i, t in enumerate(history['thesis']):
        print(f"  Step {i}: {t.round(4)}")
    
    return dynamics, history

def example_6_visualization_demo():
    """
    Example 6: Visualization capabilities
    """
    print("\n" + "="*80)
    print("EXAMPLE 6: VISUALIZATION DEMO")
    print("="*80)
    
    # Create a 2D system for easier visualization
    print("\n1. Creating 2D system for visualization...")
    system_2d = XenopoulosFourthStructure(dimension=2)
    
    # Evolve for visualization
    print("2. Evolving system for visualization...")
    synthesis_history, transitions = system_2d.evolve_system(
        epochs=100,
        verbose=False
    )
    
    # Create simple visualizations
    print("3. Creating visualizations...")
    
    # Plot 1: Synthesis norms over time
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    norms = [np.linalg.norm(s) for s in synthesis_history]
    plt.plot(norms, 'b-', alpha=0.7, linewidth=1)
    plt.axhline(system_2d.qualitative_threshold, color='r', 
                linestyle='--', alpha=0.7, label='Threshold')
    plt.title('Synthesis Norm Evolution')
    plt.xlabel('Epoch')
    plt.ylabel('||Synthesis||')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Phase space (2D)
    plt.subplot(1, 3, 2)
    synthesis_array = np.array(synthesis_history)
    plt.plot(synthesis_array[:, 0], synthesis_array[:, 1], 'b-', alpha=0.5)
    plt.scatter(synthesis_array[:, 0], synthesis_array[:, 1], 
                c=range(len(synthesis_array)), cmap='viridis', s=10)
    plt.title('Phase Space Trajectory')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.colorbar(label='Time')
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Component evolution
    plt.subplot(1, 3, 3)
    plt.plot(synthesis_array[:, 0], label='Component 1', alpha=0.7)
    plt.plot(synthesis_array[:, 1], label='Component 2', alpha=0.7)
    plt.title('Component Evolution')
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('basic_visualization.png', dpi=150)
    print(f"4. Visualization saved as 'basic_visualization.png'")
    
    plt.show()
    
    return system_2d, synthesis_history

def example_7_custom_application():
    """
    Example 7: Custom application - Conceptual evolution
    """
    print("\n" + "="*80)
    print("EXAMPLE 7: CUSTOM APPLICATION - CONCEPTUAL EVOLUTION")
    print("="*80)
    
    # Define concepts as vectors
    concepts = {
        'freedom': np.array([1.0, 0.5, 0.2]),
        'equality': np.array([0.8, 1.0, 0.3]),
        'justice': np.array([0.6, 0.7, 1.0])
    }
    
    print("Initial concepts:")
    for name, vector in concepts.items():
        print(f"  {name}: {vector.round(3)}")
    
    # Use one concept as thesis, another as antithesis
    thesis = concepts['freedom']
    antithesis = concepts['equality']
    
    print(f"\nDialectical synthesis of 'freedom' and 'equality':")
    
    # Create system
    system = XenopoulosFourthStructure(dimension=3)
    system.thesis = thesis
    system.antithesis = antithesis
    
    # Perform synthesis
    synthesis = system.dialectical_step(include_chaos=False)
    print(f"  Synthesis: {synthesis.round(4)}")
    print(f"  Norm: {np.linalg.norm(synthesis):.4f}")
    
    # Evolve further
    print("\nEvolving concept synthesis (20 steps)...")
    history = []
    for i in range(20):
        synthesis = system.dialectical_step(include_chaos=True)
        history.append(synthesis.copy())
        
        if i % 5 == 0:
            print(f"  Step {i}: norm = {np.linalg.norm(synthesis):.4f}")
    
    print(f"\nFinal synthesized concept: {history[-1].round(4)}")
    
    return concepts, history

def run_all_examples():
    """
    Run all examples sequentially
    """
    print("="*80)
    print("XENOPOULOS DIALECTICAL SYSTEM - BASIC USAGE EXAMPLES")
    print("="*80)
    
    all_results = {}
    
    try:
        # Run example 1
        system1, history1, transitions1 = example_1_quick_start()
        all_results['example1'] = {
            'system': system1,
            'history': history1,
            'transitions': transitions1
        }
        
        # Run example 2
        group, dynamics, conflict = example_2_individual_components()
        all_results['example2'] = {
            'group': group,
            'dynamics': dynamics,
            'conflict': conflict
        }
        
        # Run example 3
        system3, history3, transitions3 = example_3_complete_workflow()
        all_results['example3'] = {
            'system': system3,
            'history': history3,
            'transitions': transitions3
        }
        
        # Run example 4
        param_results = example_4_parameter_exploration()
        all_results['example4'] = param_results
        
        # Run example 5
        dynamics5, history5 = example_5_dialectical_cycles()
        all_results['example5'] = {
            'dynamics': dynamics5,
            'history': history5
        }
        
        # Run example 7 (skip 6 if you don't want visualizations)
        concepts, concept_history = example_7_custom_application()
        all_results['example7'] = {
            'concepts': concepts,
            'history': concept_history
        }
        
        print("\n" + "="*80)
        print("ALL EXAMPLES COMPLETED SUCCESSFULLY!")
        print("="*80)
        
        # Summary
        total_transitions = 0
        if 'example1' in all_results:
            total_transitions += len(all_results['example1']['transitions'])
        if 'example3' in all_results:
            total_transitions += len(all_results['example3']['transitions'])
        
        print(f"\nSummary:")
        print(f"• Ran 6 different examples")
        print(f"• Total qualitative transitions observed: {total_transitions}")
        print(f"• Systems created: 4")
        print(f"• Visualizations: 1 saved to file")
        
        # Save results for further analysis
        import pickle
        with open('basic_usage_results.pkl', 'wb') as f:
            pickle.dump(all_results, f)
        print(f"• Results saved to 'basic_usage_results.pkl'")
        
    except Exception as e:
        print(f"\n❌ Error running examples: {e}")
        import traceback
        traceback.print_exc()
    
    return all_results

def interactive_example():
    """
    Interactive example for users to experiment
    """
    print("\n" + "="*80)
    print("INTERACTIVE EXAMPLE")
    print("="*80)
    
    print("\nCreate your own dialectical system:")
    
    # Get user parameters
    try:
        dimension = int(input("Enter dimension (2-10, default 3): ") or "3")
        epochs = int(input("Enter number of epochs (10-500, default 50): ") or "50")
        chaos = float(input("Enter chaos factor (0.0-0.2, default 0.03): ") or "0.03")
        
        # Create system
        system = XenopoulosFourthStructure(
            dimension=max(2, min(10, dimension)),
            chaos_factor=max(0.0, min(0.2, chaos)),
            qualitative_threshold=0.8
        )
        
        print(f"\nSystem created with:")
        print(f"• Dimension: {system.dimension}")
        print(f"• Chaos factor: {system.chaos_factor}")
        print(f"• Initial thesis: {system.thesis.round(3)}")
        
        # Evolve
        print(f"\nEvolving for {epochs} epochs...")
        history, transitions = system.evolve_system(
            epochs=min(500, max(10, epochs)),
            verbose=False
        )
        
        # Results
        norms = [np.linalg.norm(s) for s in history]
        print(f"\nResults:")
        print(f"• Final synthesis: {history[-1].round(4)}")
        print(f"• Final norm: {norms[-1]:.4f}")
        print(f"• Qualitative transitions: {len(transitions)}")
        print(f"• Mean norm: {np.mean(norms):.4f}")
        
        if transitions:
            print(f"• First transition at epoch: {transitions[0]['epoch']}")
        
        # Simple visualization
        visualize = input("\nCreate simple visualization? (y/n): ").lower()
        if visualize == 'y':
            plt.figure(figsize=(10, 4))
            
            plt.subplot(1, 2, 1)
            plt.plot(norms, 'b-')
            plt.axhline(system.qualitative_threshold, color='r', linestyle='--')
            plt.title('Synthesis Norm')
            plt.xlabel('Epoch')
            plt.ylabel('Norm')
            plt.grid(True, alpha=0.3)
            
            if system.dimension >= 2:
                plt.subplot(1, 2, 2)
                history_array = np.array(history)
                plt.plot(history_array[:, 0], history_array[:, 1], 'b-', alpha=0.5)
                plt.title('Phase Space (Components 1 & 2)')
                plt.xlabel('Component 1')
                plt.ylabel('Component 2')
                plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.show()
        
        return system, history, transitions
        
    except ValueError as e:
        print(f"Invalid input: {e}")
        return None

# Main execution
if __name__ == "__main__":
    print("Xenopoulos Dialectical System - Basic Usage Examples")
    print("Choose an option:")
    print("1. Run all examples sequentially")
    print("2. Run interactive example")
    print("3. Run specific example (1-7)")
    
    try:
        choice = input("\nEnter your choice (1-3): ").strip()
        
        if choice == '1':
            results = run_all_examples()
        elif choice == '2':
            interactive_result = interactive_example()
        elif choice == '3':
            example_num = input("Enter example number (1-7): ").strip()
            if example_num == '1':
                example_1_quick_start()
            elif example_num == '2':
                example_2_individual_components()
            elif example_num == '3':
                example_3_complete_workflow()
            elif example_num == '4':
                example_4_parameter_exploration()
            elif example_num == '5':
                example_5_dialectical_cycles()
            elif example_num == '6':
                example_6_visualization_demo()
            elif example_num == '7':
                example_7_custom_application()
            else:
                print("Invalid example number")
        else:
            print("Invalid choice")
    
    except KeyboardInterrupt:
        print("\n\nProgram interrupted by user")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*80)
    print("BASIC USAGE EXAMPLES COMPLETE")
    print("="*80)