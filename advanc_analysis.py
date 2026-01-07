"""
advanced_analysis.py
Advanced analysis tools for Xenopoulos' Fourth Logical Structure system.
Statistical analysis, phase space exploration, and deep system diagnostics.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy import stats, signal, fft
import warnings
warnings.filterwarnings('ignore')

# Import the main system
from deepseek_python_20260105_ddd90d import XenopoulosFourthStructure

class XenopoulosAdvancedAnalyzer:
    """Advanced analysis tools for Xenopoulos dialectical system."""
    
    def __init__(self, dimension=3):
        """
        Initialize analyzer with system dimension.
        
        Parameters:
        -----------
        dimension : int
            Dimension of the state space (default: 3)
        """
        self.dimension = dimension
        self.system = XenopoulosFourthStructure(dimension=dimension)
        
    def run_comprehensive_analysis(self, epochs=1000, verbose=True):
        """
        Run comprehensive analysis including all metrics.
        
        Parameters:
        -----------
        epochs : int
            Number of epochs to evolve system
        verbose : bool
            Whether to print progress
            
        Returns:
        --------
        dict : Comprehensive analysis results
        """
        if verbose:
            print("=" * 80)
            print("ADVANCED XENOPOULOS SYSTEM ANALYSIS")
            print("=" * 80)
            print(f"Dimension: {self.dimension}")
            print(f"Epochs: {epochs}")
            print("-" * 80)
        
        # Evolve system
        if verbose:
            print("Evolving system...")
        
        synthesis_history, transitions = self.system.evolve_system(
            epochs=epochs, verbose=verbose
        )
        
        # Convert to numpy arrays
        synthesis_array = np.array(synthesis_history)
        
        # Calculate all metrics
        results = {
            'synthesis_history': synthesis_history,
            'transitions': transitions,
            'synthesis_array': synthesis_array,
            'dimension': self.dimension,
            'epochs': epochs
        }
        
        # Add various analyses
        results.update(self._calculate_basic_statistics(synthesis_array))
        results.update(self._calculate_phase_space_metrics(synthesis_array))
        results.update(self._calculate_dialectical_metrics(synthesis_array))
        results.update(self._calculate_spectral_analysis(synthesis_array))
        results.update(self._calculate_attractor_analysis(synthesis_array))
        results.update(self._calculate_transition_analysis(transitions))
        
        # Add dialectical entropy and complexity measures
        results.update(self._calculate_complexity_measures(synthesis_array))
        
        if verbose:
            self._print_summary(results)
            
        return results
    
    def _calculate_basic_statistics(self, synthesis_array):
        """Calculate basic statistical metrics."""
        norms = np.linalg.norm(synthesis_array, axis=1)
        
        stats_dict = {
            'mean_norm': float(np.mean(norms)),
            'std_norm': float(np.std(norms)),
            'min_norm': float(np.min(norms)),
            'max_norm': float(np.max(norms)),
            'median_norm': float(np.median(norms)),
            'skewness_norm': float(stats.skew(norms)),
            'kurtosis_norm': float(stats.kurtosis(norms)),
            
            'component_means': np.mean(synthesis_array, axis=0).tolist(),
            'component_stds': np.std(synthesis_array, axis=0).tolist(),
            'component_mins': np.min(synthesis_array, axis=0).tolist(),
            'component_maxs': np.max(synthesis_array, axis=0).tolist(),
        }
        
        # Add percentiles for better distribution understanding
        for p in [10, 25, 50, 75, 90]:
            stats_dict[f'percentile_{p}'] = float(np.percentile(norms, p))
            
        return stats_dict
    
    def _calculate_phase_space_metrics(self, synthesis_array):
        """Calculate phase space and dynamical metrics."""
        if len(synthesis_array) < 10:
            return {}
        
        metrics = {}
        
        # Lyapunov exponent estimation
        metrics['lyapunov_estimate'] = float(self._estimate_lyapunov(synthesis_array))
        
        # Correlation dimension estimation
        metrics['correlation_dimension'] = float(self._estimate_correlation_dimension(synthesis_array))
        
        # Recurrence analysis
        metrics['recurrence_rate'] = float(self._calculate_recurrence_rate(synthesis_array))
        
        # Phase space volume and complexity
        if len(synthesis_array) > 1:
            try:
                jacobian = self._calculate_jacobian_estimate(synthesis_array)
                det = np.linalg.det(jacobian)
                metrics['phase_volume'] = float(np.mean(np.abs(det))) if not np.isnan(det) else 0.0
                metrics['jacobian_norm'] = float(np.linalg.norm(jacobian))
            except:
                metrics['phase_volume'] = 0.0
                metrics['jacobian_norm'] = 0.0
        
        return metrics
    
    def _calculate_dialectical_metrics(self, synthesis_array):
        """Calculate dialectical process specific metrics."""
        if len(synthesis_array) < 2:
            return {}
        
        metrics = {}
        norms = np.linalg.norm(synthesis_array, axis=1)
        
        # Rate of change and derivatives
        differences = np.diff(synthesis_array, axis=0)
        change_rates = np.linalg.norm(differences, axis=1)
        
        metrics['mean_change_rate'] = float(np.mean(change_rates))
        metrics['std_change_rate'] = float(np.std(change_rates))
        metrics['max_change_rate'] = float(np.max(change_rates)) if len(change_rates) > 0 else 0.0
        
        # Acceleration (second derivative)
        if len(change_rates) > 1:
            acceleration = np.diff(change_rates)
            metrics['mean_acceleration'] = float(np.mean(acceleration))
            metrics['std_acceleration'] = float(np.std(acceleration))
        
        # Dialectical oscillation metrics
        if len(norms) > 3:
            # Find peaks in synthesis norm (potential transitions)
            try:
                peaks, properties = signal.find_peaks(norms, prominence=0.1)
                
                metrics['num_peaks'] = int(len(peaks))
                if len(peaks) > 0:
                    metrics['mean_peak_prominence'] = float(np.mean(properties['prominences']))
                    metrics['mean_peak_width'] = float(np.mean(properties['widths'])) if 'widths' in properties else 0.0
                
                # Period analysis
                if len(peaks) > 1:
                    periods = np.diff(peaks)
                    metrics['peak_period_mean'] = float(np.mean(periods))
                    metrics['peak_period_std'] = float(np.std(periods))
                    metrics['peak_regularity'] = float(metrics['peak_period_std'] / metrics['peak_period_mean']) if metrics['peak_period_mean'] > 0 else 0.0
            
            except:
                metrics['num_peaks'] = 0
        
        # Synthesis variability
        if np.mean(norms) > 0:
            metrics['synthesis_variability'] = float(np.var(norms) / np.mean(norms))
        else:
            metrics['synthesis_variability'] = 0.0
        
        # Dialectical activity measure
        metrics['dialectical_activity'] = float(np.sum(np.abs(change_rates)) / len(change_rates)) if len(change_rates) > 0 else 0.0
        
        # Entropy of changes
        if len(change_rates) > 0:
            hist, _ = np.histogram(change_rates, bins=min(20, len(change_rates)))
            hist = hist / np.sum(hist)
            metrics['change_entropy'] = float(-np.sum(hist * np.log(hist + 1e-10)))
        
        return metrics
    
    def _calculate_spectral_analysis(self, synthesis_array):
        """Perform spectral/frequency analysis."""
        if len(synthesis_array) < 50:
            return {}
        
        metrics = {}
        norms = np.linalg.norm(synthesis_array, axis=1)
        
        # Power spectral density
        try:
            nperseg = min(256, len(norms)//4)
            freqs, psd = signal.welch(norms, fs=1.0, nperseg=nperseg)
            
            # Dominant frequency
            dominant_idx = np.argmax(psd)
            metrics['dominant_frequency'] = float(freqs[dominant_idx]) if len(psd) > 0 else 0.0
            
            # Spectral entropy
            psd_norm = psd / np.sum(psd) if np.sum(psd) > 0 else psd
            metrics['spectral_entropy'] = float(-np.sum(psd_norm * np.log(psd_norm + 1e-10)))
            
            # Spectral moments
            metrics['psd_peak'] = float(np.max(psd)) if len(psd) > 0 else 0.0
            metrics['psd_mean'] = float(np.mean(psd)) if len(psd) > 0 else 0.0
            metrics['psd_std'] = float(np.std(psd)) if len(psd) > 0 else 0.0
            metrics['total_spectral_power'] = float(np.sum(psd)) if len(psd) > 0 else 0.0
            
            # Band power ratios
            if len(freqs) > 0:
                total_power = np.sum(psd)
                low_freq_mask = freqs <= 0.1
                high_freq_mask = freqs > 0.1
                
                if total_power > 0:
                    metrics['low_freq_power_ratio'] = float(np.sum(psd[low_freq_mask]) / total_power)
                    metrics['high_freq_power_ratio'] = float(np.sum(psd[high_freq_mask]) / total_power)
            
        except Exception as e:
            # Initialize metrics with zeros if spectral analysis fails
            spectral_metrics = ['dominant_frequency', 'spectral_entropy', 'psd_peak', 
                               'psd_mean', 'psd_std', 'total_spectral_power',
                               'low_freq_power_ratio', 'high_freq_power_ratio']
            for metric in spectral_metrics:
                metrics[metric] = 0.0
        
        return metrics
    
    def _calculate_attractor_analysis(self, synthesis_array):
        """Analyze attractor properties."""
        if len(synthesis_array) < 20:
            return {}
        
        metrics = {}
        
        # Phase space reconstruction (Takens embedding)
        embedded = self._takens_embedding(synthesis_array)
        
        if embedded is not None and len(embedded) > 0:
            # Attractor dimensions
            attractor_size = np.std(embedded, axis=0)
            metrics['attractor_size_mean'] = float(np.mean(attractor_size))
            metrics['attractor_size_std'] = float(np.std(attractor_size))
            metrics['attractor_size_max'] = float(np.max(attractor_size))
            
            # Attractor shape metrics
            cov_matrix = np.cov(embedded.T)
            eigenvalues = np.linalg.eigvals(cov_matrix)
            eigenvalues = np.sort(eigenvalues)[::-1]
            
            if len(eigenvalues) > 0 and np.sum(eigenvalues) > 0:
                metrics['attractor_eigenvalues'] = eigenvalues.tolist()
                metrics['attractor_variance_explained'] = (eigenvalues[0] / np.sum(eigenvalues)) if len(eigenvalues) > 0 else 0.0
                metrics['attractor_condition_number'] = float(eigenvalues[0] / eigenvalues[-1]) if eigenvalues[-1] > 0 else 0.0
        
        # Phase space coverage
        metrics['phase_space_coverage'] = float(self._calculate_coverage(synthesis_array))
        
        # Trajectory complexity
        metrics['trajectory_complexity'] = float(self._calculate_trajectory_complexity(synthesis_array))
        
        # Fractal dimension estimation
        metrics['fractal_dimension_estimate'] = float(self._estimate_fractal_dimension(synthesis_array))
        
        return metrics
    
    def _calculate_transition_analysis(self, transitions):
        """Analyze qualitative transitions."""
        metrics = {
            'num_transitions': 0,
            'transition_rate': 0.0,
            'mean_transition_magnitude': 0.0,
            'std_transition_magnitude': 0.0,
            'transition_regularity': 0.0,
            'transition_epochs': [],
            'transition_magnitudes': []
        }
        
        if not transitions:
            return metrics
        
        transition_epochs = [t['epoch'] for t in transitions]
        transition_magnitudes = [t['synthesis_norm'] for t in transitions]
        
        metrics['num_transitions'] = int(len(transitions))
        metrics['transition_rate'] = float(len(transitions) / self.system.epoch) if self.system.epoch > 0 else 0.0
        metrics['mean_transition_magnitude'] = float(np.mean(transition_magnitudes))
        metrics['std_transition_magnitude'] = float(np.std(transition_magnitudes))
        metrics['transition_epochs'] = transition_epochs
        metrics['transition_magnitudes'] = transition_magnitudes
        
        # Transition interval analysis
        if len(transition_epochs) > 1:
            intervals = np.diff(transition_epochs)
            if np.mean(intervals) > 0:
                metrics['transition_regularity'] = float(np.std(intervals) / np.mean(intervals))
                metrics['mean_transition_interval'] = float(np.mean(intervals))
                metrics['std_transition_interval'] = float(np.std(intervals))
        
        return metrics
    
    def _calculate_complexity_measures(self, synthesis_array):
        """Calculate additional complexity measures."""
        if len(synthesis_array) < 10:
            return {}
        
        metrics = {}
        norms = np.linalg.norm(synthesis_array, axis=1)
        
        # Approximate Entropy (ApEn)
        metrics['approximate_entropy'] = float(self._calculate_approximate_entropy(norms))
        
        # Sample Entropy (SampEn)
        metrics['sample_entropy'] = float(self._calculate_sample_entropy(norms))
        
        # Hurst exponent (roughness measure)
        metrics['hurst_exponent'] = float(self._calculate_hurst_exponent(norms))
        
        # Lempel-Ziv complexity
        metrics['lempel_ziv_complexity'] = float(self._calculate_lempel_ziv_complexity(norms))
        
        # Permutation entropy
        metrics['permutation_entropy'] = float(self._calculate_permutation_entropy(norms))
        
        return metrics
    
    def _calculate_approximate_entropy(self, time_series, m=2, r=None):
        """Calculate Approximate Entropy."""
        try:
            N = len(time_series)
            if N < 50:
                return 0.0
            
            if r is None:
                r = 0.2 * np.std(time_series)
            
            def _phi(m):
                x = np.array([time_series[i:i + m] for i in range(N - m + 1)])
                C = np.sum(np.max(np.abs(x[:, None] - x[None, :]), axis=2) <= r, axis=1) / (N - m + 1)
                return np.sum(np.log(C)) / (N - m + 1)
            
            return abs(_phi(m + 1) - _phi(m))
        except:
            return 0.0
    
    def _calculate_sample_entropy(self, time_series, m=2, r=None):
        """Calculate Sample Entropy."""
        try:
            N = len(time_series)
            if N < 50:
                return 0.0
            
            if r is None:
                r = 0.2 * np.std(time_series)
            
            def _count_matches(m):
                x = np.array([time_series[i:i + m] for i in range(N - m + 1)])
                distances = np.max(np.abs(x[:, None] - x[None, :]), axis=2)
                np.fill_diagonal(distances, np.inf)
                return np.sum(distances <= r) - (N - m + 1)  # subtract self-matches
            
            A = _count_matches(m + 1)
            B = _count_matches(m)
            
            if B > 0 and A > 0:
                return -np.log(A / B)
            return 0.0
        except:
            return 0.0
    
    def _calculate_hurst_exponent(self, time_series):
        """Calculate Hurst exponent using R/S analysis."""
        try:
            N = len(time_series)
            if N < 100:
                return 0.5
            
            # Rescaled Range Analysis
            max_lag = min(N // 4, 50)
            lags = range(10, max_lag)
            rs_values = []
            
            for lag in lags:
                # Calculate R/S for each lag
                rs = []
                for i in range(0, N - lag, lag):
                    segment = time_series[i:i + lag]
                    mean_segment = np.mean(segment)
                    deviations = segment - mean_segment
                    cumulative_dev = np.cumsum(deviations)
                    R = np.max(cumulative_dev) - np.min(cumulative_dev)
                    S = np.std(segment)
                    if S > 0:
                        rs.append(R / S)
                
                if rs:
                    rs_values.append(np.mean(rs))
            
            if len(rs_values) > 1:
                # Linear regression in log-log space
                log_lags = np.log(lags[:len(rs_values)])
                log_rs = np.log(rs_values)
                slope, _, _, _, _ = stats.linregress(log_lags, log_rs)
                return float(slope)
            
            return 0.5
        except:
            return 0.5
    
    def _calculate_lempel_ziv_complexity(self, time_series, threshold=None):
        """Calculate Lempel-Ziv complexity."""
        try:
            if threshold is None:
                threshold = np.median(time_series)
            
            # Binarize time series
            binary_series = ''.join(['1' if x > threshold else '0' for x in time_series])
            
            # Lempel-Ziv algorithm
            n = len(binary_series)
            c = 1
            i = 0
            j = 1
            
            while i + j <= n:
                if binary_series[i:i + j] in binary_series[0:i + j - 1]:
                    j += 1
                else:
                    c += 1
                    i += j
                    j = 1
            
            # Normalize complexity
            return float(c * np.log2(n) / n) if n > 0 else 0.0
        except:
            return 0.0
    
    def _calculate_permutation_entropy(self, time_series, order=3, delay=1):
        """Calculate Permutation Entropy."""
        try:
            n = len(time_series)
            if n < order * 10:
                return 0.0
            
            permutations = {}
            
            for i in range(n - (order - 1) * delay):
                # Extract pattern
                indices = list(range(i, i + order * delay, delay))
                pattern = time_series[indices]
                
                # Get permutation pattern
                sorted_indices = np.argsort(pattern)
                pattern_str = ''.join(str(x) for x in sorted_indices)
                
                permutations[pattern_str] = permutations.get(pattern_str, 0) + 1
            
            # Calculate entropy
            total = sum(permutations.values())
            entropy = 0.0
            for count in permutations.values():
                p = count / total
                entropy -= p * np.log(p)
            
            # Normalize by maximum entropy
            max_entropy = np.log(np.math.factorial(order))
            return float(entropy / max_entropy) if max_entropy > 0 else 0.0
        except:
            return 0.0
    
    def _estimate_lyapunov(self, trajectory, embedding_dim=3, tau=1):
        """Estimate largest Lyapunov exponent."""
        try:
            n = len(trajectory)
            if n < 50:
                return 0.0
            
            # Simple finite-time Lyapunov estimation
            distances = []
            for i in range(n-1):
                if i + tau < n:
                    dist = np.linalg.norm(trajectory[i+tau] - trajectory[i])
                    if dist > 0:
                        distances.append(dist)
            
            if len(distances) > 10:
                # Linear fit in log space
                times = np.arange(len(distances)) * tau
                log_dists = np.log(np.array(distances) + 1e-10)
                
                if len(times) > 1 and np.std(log_dists) > 0:
                    slope, _, _, _, _ = stats.linregress(times, log_dists)
                    return float(slope)
            
            return 0.0
        except:
            return 0.0
    
    def _estimate_correlation_dimension(self, trajectory, max_embedding=3):
        """Estimate correlation dimension."""
        try:
            n = len(trajectory)
            if n < 30:
                return 0.0
            
            # Simple correlation sum for small r
            r = 0.1 * np.std(trajectory)
            count = 0
            total = 0
            
            for i in range(n):
                for j in range(i+1, n):
                    dist = np.linalg.norm(trajectory[i] - trajectory[j])
                    if dist < r:
                        count += 1
                    total += 1
            
            if total > 0 and count > 0:
                C_r = count / total
                if C_r > 0 and r > 0:
                    return float(np.log(C_r) / np.log(r))
            
            return 0.0
        except:
            return 0.0
    
    def _estimate_fractal_dimension(self, trajectory):
        """Estimate fractal dimension using box-counting method."""
        try:
            n = len(trajectory)
            if n < 20:
                return 0.0
            
            # Normalize trajectory
            traj_norm = (trajectory - np.min(trajectory)) / (np.max(trajectory) - np.min(trajectory) + 1e-10)
            
            # Box sizes
            box_sizes = 2**np.arange(1, 6)  # Powers of 2
            counts = []
            
            for box_size in box_sizes:
                # Create grid
                grid = np.zeros((box_size, box_size, box_size))
                
                # Map points to grid
                for point in traj_norm:
                    x, y, z = (point[:3] * (box_size - 1)).astype(int)
                    grid[x, y, z] = 1
                
                counts.append(np.sum(grid > 0))
            
            # Linear regression in log-log space
            log_sizes = np.log(1 / box_sizes)
            log_counts = np.log(counts)
            
            if len(log_sizes) > 1 and np.std(log_counts) > 0:
                slope, _, _, _, _ = stats.linregress(log_sizes, log_counts)
                return float(slope)
            
            return 0.0
        except:
            return 0.0
    
    def _calculate_recurrence_rate(self, trajectory, threshold=0.1):
        """Calculate recurrence rate."""
        try:
            n = len(trajectory)
            if n < 10:
                return 0.0
            
            recurrence_count = 0
            total_pairs = 0
            
            for i in range(n):
                for j in range(i+1, n):  # Only upper triangle
                    if np.linalg.norm(trajectory[i] - trajectory[j]) < threshold:
                        recurrence_count += 2  # Symmetric
                    total_pairs += 1
            
            return float(recurrence_count / (n * n)) if n > 0 else 0.0
        except:
            return 0.0
    
    def _calculate_jacobian_estimate(self, trajectory):
        """Estimate local Jacobian from trajectory."""
        if len(trajectory) < 2:
            return np.eye(self.dimension)
        
        # Simple finite difference Jacobian estimate
        jacobian = np.zeros((self.dimension, self.dimension))
        count = 0
        
        for i in range(min(10, len(trajectory)-1)):
            delta_x = trajectory[i+1] - trajectory[i]
            norm_sq = np.dot(trajectory[i], trajectory[i])
            if norm_sq > 1e-10:
                jacobian += np.outer(delta_x, trajectory[i]) / norm_sq
                count += 1
        
        return jacobian / count if count > 0 else np.eye(self.dimension)
    
    def _takens_embedding(self, time_series, dim=3, tau=1):
        """Perform Takens time-delay embedding."""
        n = len(time_series)
        if n < dim * tau:
            return None
        
        embedded = []
        for i in range(n - (dim-1)*tau):
            point = []
            for j in range(dim):
                point.extend(time_series[i + j*tau])
            embedded.append(point)
        
        return np.array(embedded)
    
    def _calculate_coverage(self, trajectory):
        """Calculate phase space coverage."""
        if len(trajectory) < 2:
            return 0.0
        
        # Calculate convex hull volume (simplified)
        ranges = np.max(trajectory, axis=0) - np.min(trajectory, axis=0)
        coverage = np.prod(ranges)
        
        return float(coverage)
    
    def _calculate_trajectory_complexity(self, trajectory):
        """Calculate trajectory complexity measure."""
        if len(trajectory) < 3:
            return 0.0
        
        # Approximate path length
        diffs = np.diff(trajectory, axis=0)
        path_length = np.sum(np.linalg.norm(diffs, axis=1))
        
        # Straight line distance
        straight_distance = np.linalg.norm(trajectory[-1] - trajectory[0])
        
        # Complexity = path length / straight distance
        if straight_distance > 0:
            return float(path_length / straight_distance)
        return float(path_length)
    
    def _print_summary(self, results):
        """Print comprehensive analysis summary."""
        print("\n" + "=" * 80)
        print("ADVANCED ANALYSIS SUMMARY")
        print("=" * 80)
        
        print(f"\n📊 BASIC STATISTICS:")
        print(f"   • Mean synthesis norm: {results['mean_norm']:.4f} ± {results['std_norm']:.4f}")
        print(f"   • Range: [{results['min_norm']:.4f}, {results['max_norm']:.4f}]")
        print(f"   • Skewness: {results['skewness_norm']:.3f}, Kurtosis: {results['kurtosis_norm']:.3f}")
        
        print(f"\n🔄 DIALECTICAL METRICS:")
        print(f"   • Dialectical activity: {results.get('dialectical_activity', 0):.4f}")
        print(f"   • Mean change rate: {results.get('mean_change_rate', 0):.4f}")
        print(f"   • Synthesis variability: {results.get('synthesis_variability', 0):.4f}")
        
        print(f"\n🧠 COMPLEXITY MEASURES:")
        print(f"   • Approximate Entropy: {results.get('approximate_entropy', 0):.4f}")
        print(f"   • Sample Entropy: {results.get('sample_entropy', 0):.4f}")
        print(f"   • Hurst Exponent: {results.get('hurst_exponent', 0):.4f}")
        print(f"   • Permutation Entropy: {results.get('permutation_entropy', 0):.4f}")
        
        print(f"\n⚡ QUALITATIVE TRANSITIONS:")
        print(f"   • Number of transitions: {results['num_transitions']}")
        if results['num_transitions'] > 0:
            print(f"   • Transition rate: {results['transition_rate']:.4f} per epoch")
            print(f"   • Mean magnitude: {results['mean_transition_magnitude']:.4f}")
            print(f"   • Regularity: {results['transition_regularity']:.4f}")
        
        print(f"\n🌀 PHASE SPACE ANALYSIS:")
        print(f"   • Lyapunov estimate: {results.get('lyapunov_estimate', 0):.4f}")
        print(f"   • Correlation dimension: {results.get('correlation_dimension', 0):.3f}")
        print(f"   • Fractal dimension: {results.get('fractal_dimension_estimate', 0):.3f}")
        print(f"   • Recurrence rate: {results.get('recurrence_rate', 0):.4f}")
        
        print(f"\n📈 SPECTRAL ANALYSIS:")
        print(f"   • Dominant frequency: {results.get('dominant_frequency', 0):.4f}")
        print(f"   • Spectral entropy: {results.get('spectral_entropy', 0):.4f}")
        print(f"   • Low/High freq ratio: {results.get('low_freq_power_ratio', 0):.3f}/{results.get('high_freq_power_ratio', 0):.3f}")
        
        print(f"\n🎯 SYSTEM PROPERTIES:")
        print(f"   • Attractor size: {results.get('attractor_size_mean', 0):.4f}")
        print(f"   • Phase space coverage: {results.get('phase_space_coverage', 0):.4f}")
        print(f"   • Trajectory complexity: {results.get('trajectory_complexity', 0):.4f}")
        
        print("\n" + "=" * 80)
    
    def visualize_advanced_analysis(self, results, save_path=None):
        """
        Create comprehensive visualization of advanced analysis.
        
        Parameters:
        -----------
        results : dict
            Analysis results from run_comprehensive_analysis
        save_path : str, optional
            Path to save the figure
        """
        synthesis_array = results['synthesis_array']
        norms = np.linalg.norm(synthesis_array, axis=1)
        
        fig = plt.figure(figsize=(20, 16))
        fig.suptitle('ADVANCED XENOPOULOS SYSTEM ANALYSIS', 
                    fontsize=16, fontweight='bold', y=1.02)
        
        # 1. Synthesis Norm Evolution
        ax1 = plt.subplot(3, 4, 1)
        ax1.plot(norms, 'b-', alpha=0.7, linewidth=1)
        ax1.axhline(results['mean_norm'], color='r', linestyle='--', 
                   label=f'Mean: {results["mean_norm"]:.3f}')
        ax1.fill_between(range(len(norms)), 
                        results['mean_norm'] - results['std_norm'],
                        results['mean_norm'] + results['std_norm'],
                        alpha=0.2, color='gray')
        
        # Mark transitions
        if results['num_transitions'] > 0:
            transition_epochs = results['transition_epochs']
            transition_values = results['transition_magnitudes']
            ax1.scatter(transition_epochs, transition_values, 
                       color='gold', s=50, zorder=5, label='Transitions')
        
        ax1.set_title('Synthesis Norm Evolution')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('||Synthesis||')
        ax1.legend(fontsize=8)
        ax1.grid(True, alpha=0.3)
        
        # 2. Phase Space 3D
        ax2 = plt.subplot(3, 4, 2, projection='3d')
        if len(synthesis_array) > 10 and synthesis_array.shape[1] >= 3:
            ax2.plot(synthesis_array[:, 0], synthesis_array[:, 1], 
                    synthesis_array[:, 2], 'b-', alpha=0.6, linewidth=0.8)
            scatter = ax2.scatter(synthesis_array[:, 0], synthesis_array[:, 1], 
                                 synthesis_array[:, 2], c=norms, cmap='viridis', 
                                 s=10, alpha=0.6)
            plt.colorbar(scatter, ax=ax2, label='Norm')
        ax2.set_title('Phase Space Trajectory')
        ax2.set_xlabel('Component 1')
        ax2.set_ylabel('Component 2')
        ax2.set_zlabel('Component 3')
        
        # 3. Distribution Analysis
        ax3 = plt.subplot(3, 4, 3)
        ax3.hist(norms, bins=30, density=True, alpha=0.7, 
                color='darkorange', edgecolor='black')
        ax3.axvline(results['mean_norm'], color='red', linestyle='--', 
                   label=f'Mean: {results["mean_norm"]:.3f}')
        ax3.axvline(results['median_norm'], color='blue', linestyle=':', 
                   label=f'Median: {results["median_norm"]:.3f}')
        
        # Fit normal distribution
        try:
            x = np.linspace(results['min_norm'], results['max_norm'], 100)
            pdf = stats.norm.pdf(x, results['mean_norm'], results['std_norm'])
            ax3.plot(x, pdf, 'k-', linewidth=1.5, alpha=0.7, label='Normal Fit')
        except:
            pass
        
        ax3.set_title('Synthesis Norm Distribution')
        ax3.set_xlabel('Synthesis Norm')
        ax3.set_ylabel('Probability Density')
        ax3.legend(fontsize=8)
        ax3.grid(True, alpha=0.3)
        
        # 4. Component Correlation Matrix
        ax4 = plt.subplot(3, 4, 4)
        if synthesis_array.shape[1] > 1:
            corr_matrix = np.corrcoef(synthesis_array.T)
            im = ax4.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
            plt.colorbar(im, ax=ax4)
            
            # Add correlation values
            for i in range(self.dimension):
                for j in range(self.dimension):
                    ax4.text(j, i, f'{corr_matrix[i, j]:.2f}', 
                            ha='center', va='center', color='white', fontsize=9)
        
        ax4.set_title('Component Correlation Matrix')
        ax4.set_xticks(range(self.dimension))
        ax4.set_yticks(range(self.dimension))
        ax4.set_xticklabels([f'C{i+1}' for i in range(self.dimension)])
        ax4.set_yticklabels([f'C{i+1}' for i in range(self.dimension)])
        
        # 5. Rate of Change Analysis
        ax5 = plt.subplot(3, 4, 5)
        if len(synthesis_array) > 1:
            changes = np.diff(norms)
            ax5.plot(range(1,