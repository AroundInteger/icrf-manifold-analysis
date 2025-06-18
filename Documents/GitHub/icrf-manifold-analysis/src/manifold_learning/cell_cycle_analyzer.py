"""
Cell Cycle Manifold Analyzer
============================

Core module for manifold learning analysis of cell cycle flow cytometry data.
Implements the "polyploidy garage" framework for ICRF-193 studies.

Author: [Your Name]
Created: 2025
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import UMAP
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import seaborn as sns
from scipy.spatial.distance import pdist, squareform
from scipy.stats import gaussian_kde
from scipy.signal import find_peaks
import warnings
warnings.filterwarnings('ignore')


class CellCycleManifoldAnalyzer:
    """
    Manifold learning analysis for cell cycle flow cytometry data.
    
    This class implements the "polyploidy garage" model where cell cycle
    dynamics are visualized as a multi-story garage with different ploidy
    levels occupying different floors.
    
    Attributes:
        manifolds (dict): Storage for raw data and metadata
        embeddings (dict): Learned manifold embeddings
        population_labels (dict): Ploidy-based cell population assignments
        random_state (int): Random seed for reproducible results
    """
    
    def __init__(self, random_state=42):
        """
        Initialize the analyzer.
        
        Parameters:
            random_state (int): Random seed for reproducible results
        """
        self.random_state = random_state
        self.manifolds = {}
        self.embeddings = {}
        self.population_labels = {}
        
    def load_flow_data(self, gfp_data, draq5_data, timepoint_label):
        """
        Load and preprocess flow cytometry data.
        
        Parameters:
            gfp_data (array-like): GFP-Cyclin B1 fluorescence values
            draq5_data (array-like): DRAQ5 DNA content values
            timepoint_label (str): Label for this timepoint (e.g., '0hr', '24hr', '42hr')
            
        Returns:
            ndarray: Combined feature matrix [n_cells, 2]
        """
        # Convert to numpy arrays
        gfp_data = np.asarray(gfp_data)
        draq5_data = np.asarray(draq5_data)
        
        # Validate data
        if len(gfp_data) != len(draq5_data):
            raise ValueError("GFP and DRAQ5 data must have same length")
        
        # Remove any invalid values
        valid_mask = (~np.isnan(gfp_data)) & (~np.isnan(draq5_data))
        gfp_data = gfp_data[valid_mask]
        draq5_data = draq5_data[valid_mask]
        
        # Combine into feature matrix
        X = np.column_stack([gfp_data, draq5_data])
        
        # Store raw data
        self.manifolds[timepoint_label] = {
            'raw_data': X,
            'gfp': gfp_data,
            'draq5': draq5_data,
            'n_cells': len(gfp_data)
        }
        
        print(f"Loaded {len(gfp_data)} cells for timepoint {timepoint_label}")
        print(f"  GFP range: {gfp_data.min():.1f} - {gfp_data.max():.1f}")
        print(f"  DRAQ5 range: {draq5_data.min():.1f} - {draq5_data.max():.1f}")
        
        return X
    
    def segment_populations_garage(self, timepoint_label, auto_threshold=True, 
                                  manual_thresholds=None, min_peak_height=0.05):
        """
        Segment populations using "polyploidy garage" model.
        Each "floor" represents a different ploidy level.
        
        Parameters:
            timepoint_label (str): Which timepoint to segment
            auto_threshold (bool): Automatically detect ploidy levels using peak detection
            manual_thresholds (list): Manual DRAQ5 thresholds if auto_threshold=False
            min_peak_height (float): Minimum relative peak height for detection
            
        Returns:
            tuple: (labels, pop_fractions, thresholds)
        """
        if timepoint_label not in self.manifolds:
            raise ValueError(f"No data found for timepoint: {timepoint_label}")
            
        data = self.manifolds[timepoint_label]
        draq5 = data['draq5']
        
        if auto_threshold:
            # Detect ploidy levels automatically using histogram peaks
            # Create smooth histogram
            kde = gaussian_kde(draq5)
            x_range = np.linspace(draq5.min(), draq5.max(), 1000)
            density = kde(x_range)
            
            # Find peaks (ploidy levels)
            peaks, properties = find_peaks(density, 
                                         height=min_peak_height * density.max(), 
                                         distance=20)
            peak_positions = x_range[peaks]
            
            # Sort peaks and assign ploidy levels
            peak_positions = np.sort(peak_positions)
            print(f"\nDetected ploidy peaks at DRAQ5 levels: {peak_positions}")
            
            # Create thresholds between peaks
            if len(peak_positions) > 1:
                thresholds = []
                for i in range(len(peak_positions)-1):
                    # Threshold = midpoint between consecutive peaks
                    threshold = (peak_positions[i] + peak_positions[i+1]) / 2
                    thresholds.append(threshold)
            else:
                # Single peak - use quantile-based thresholding
                thresholds = [np.percentile(draq5, 75)]
                
        else:
            thresholds = manual_thresholds or [300, 500, 800]  # Default for 2N/4N/8N/16N
        
        # Assign cells to "garage floors"
        labels = np.zeros(len(draq5), dtype=int)
        for i, threshold in enumerate(thresholds):
            labels[draq5 >= threshold] = i + 1
        
        # Store labels and statistics
        self.population_labels[timepoint_label] = labels
        
        pop_counts = np.bincount(labels)
        pop_fractions = pop_counts / len(labels)
        
        # Create garage floor mapping
        max_floor = len(pop_counts) - 1
        garage_floors = {}
        for i in range(len(pop_counts)):
            if i == 0:
                garage_floors[i] = '2N (Ground)'
            else:
                ploidy = 2 ** (i + 1)
                garage_floors[i] = f'{ploidy}N (Floor {i})'
        
        print(f"\n{timepoint_label} Polyploidy Garage Distribution:")
        for i, count in enumerate(pop_counts):
            floor_name = garage_floors[i]
            print(f"  {floor_name}: {count} cells ({pop_fractions[i]:.1%})")
        
        # Store garage metadata
        self.manifolds[timepoint_label]['garage_floors'] = garage_floors
        self.manifolds[timepoint_label]['thresholds'] = thresholds
        self.manifolds[timepoint_label]['peak_positions'] = peak_positions if auto_threshold else None
            
        return labels, pop_fractions, thresholds
    
    def learn_manifold(self, timepoint_label, method='umap', **kwargs):
        """
        Learn manifold embedding for a given timepoint.
        
        Parameters:
            timepoint_label (str): Which timepoint to analyze
            method (str): 'umap', 'pca', or 'diffusion_map'
            **kwargs: Parameters for the manifold learning method
            
        Returns:
            ndarray: Learned embedding coordinates
        """
        if timepoint_label not in self.manifolds:
            raise ValueError(f"No data found for timepoint: {timepoint_label}")
            
        X = self.manifolds[timepoint_label]['raw_data']
        
        # Standardize data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        if method == 'umap':
            # UMAP parameters optimized for cell cycle data
            default_params = {
                'n_neighbors': 50,  # Larger neighborhood for smooth manifolds
                'min_dist': 0.1,    # Allow close packing
                'n_components': 2,  # 2D for visualization
                'metric': 'euclidean',
                'random_state': self.random_state
            }
            default_params.update(kwargs)
            
            reducer = UMAP(**default_params)
            embedding = reducer.fit_transform(X_scaled)
            
        elif method == 'pca':
            default_params = {'n_components': 2, 'random_state': self.random_state}
            default_params.update(kwargs)
            
            reducer = PCA(**default_params)
            embedding = reducer.fit_transform(X_scaled)
            
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Store results
        self.embeddings[timepoint_label] = {
            'embedding': embedding,
            'method': method,
            'reducer': reducer,
            'scaler': scaler,
            'parameters': default_params if method in ['umap', 'pca'] else kwargs
        }
        
        print(f"Learned {method} manifold for {timepoint_label}")
        return embedding
    
    def compute_manifold_metrics(self, timepoint_label):
        """
        Compute quantitative metrics for manifold structure.
        
        Parameters:
            timepoint_label (str): Which timepoint to analyze
            
        Returns:
            dict: Computed metrics
        """
        if timepoint_label not in self.embeddings:
            raise ValueError(f"No embedding found for {timepoint_label}")
            
        embedding = self.embeddings[timepoint_label]['embedding']
        
        # Compute pairwise distances in embedding space
        distances = pdist(embedding)
        
        # Basic manifold metrics
        metrics = {
            'mean_distance': np.mean(distances),
            'std_distance': np.std(distances),
            'diameter': np.max(distances),
            'compactness': np.std(distances) / np.mean(distances),
            'n_cells': len(embedding)
        }
        
        # If we have population labels, compute separation metrics
        if timepoint_label in self.population_labels:
            labels = self.population_labels[timepoint_label]
            unique_labels = np.unique(labels)
            
            if len(unique_labels) > 1:
                # Compute inter-population distances
                inter_distances = []
                for i in range(len(unique_labels)):
                    for j in range(i+1, len(unique_labels)):
                        pop_i = embedding[labels == unique_labels[i]]
                        pop_j = embedding[labels == unique_labels[j]]
                        
                        if len(pop_i) > 0 and len(pop_j) > 0:
                            # Mean distance between population centroids
                            centroid_i = np.mean(pop_i, axis=0)
                            centroid_j = np.mean(pop_j, axis=0)
                            inter_distances.append(np.linalg.norm(centroid_i - centroid_j))
                
                metrics['population_separation'] = np.mean(inter_distances) if inter_distances else 0
                metrics['n_populations'] = len(unique_labels)
        
        return metrics
    
    def visualize_manifold(self, timepoint_label, color_by='ploidy', figsize=(12, 5)):
        """
        Visualize the learned manifold.
        
        Parameters:
            timepoint_label (str): Which timepoint to visualize
            color_by (str): 'ploidy', 'gfp', 'draq5', or 'density'
            figsize (tuple): Figure size
            
        Returns:
            matplotlib.figure.Figure: The generated figure
        """
        if timepoint_label not in self.embeddings:
            raise ValueError(f"No embedding found for {timepoint_label}")
            
        embedding = self.embeddings[timepoint_label]['embedding']
        data = self.manifolds[timepoint_label]
        
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # Original space
        ax1 = axes[0]
        if color_by == 'ploidy' and timepoint_label in self.population_labels:
            labels = self.population_labels[timepoint_label]
            scatter = ax1.scatter(data['draq5'], data['gfp'], 
                                c=labels, cmap='viridis', alpha=0.6, s=1)
            plt.colorbar(scatter, ax=ax1, label='Population')
        else:
            ax1.scatter(data['draq5'], data['gfp'], alpha=0.6, s=1, c='blue')
            
        ax1.set_xlabel('DRAQ5 (DNA Content)')
        ax1.set_ylabel('GFP-Cyclin B1')
        ax1.set_title(f'Original Space - {timepoint_label}')
        
        # Manifold space
        ax2 = axes[1]
        if color_by == 'ploidy' and timepoint_label in self.population_labels:
            labels = self.population_labels[timepoint_label]
            scatter = ax2.scatter(embedding[:, 0], embedding[:, 1], 
                                c=labels, cmap='viridis', alpha=0.6, s=1)
            plt.colorbar(scatter, ax=ax2, label='Population')
        elif color_by == 'gfp':
            scatter = ax2.scatter(embedding[:, 0], embedding[:, 1], 
                                c=data['gfp'], cmap='plasma', alpha=0.6, s=1)
            plt.colorbar(scatter, ax=ax2, label='GFP-Cyclin B1')
        elif color_by == 'draq5':
            scatter = ax2.scatter(embedding[:, 0], embedding[:, 1], 
                                c=data['draq5'], cmap='plasma', alpha=0.6, s=1)
            plt.colorbar(scatter, ax=ax2, label='DRAQ5')
        else:
            ax2.scatter(embedding[:, 0], embedding[:, 1], alpha=0.6, s=1, c='red')
            
        ax2.set_xlabel('Manifold Dimension 1')
        ax2.set_ylabel('Manifold Dimension 2')
        ax2.set_title(f'Manifold Space - {timepoint_label}')
        
        plt.tight_layout()
        plt.show()
        
        return fig
    
    def visualize_polyploidy_garage(self, timepoint_label, figsize=(15, 10)):
        """
        Visualize the "polyploidy garage" concept with floor-by-floor analysis.
        
        Parameters:
            timepoint_label (str): Which timepoint to visualize
            figsize (tuple): Figure size
            
        Returns:
            tuple: (figure, floor_statistics)
        """
        if timepoint_label not in self.manifolds:
            raise ValueError(f"No data found for {timepoint_label}")
            
        data = self.manifolds[timepoint_label]
        labels = self.population_labels.get(timepoint_label, None)
        embedding = self.embeddings.get(timepoint_label, {}).get('embedding', None)
        garage_floors = data.get('garage_floors', {})
        
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        
        # Plot 1: Original data with garage floor coloring
        ax = axes[0, 0]
        if labels is not None:
            unique_labels = np.unique(labels)
            colors = plt.cm.viridis(np.linspace(0, 1, len(unique_labels)))
            
            for i, label in enumerate(unique_labels):
                mask = labels == label
                floor_name = garage_floors.get(label, f'Floor {label}')
                ax.scatter(data['draq5'][mask], data['gfp'][mask], 
                          c=[colors[i]], alpha=0.6, s=1, label=floor_name)
            ax.legend()
        else:
            ax.scatter(data['draq5'], data['gfp'], alpha=0.6, s=1, c='blue')
        
        ax.set_xlabel('DRAQ5 (DNA Content)')
        ax.set_ylabel('GFP-Cyclin B1')
        ax.set_title(f'Polyploidy Garage - {timepoint_label}')
        
        # Plot 2: Manifold embedding with garage floors (if available)
        ax = axes[0, 1]
        if embedding is not None and labels is not None:
            unique_labels = np.unique(labels)
            colors = plt.cm.viridis(np.linspace(0, 1, len(unique_labels)))
            
            for i, label in enumerate(unique_labels):
                mask = labels == label
                floor_name = garage_floors.get(label, f'Floor {label}')
                ax.scatter(embedding[mask, 0], embedding[mask, 1], 
                          c=[colors[i]], alpha=0.6, s=1, label=floor_name)
            ax.legend()
            ax.set_xlabel('Manifold Dimension 1')
            ax.set_ylabel('Manifold Dimension 2')
            ax.set_title(f'Manifold Space - {timepoint_label}')
        else:
            ax.text(0.5, 0.5, 'No manifold\nembedding available', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Manifold Space (Not Available)')
        
        # Plot 3: DRAQ5 histogram with floor boundaries
        ax = axes[0, 2]
        ax.hist(data['draq5'], bins=50, alpha=0.7, color='lightblue', edgecolor='black')
        
        # Add floor boundary lines
        if 'thresholds' in data:
            for i, threshold in enumerate(data['thresholds']):
                ax.axvline(threshold, color='red', linestyle='--', 
                          label=f'Floor {i+1} boundary' if i == 0 else '')
        
        ax.set_xlabel('DRAQ5 (DNA Content)')
        ax.set_ylabel('Cell Count')
        ax.set_title('DNA Content Distribution')
        if 'thresholds' in data:
            ax.legend()
        
        # Plot 4: Floor-by-floor cycle analysis
        ax = axes[1, 0]
        floor_stats = []
        if labels is not None:
            unique_labels = np.unique(labels)
            colors = plt.cm.viridis(np.linspace(0, 1, len(unique_labels)))
            
            for label in unique_labels:
                mask = labels == label
                floor_gfp = data['gfp'][mask]
                floor_draq5 = data['draq5'][mask]
                
                # Normalize DRAQ5 within each floor (relative cell cycle position)
                if len(floor_draq5) > 10:  # Only if enough cells
                    draq5_range = floor_draq5.max() - floor_draq5.min()
                    if draq5_range > 0:
                        draq5_norm = (floor_draq5 - floor_draq5.min()) / draq5_range
                        ax.scatter(draq5_norm, floor_gfp, c=[colors[label]], alpha=0.5, s=1,
                                  label=garage_floors.get(label, f'Floor {label}'))
                    
                    floor_stats.append({
                        'floor': label,
                        'n_cells': len(floor_gfp),
                        'gfp_mean': np.mean(floor_gfp),
                        'gfp_std': np.std(floor_gfp),
                        'draq5_range': draq5_range
                    })
            
            ax.legend()
        
        ax.set_xlabel('Normalized Cell Cycle Position (within floor)')
        ax.set_ylabel('GFP-Cyclin B1')
        ax.set_title('Cell Cycle Patterns by Floor')
        
        # Plot 5: Floor statistics
        ax = axes[1, 1]
        if floor_stats:
            floors = [s['floor'] for s in floor_stats]
            n_cells = [s['n_cells'] for s in floor_stats]
            floor_names = [garage_floors.get(f, f'Floor {f}') for f in floors]
            
            bars = ax.bar(range(len(floor_names)), n_cells, 
                         color=colors[:len(floors)], alpha=0.7)
            ax.set_xticks(range(len(floor_names)))
            ax.set_xticklabels(floor_names, rotation=45, ha='right')
            ax.set_ylabel('Number of Cells')
            ax.set_title('Population per Floor')
            
            # Add percentage labels on bars
            total_cells = sum(n_cells)
            for i, (bar, count) in enumerate(zip(bars, n_cells)):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + total_cells*0.01,
                       f'{count/total_cells:.1%}', ha='center', va='bottom')
        
        # Plot 6: Garage structure overview
        ax = axes[1, 2]
        if labels is not None:
            unique_labels = np.unique(labels)
            colors = plt.cm.viridis(np.linspace(0, 1, len(unique_labels)))
            
            for i, label in enumerate(unique_labels):
                mask = labels == label
                ax.scatter(data['draq5'][mask], data['gfp'][mask], 
                          c=[colors[i]], alpha=0.6, s=1,
                          label=garage_floors.get(label, f'Floor {label}'))
            
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        else:
            ax.scatter(data['draq5'], data['gfp'], alpha=0.6, s=1, c='blue')
        
        ax.set_xlabel('DRAQ5 (DNA Content)')
        ax.set_ylabel('GFP-Cyclin B1')
        ax.set_title('Garage Structure Overview')
        
        plt.tight_layout()
        plt.show()
        
        return fig, floor_stats
    
    def compare_timepoints(self, reference_timepoint, comparison_timepoints):
        """
        Compare manifold structure across timepoints.
        
        Parameters:
            reference_timepoint (str): Baseline timepoint for comparison
            comparison_timepoints (list): List of timepoints to compare against reference
            
        Returns:
            dict: Comparison results and metrics
        """
        if reference_timepoint not in self.embeddings:
            raise ValueError(f"Reference timepoint {reference_timepoint} not found")
            
        results = {}
        
        # Get reference metrics
        ref_metrics = self.compute_manifold_metrics(reference_timepoint)
        results[reference_timepoint] = ref_metrics
        
        print(f"\nManifold Comparison (Reference: {reference_timepoint})")
        print("-" * 50)
        print(f"Reference - Cells: {ref_metrics['n_cells']}, "
              f"Compactness: {ref_metrics['compactness']:.3f}")
        
        for tp in comparison_timepoints:
            if tp in self.embeddings:
                metrics = self.compute_manifold_metrics(tp)
                results[tp] = metrics
                
                # Compute changes relative to reference
                compactness_change = (metrics['compactness'] - ref_metrics['compactness']) / ref_metrics['compactness']
                diameter_change = (metrics['diameter'] - ref_metrics['diameter']) / ref_metrics['diameter']
                
                print(f"\n{tp}:")
                print(f"  Cells: {metrics['n_cells']}")
                print(f"  Compactness change: {compactness_change:.1%}")
                print(f"  Diameter change: {diameter_change:.1%}")
                
                if 'population_separation' in metrics:
                    print(f"  Population separation: {metrics['population_separation']:.3f}")
                if 'n_populations' in metrics:
                    print(f"  Number of populations: {metrics['n_populations']}")
        
        return results


def analyze_icrf_experiment(data_0hr, data_24hr, data_42hr):
    """
    Complete analysis pipeline for ICRF-193 experiment.
    
    Parameters:
        data_0hr, data_24hr, data_42hr (dict): Each should contain 'gfp' and 'draq5' arrays
        
    Returns:
        tuple: (analyzer, comparison_results)
    """
    
    analyzer = CellCycleManifoldAnalyzer()
    
    # Load data for all timepoints
    print("Loading flow cytometry data...")
    analyzer.load_flow_data(data_0hr['gfp'], data_0hr['draq5'], '0hr')
    analyzer.load_flow_data(data_24hr['gfp'], data_24hr['draq5'], '24hr')
    analyzer.load_flow_data(data_42hr['gfp'], data_42hr['draq5'], '42hr')
    
    # Segment populations
    print("\nSegmenting cell populations...")
    for tp in ['0hr', '24hr', '42hr']:
        analyzer.segment_populations_garage(tp)
    
    # Learn manifolds
    print("\nLearning manifold embeddings...")
    for tp in ['0hr', '24hr', '42hr']:
        analyzer.learn_manifold(tp, method='umap')
    
    # Visualize results
    print("\nGenerating visualizations...")
    for tp in ['0hr', '24hr', '42hr']:
        analyzer.visualize_manifold(tp, color_by='ploidy')
        analyzer.visualize_polyploidy_garage(tp)
    
    # Compare timepoints
    print("\nComparing manifold structure across timepoints...")
    comparison = analyzer.compare_timepoints('0hr', ['24hr', '42hr'])
    
    return analyzer, comparison


def quick_analysis_single_timepoint(gfp_data, draq5_data, timepoint_label="sample"):
    """
    Quick analysis for a single timepoint - useful for testing.
    
    Parameters:
        gfp_data (array-like): GFP-Cyclin B1 values
        draq5_data (array-like): DRAQ5 DNA content values
        timepoint_label (str): Label for this dataset
        
    Returns:
        CellCycleManifoldAnalyzer: Configured analyzer with results
    """
    
    analyzer = CellCycleManifoldAnalyzer()
    
    # Load and analyze data
    print(f"Analyzing {timepoint_label} data...")
    analyzer.load_flow_data(gfp_data, draq5_data, timepoint_label)
    
    # Segment populations
    labels, fractions, thresholds = analyzer.segment_populations_garage(timepoint_label)
    
    # Learn manifold
    embedding = analyzer.learn_manifold(timepoint_label, method='umap')
    
    # Compute metrics
    metrics = analyzer.compute_manifold_metrics(timepoint_label)
    
    # Visualize
    analyzer.visualize_manifold(timepoint_label, color_by='ploidy')
    analyzer.visualize_polyploidy_garage(timepoint_label)
    
    print(f"\nAnalysis complete for {timepoint_label}!")
    print(f"Manifold metrics: {metrics}")
    
    return analyzer


# Example usage and testing functions
if __name__ == "__main__":
    # This will run when the module is executed directly
    print("Cell Cycle Manifold Analyzer Module")
    print("===================================")
    print("Import this module and use:")
    print("  from cell_cycle_analyzer import CellCycleManifoldAnalyzer")
    print("  analyzer = CellCycleManifoldAnalyzer()")
    print("  analyzer.load_flow_data(gfp, draq5, 'timepoint')")
    print("  # ... continue with analysis")